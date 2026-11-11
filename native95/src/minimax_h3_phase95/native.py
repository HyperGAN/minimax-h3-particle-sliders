"""Full native H3 phase runtime; original single-device backend is untouched."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
from minimax_h3_native95.native import (
    PikaH3FlexBackend as OriginalBackend, H3FlexUnavailable, PreparedH3Flex,
    _verified_adapter, _cleanup_preserving_primary, _raise_cleanup_errors,
    _attempt_cleanups, request_frames, prepared_frame_count, frame_metadata,
    MODEL_ID, MODEL_REVISION, DIFFUSERS_REVISION, PIKA_ADAPTER, SCHEDULE,
)
from .contracts import RELEASE, SOURCE_PROFILE, file_hash
from .placement import Placement, OriginalDiTPlacement, construction_rng
from .conditioning import (ValidatedSnapshot, Conditions, selected_pipeline,
                           subpipeline, close_pipeline)


class PikaH3FlexBackend(OriginalBackend):
    """Complete DiT/fixed Pika, resident24/26 stages and one refiner/device, FP32 game Parameters0.

    No TE or VAE is loaded. Full original CPU-encoded conditions are mandatory.
    Device copies preserve differentiability, dtypes, indices and coordinates.
    No model operation or solver is replaced.
    """
    def __init__(self, snapshot, *, conditioning, conditioning_sha256,
                 device='cuda:0', rank=8, alpha=8., source=None):
        self.plan = Placement().validate()  # Refuse before any CUDA call.
        if device!='cuda:0' or rank!=8 or alpha!=8.:
            raise ValueError('exact declared game0/rank8/alpha8 phase profile required')
        if torch.cuda.device_count()!=2:
            raise ValueError('two genuinely admitted visible GPUs required')
        self.device, self.entry_device = torch.device(self.plan.game_device), torch.device(self.plan.entry_device)
        self.pipe = self.transformer = self.network = self._placement = self._ff_chunks = None
        self._guard_handles, self._closed = [], False
        self.source = ValidatedSnapshot(snapshot) if source is None else source
        if self.source.root != Path(snapshot).resolve():
            raise ValueError('one exact original snapshot required across phases')
        self.conditions = Conditions(conditioning, conditioning_sha256)
        try:
            with construction_rng((0,1)):
                self.pipe = selected_pipeline(self.source, ['transformer','scheduler','audio_scheduler'], dtype=torch.bfloat16)
                state, digest = _verified_adapter(self.source.root)
                self.pipe.load_lora_weights(state, adapter_name=PIKA_ADAPTER)
                del state
                self.pipe._pika_verified_adapter_sha256 = digest
                self.pipe.set_adapters([PIKA_ADAPTER], adapter_weights=[1.])
                self.pipe.register_to_config(denoising_step_list=list(SCHEDULE))
                self.transformer = self.pipe.transformer
                if (self.transformer.config.hidden_size!=5376 or self.transformer.config.text_dim!=5120
                        or self.transformer.config.num_layers!=50 or self.transformer.config.num_refiner_layers!=2):
                    raise H3FlexUnavailable('complete original H3 architecture required')
                self._freeze_base()
                self._placement = OriginalDiTPlacement(self.transformer, self.plan)
                torch.default_generator.manual_seed(0)
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(0)
                self._attach_slider(rank=rank, alpha=alpha)
            from .feed_forward_chunks import OriginalFeedForwardChunks
            self._ff_chunks = OriginalFeedForwardChunks(self.transformer)
            self._trainable_identity = tuple(self.network.parameters())
            self._check_placement(); self.source.check()
        except BaseException as primary:
            _cleanup_preserving_primary(primary, self.close)
            raise

    def _build_slider(self, *, rank, alpha):
        from .adapter import HNDLSlider
        return HNDLSlider(self.transformer, rank=rank, alpha=alpha, game_device=self.plan.game_device)

    def _check_placement(self):
        self._placement.assert_parameters()
        if self._ff_chunks is not None:
            self._ff_chunks.assert_owned()
        current = tuple(self.network.parameters())
        if (len(current)!=len(self._trainable_identity)
                or any(p is not old for p,old in zip(current,self._trainable_identity))
                or any(p.device!=self.device or p.dtype!=torch.float32 for p in current)):
            raise H3FlexUnavailable('original FP32 slider Parameters must remain on one game device')

    def _preparation(self, prompt, duration, resolution, seed, *, complete):
        if (not isinstance(prompt,str) or not prompt.strip() or len(prompt)>8000
                or type(seed)is not int or not 0<=seed<2**63-100):
            raise ValueError('complete native prompt and explicit seed required')
        if resolution not in ('480p','768p'):
            raise ValueError('original full480p/768p geometry required')
        self._check_fixed_pika(); self._check_placement(); self.source.check()
        names = ['denoise.no_keyframe_anchors','denoise.prepare_layout',
                 'denoise.prepare_latents','denoise.set_timesteps']
        if complete:
            names += ['denoise.denoise','denoise.after_denoise']
        child = subpipeline(self.pipe, names)
        try:
            height,width = RELEASE['packed_geometry']['canvases'][resolution]
            # Both initial noise and original audio/video re-noise consume the
            # SAME isolated original CPU generator, in the official draw order.
            # Official packing constructs integer tags/FP64 coordinates on CPU;
            # its encoder emits CPU text tags even when embeddings are CUDA.
            # Retained embeddings are an intermediate for the preparation-only
            # child, so seed PipelineState rather than passing a filtered kwarg.
            from diffusers.modular_pipelines.modular_pipeline import PipelineState
            saved = self.conditions.get(prompt, 'cpu')
            state = PipelineState()
            state.set('prompt_embeds', saved['prompt_embeds'].to(device=self.entry_device))
            state.set('text_token_tags', saved['text_token_tags'])
            return child(state=state,
                height=height,width=width,num_frames=request_frames(duration),
                generator=torch.Generator(device='cpu').manual_seed(seed))
        finally:
            # This clears child registrations, not the objects still owned by
            # the main runtime. No actual base-memory release is claimed here.
            close_pipeline(child)

    def prepare_t2va(self, prompt, *, duration=5, resolution='480p', seed=7):
        from minimax_h3_native95.native import PreparedH3Flex
        state = self._preparation(prompt,duration,resolution,seed,complete=False)
        # This component-only preparation chain has no model owner from which
        # ModularPipeline can infer CUDA. Preserve its exact CPU-generator draw
        # order and native layout arithmetic, then stage ALL model inputs on the
        # declared entry device without changing values, shape or dtype.
        fields = {name:state.get(name).to(device=self.entry_device) for name in
                  ('position_ids','token_tags','video_indices','audio_indices','text_indices')}
        times,indices = (value.to(device=self.entry_device) for value in state.get('row_timestep_plan')[0])
        return PreparedH3Flex(frame_metadata=frame_metadata(request_frames(duration),state.get('num_frames')),
            hidden_states=state.get('latents').to(device=self.entry_device)[None],
            audio_hidden_states=state.get('audio_latents').to(device=self.entry_device)[None],
            encoder_hidden_states=state.get('prompt_embeds').to(device=self.entry_device),
            timestep=times,timestep_indices=indices,**fields)

    def velocity(self, packed):
        self._check_placement()
        return super().velocity(packed)

    @torch.no_grad()
    def render_latents(self, prompt, *, strength=0., duration=5, resolution='480p', seed=7):
        """INTERNAL complete denoise output; explicitly NOT finished media."""
        from .render import scheduler_state
        with scheduler_state(self.pipe), self.scaled(strength):
            state = self._preparation(prompt,duration,resolution,seed,complete=True)
        frames = state.get('num_frames')
        if frames!=prepared_frame_count(request_frames(duration)):
            raise H3FlexUnavailable('complete original frame grid required')
        video,audio = state.get('latents'),state.get('audio_latents')
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise H3FlexUnavailable('complete finite video AND stereo-audio latents required')
        return dict(schema='h3-phase-owned-native-AV-latents-NOT-finished-media-v1',
            latents=video.detach().cpu().clone(),audio_latents=audio.detach().cpu().clone(),num_frames=frames,
            frame_metadata=frame_metadata(request_frames(duration),frames),selection='raw',ema_decay=None)

    def render(self, *args, **kwargs):
        raise H3FlexUnavailable('use phase render_pair; latent output is not finished native AV')

    @property
    def identity(self):
        return dict(super().identity,source_profile=SOURCE_PROFILE,placement=dict(self.plan.__dict__),
            feed_forward_execution=self._ff_chunks.witness(),
            phase_sources={name:file_hash(Path(__file__).with_name(name))
                           for name in ('native.py','placement.py','conditioning.py','adapter.py','phase-native-sources.json',
                                        'grouped_checkpoint.py','grouped_forward.py','grouped-runtime-sources.json','feed_forward_chunks.py')})

    def close(self):
        if getattr(self,'_closed',False):
            return
        actions=[]
        if self.network is not None:
            actions.append(('slider forwards',self.network.close))
        if self._ff_chunks is not None:
            actions.append(('original FF chunk dispatch',self._ff_chunks.close))
        actions.append(('fixed Pika guards',self._restore_guards))
        if self._placement is not None:
            actions.append(('native device hooks',self._placement.close))
        _attempt_cleanups(*actions)  # Unresolved owners retain a real cleanup failure.
        pipe=self.pipe
        self.pipe=self.transformer=self.network=self._placement=self._ff_chunks=None
        self._trainable_identity=(); self.conditions=None
        self._release_layers={}  # Original PEFT aliases otherwise retain most DiT weights.
        self._attach_state=None
        close_pipeline(pipe)
        self._closed=True


@torch.no_grad()
def decode_latents(snapshot, latent_result, *, device='cuda:0', check=lambda:None, source=None):
    """After the DiT owner closes, decode using BOTH complete original FP32 VAEs."""
    if (latent_result.get('schema')!='h3-phase-owned-native-AV-latents-NOT-finished-media-v1'
            or latent_result.get('selection')!='raw' or latent_result.get('ema_decay')is not None):
        raise ValueError('exact complete native latent phase required')
    source=ValidatedSnapshot(snapshot) if source is None else source
    video_pipe=audio_pipe=child_video=child_audio=None
    try:
        check()
        with construction_rng(() if torch.device(device).type=='cpu' else (0,1)):
            video_pipe=selected_pipeline(source,['vae'],dtype=torch.float32)
            video_pipe.vae.requires_grad_(False).eval().to(device=device)
        child_video=subpipeline(video_pipe,['decode.video'])
        video=child_video(latents=latent_result['latents'].to(device=device,dtype=torch.float32),output_type='pil').get('videos')
        close_pipeline(child_video); child_video=None
        close_pipeline(video_pipe); video_pipe=None
        check()
        with construction_rng(() if torch.device(device).type=='cpu' else (0,1)):
            audio_pipe=selected_pipeline(source,['audio_vae'],dtype=torch.float32)
            audio_pipe.audio_vae.requires_grad_(False).eval().to(device=device)
        child_audio=subpipeline(audio_pipe,['decode.audio'])
        result=child_audio(audio_latents=latent_result['audio_latents'].to(device=device,dtype=torch.float32),
                           num_frames=latent_result['num_frames'])
        audio,rate=result.get('audio'),result.get('sampling_rate')
        if (not isinstance(video,list) or len(video)!=1 or len(video[0])!=latent_result['num_frames']
                or audio.shape[:2]!=(1,2) or not torch.isfinite(audio).all() or rate!=32000):
            raise H3FlexUnavailable('complete original video AND native stereo waveform required')
        return dict(videos=video,audio=audio.detach().cpu().clone(),sampling_rate=rate,
                    frame_metadata=deepcopy(latent_result['frame_metadata']))
    finally:
        from particle_sliders.restoration import _restore_all
        _restore_all([lambda:close_pipeline(child_video),lambda:close_pipeline(video_pipe),
                      lambda:close_pipeline(child_audio),lambda:close_pipeline(audio_pipe)],primary=sys.exception())
