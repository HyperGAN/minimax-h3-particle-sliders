"""Complete original no-grad TE phase; no truncated model or teacher replay."""
from pathlib import Path
import gc
import sys
import json
import torch
from particle_sliders.lifecycle import atomic_save
from minimax_h3_native95.native import validate_snapshot, require_native_runtime
from .contracts import RELEASE, digest, file_hash, record
from .placement import PROFILE, construction_rng

SCHEMA = 'h3-complete-Qwen3VL-hidden50-phase-conditioning-v1'


def require_phase_native_runtime():
    """Retain every original native guard and bind the newly used leaf APIs."""
    require_native_runtime()
    from importlib.metadata import version
    import diffusers
    import transformers
    pins=json.loads(Path(__file__).with_name('phase-native-sources.json').read_text())
    if (pins['diffusers_revision']!=RELEASE['diffusers_revision']
            or pins['transformers_version']!='5.22.1' or version('transformers')!='5.22.1'):
        raise ValueError('exact original H3 Diffusers and full Qwen3VL runtime required')
    roots={'diffusers':Path(diffusers.__file__).parent,'transformers':Path(transformers.__file__).parent}
    for name,pin in pins['files'].items():
        package,relative=name.split('/',1);path=roots[package]/relative
        if path.stat().st_size!=pin['bytes'] or file_hash(path)!=pin['sha256']:
            raise ValueError('additional phase-native operative Source differs: '+name)


def model_identity():
    return dict(model=RELEASE['model_id'], revision=RELEASE['model_revision'],
                release_sha256=digest(RELEASE), diffusers_revision=RELEASE['diffusers_revision'],
                text_encoder_layer=50, full_language_layers=64,
                tokenization='verbatim prompt; add_special_tokens=False; no truncation or chat template')


class ValidatedSnapshot:
    """One genuine original file validation, retained across owned phases."""
    def __init__(self, snapshot):
        require_phase_native_runtime()
        self.root = validate_snapshot(snapshot)
        paths = list(RELEASE['local_snapshot_files']['files']) + ['fl2va/adapter.safetensors', 'fl2va/manifest.json']
        self._stats = {name:self._stat(self.root/name) for name in paths}

    @staticmethod
    def _stat(path):
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def check(self):
        if any(self._stat(self.root/name) != saved for name, saved in self._stats.items()):
            raise RuntimeError('original validated model changed between native phases')


def selected_pipeline(source, names, *, dtype):
    """Original local d0b9 selected-component API; do not load all components."""
    from diffusers import ModularPipeline
    source.check()
    pipe = ModularPipeline.from_pretrained(str(source.root), workflow='t2va', local_files_only=True)
    pipe.load_components(names=names, pretrained_model_name_or_path=str(source.root),
                         local_files_only=True, dtype=dtype)
    if any(getattr(pipe, name, None) is None for name in names):
        # The official method logs component exceptions rather than raising.
        pipe.unload_components(names)
        raise RuntimeError('a required original phase component failed loading: '+','.join(names))
    source.check()
    return pipe


def subpipeline(pipe, names):
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    blocks = SequentialPipelineBlocks.from_blocks_dict({name:pipe.blocks.sub_blocks[name] for name in names})
    child = blocks.init_pipeline()
    child.update_components(**{name:value for name,value in pipe.components.items()
                               if name in child.components and value is not None})
    child.register_to_config(denoising_step_list=list(RELEASE['fl2va']['sampling']['knots']))
    return child


def close_pipeline(pipe):
    if pipe is not None:
        pipe.unload_components(list(pipe.component_names))
    gc.collect()


@torch.no_grad()
def encode_conditioning(snapshot, prompts, output, *, check=lambda:None, source=None):
    """Own the COMPLETE BF16 TE on CPU, write detached conditions, then close.

    `source` may be reused by the sole owner after this returns. It carries no
    component object and does not prove future GPU admission or fit.
    """
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(p,str) or not p.strip() or len(p)>8000 for p in prompts)):
        raise ValueError('frozen complete native prompt list required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('reuse existing conditioning; never repeat completed native TE calls')
    source = ValidatedSnapshot(snapshot) if source is None else source
    if source.root != Path(snapshot).resolve():
        raise ValueError('one original snapshot owner required')
    pipe = child = None
    # A large CPU TE phase can fail after an earlier prompt completed. Preserve
    # each exact output; no completed conditioner call is silently repeated.
    from .journal import write_json
    parts=output.with_name(output.name+'.prompt-parts'); parts.mkdir(exist_ok=True)
    declaration=dict(schema=SCHEMA,profile=PROFILE,model_identity=model_identity(),prompts=prompts,
                     conditioning_source_sha256=file_hash(__file__),selection='raw',ema_decay=None)
    cohort=parts/'cohort.json'
    if cohort.exists():
        if json.loads(cohort.read_text())!=declaration:raise ValueError('partial full-TE prompt cohort differs')
    else:write_json(cohort,declaration)
    values={}
    pending=[]
    for index,prompt in enumerate(dict.fromkeys(prompts)):
        path=parts/f'prompt-{index:04d}.pt';attempt=parts/f'prompt-{index:04d}-attempt.json'
        complete=parts/f'prompt-{index:04d}-complete.json'
        binding=dict(cohort=record(cohort),prompt=prompt,index=index)
        if complete.exists():
            saved=json.loads(complete.read_text())
            if saved.get('binding')!=binding or saved.get('result')!=record(path):raise ValueError('completed native condition changed')
            values[prompt]=torch.load(path,map_location='cpu',weights_only=True)
        elif attempt.exists():
            raise RuntimeError('unknown interrupted native text call retained; no automatic forward replay')
        else:pending.append((prompt,path,attempt,complete,binding))
    try:
        if pending:
            check()
            with construction_rng():
                pipe = selected_pipeline(source, ['text_encoder','tokenizer','processor'], dtype=torch.bfloat16)
                pipe.text_encoder.requires_grad_(False).eval().to(device='cpu')
            if pipe.text_encoder.config.text_config.num_hidden_layers != 64:
                raise ValueError('full original64-layer Qwen3VL required, not a50-layer substitute')
            child = subpipeline(pipe, ['text_encoder'])
        for prompt,path,attempt,complete,binding in pending:
            check(); source.check()
            write_json(attempt,dict(schema='h3-phase-native-text-call-attempt-v1',binding=binding))
            state = child(prompt=prompt)
            embeds, tags = state.get('prompt_embeds'), state.get('text_token_tags')
            if (not torch.is_tensor(embeds) or embeds.ndim!=3 or embeds.shape[0]!=1
                    or embeds.shape[2]!=5120 or tags.shape!=(embeds.shape[1],)
                    or embeds.dtype!=torch.bfloat16 or not torch.isfinite(embeds).all()
                    or tags.dtype!=torch.long or not torch.all(tags==pipe.text_tag)):
                raise ValueError('complete finite native hidden50 and original token tags required')
            values[prompt] = dict(prompt_embeds=embeds.detach().cpu().clone(),
                                  text_token_tags=tags.detach().cpu().clone())
            atomic_save(path,values[prompt])
            write_json(complete,dict(schema='h3-phase-native-text-call-complete-v1',binding=binding,result=record(path)))
            del state, embeds, tags
        packet = dict(schema=SCHEMA, profile=PROFILE, model_identity=model_identity(), prompts=prompts,
                      values=values, selection='raw', ema_decay=None,
                      conditioning_source_sha256=file_hash(__file__), teacher_calls=0)
        atomic_save(output, packet)
        return record(output)
    finally:
        from particle_sliders.restoration import _restore_all
        _restore_all([lambda:close_pipeline(child),lambda:close_pipeline(pipe)],primary=sys.exception())


class Conditions:
    def __init__(self, path, sha256):
        self.path, self.sha256 = Path(path), sha256
        if file_hash(self.path)!=sha256:
            raise ValueError('complete native text-conditioning bytes differ')
        value = torch.load(self.path, map_location='cpu', weights_only=True)
        if (value.get('schema')!=SCHEMA or value.get('profile')!=PROFILE
                or value.get('model_identity')!=model_identity() or value.get('ema_decay')is not None
                or value.get('selection')!='raw' or value.get('teacher_calls')!=0
                or value.get('conditioning_source_sha256')!=file_hash(__file__)
                or set(value['values'])!=set(value['prompts'])):
            raise ValueError('exact original complete no-grad phase conditioning required')
        self.values = value['values']
        for prompt, item in self.values.items():
            embeds,tags=item['prompt_embeds'],item['text_token_tags']
            if (not isinstance(prompt,str) or not prompt.strip() or embeds.ndim!=3 or embeds.shape[0]!=1
                    or embeds.shape[2]!=5120 or embeds.dtype!=torch.bfloat16
                    or tags.shape!=(embeds.shape[1],) or tags.dtype!=torch.long
                    or not torch.isfinite(embeds).all() or not torch.all(tags==1)):
                raise ValueError('complete original finite hidden50 and text-tag rows required')

    def get(self, prompt, device):
        if prompt not in self.values:
            raise ValueError('prompt absent from completed CPU conditioning; do not load TE beside DiT')
        value = self.values[prompt]
        return {name:tensor.to(device=device) for name,tensor in value.items()}
