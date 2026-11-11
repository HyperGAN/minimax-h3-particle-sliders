"""Current shared PG095 lifecycle; no product loss/optimizer/training loop."""
from copy import deepcopy
from contextlib import nullcontext
import torch
from particle_sliders.lifecycle import Binding,Capabilities,UnsupportedOperation
from particle_sliders.response import ResponseBackend,ResponseCapabilities
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from particle_sliders.pg095 import ConditionalPG095,engine_options
from particle_sliders.restoration import _restore_all
from .contracts import (BACKEND_NAME,SOURCE_PROFILE,PROTOCOL,INITIALIZATION_POLICY,WEIGHT_POLICY,
    geometry,validate_options,validate_installed_core,rates)
from .native import PikaH3FlexBackend
from .adapter import H3ResponseAdapter,observe_native
from .bounded_critic import HNDLCritic
from .cache import CachedRows
from .initialization import Construction,construction_for,initialize_owned,validate_receipt

def capabilities(options):
    return ResponseCapabilities('adapter_gan','video_audio','native_packed_av_velocity','continuous_vector',
        tuple(geometry(options)['response_shape']),'0.9.5','release-0.9.5',penalty_input_domain='continuous',
        base_precisions=('bfloat16',),qualification='New full-native AV Source; tiny/pretrained/GPU/quality checks UNRUN')

INFERENCE_CAPABILITIES=Capabilities('adapter_gan',True,base_precisions=('bfloat16',),render_outputs=('native_av_latents',),
    operations=('render','response'),qualification='INTERNAL original AV denoise latents, NOT finished media; public phase render_pair completes both VAE phases')

def preflight(options,request,staged):
    if set(staged)-{'construction'}:raise ValueError('no fixture/borrowed runtime substitution in public native factory')
    c=staged.get('construction')
    from .contracts import digest
    if not isinstance(c,Construction)or c.options_sha256!=digest(options):raise ValueError('use open_training for explicit fresh/resume intent before prepare')
    if request['native']:raise ValueError('training uses only its fixed original native cache')

def prepare(options,*,construction=None):
    validate_options(options);validate_installed_core();preflight(options,{'native':{}},{'construction':construction})
    runtime=PikaH3FlexBackend(options['snapshot'],device=options.get('device','cuda:0'),
        conditioning=options['conditioning'],conditioning_sha256=options['conditioning_sha256']);adapter=None
    def close():_restore_all([*([adapter.close]if adapter is not None else[]),runtime.close])
    try:
        shape=geometry(options);rows=CachedRows(options['cache'],options['cache_sha256'],runtime,shape)
        devices=[]if runtime.device.type=='cpu'else[runtime.device.index]
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),devices)
            critic=HNDLCritic(shape['response_shape'][1],len(rows)).to(device=runtime.device,dtype=torch.float32)
            receipt=initialize_owned(runtime.network,critic,construction,options)
        adapter=H3ResponseAdapter(runtime,shape['response_shape'],model_identity=runtime.identity,checkpointing=options.get('checkpointing',False))
        adapter.fresh_initialization=receipt.as_dict()
        game=ConditionalPG095(adapter,rows,critic,particles=None,response_domain='continuous_vector',
            batch_size=options.get('batch',2),total_steps=options.get('steps',4),seed=options.get('seed',7),
            recipe_overrides=rates(options),**engine_options(options))
        configuration=dict(geometry=shape,rank=8,alpha=8.,weight_policy=WEIGHT_POLICY,
            source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,initialization=receipt.as_dict())
        spec=NATIVE_BACKEND.specification(adapter,model_identity=runtime.identity,protocol=PROTOCOL,configuration=configuration)
        return Binding(game,close,inference=spec,before_update=lambda request:bool(rows.identity()),observation=lambda:observe_native(runtime))
    except BaseException as primary:
        _restore_all([close],primary=primary);raise

def validate_inference(options,spec):
    validate_options(options,training=False)
    c=spec.configuration
    if(spec.identity['protocol']!=PROTOCOL or set(c)!={'geometry','rank','alpha','weight_policy','source_profile','initialization_policy','initialization'}
        or c['rank']!=8 or c['alpha']!=8. or c['weight_policy']!=WEIGHT_POLICY or c['source_profile']!=SOURCE_PROFILE
        or c['initialization_policy']!=INITIALIZATION_POLICY or geometry(c['geometry'])!=c['geometry']):raise ValueError('exact new native AV RAW source/geometry required; no historical retagging')
    validate_receipt(c['initialization'],c['initialization'].get('cohort_id'))

def preflight_inference(options,request,spec):
    native=request['native']
    if request['operation']=='render':
        if set(native)-{'prompt','images'}or native.get('images')is not None:raise ValueError('t2va only; no ref2va/keyframe substitutions')
        from .render import validate_request
        validate_request(native.get('prompt'),request['strength'],0,request['output_type'])
    elif set(native)!={'packed'}:raise ValueError('complete original packed layout required for native response')

def prepare_inference(options,spec):
    validate_inference(options,spec);validate_installed_core()
    runtime=PikaH3FlexBackend(options['snapshot'],device=options.get('device','cuda:0'),
        conditioning=options['conditioning'],conditioning_sha256=options['conditioning_sha256']);adapter=None
    def close():_restore_all([*([adapter.close]if adapter is not None else[]),runtime.close])
    try:
        shape=spec.configuration['geometry']
        adapter=H3ResponseAdapter(runtime,shape['response_shape'],model_identity=runtime.identity)
        adapter.fresh_initialization=spec.configuration['initialization'] # Metadata only; never invoke fresh helper.
        identity=native_identity(adapter,model_identity=runtime.identity,protocol=PROTOCOL)
        def response(*,strength,seed,packed):return adapter._response(packed,strength).detach().clone()
        def render(*,strength,seed,prompt,images=None,output_type='native_av'):
            if images is not None:raise ValueError('text-only native AV workflow required')
            return runtime.render_latents(prompt,strength=strength,seed=seed,duration=shape['duration'],resolution=shape['resolution'])
        return InferenceBinding(adapter.network,identity,close,response=response,render=render,observation=lambda:observe_native(runtime))
    except BaseException as primary:
        _restore_all([close],primary=primary);raise

NATIVE_BACKEND=InferenceBackend(BACKEND_NAME,INFERENCE_CAPABILITIES,prepare_inference,validate_inference,preflight=preflight_inference)
BACKEND=ResponseBackend(BACKEND_NAME,capabilities({}),prepare,validate_options,configure_capabilities=capabilities,preflight=preflight)

def open_training(config,*,resume=None):
    from particle_sliders.lifecycle import read_packet
    validate_options(config.options);validate_installed_core()
    packet=deepcopy(resume)if isinstance(resume,dict)else read_packet(resume)if resume is not None else None
    construction=construction_for(config,packet)
    session=BACKEND.open(config,operation='resume'if packet is not None else'train',construction=construction)
    try:
        if packet is not None:session.restore(packet)
        return session
    except BaseException as primary:
        _restore_all([session.close],primary=primary);raise
