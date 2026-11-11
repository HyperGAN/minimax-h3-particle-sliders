"""True native six-step decoded AV, lossless full frame sequence plus float WAV."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib,io,json,math,os,struct
from pathlib import Path
from .contracts import record

def validate_request(prompt,strength,seed,output_type):
    if not isinstance(prompt,str)or not prompt.strip()or len(prompt)>8000:raise ValueError('complete native prompt required')
    if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('finite unipolar slider strength0..1 required')
    if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('native seed required')
    if output_type not in('native_av','native_av_latents'):raise ValueError('explicit finished-AV or INTERNAL native-latent phase required')

@contextmanager
def scheduler_state(pipe):
    # Each complete OFF/ON call starts with fresh native scheduler state. The
    # original model/adapter objects and fixed release scale remain unchanged.
    old={name:getattr(pipe,name)for name in('scheduler','audio_scheduler')}
    from particle_sliders.restoration import _restore_all
    try:
        pipe.update_components(**{name:deepcopy(value)for name,value in old.items()})
        yield
    except BaseException as primary:
        _restore_all([lambda:pipe.update_components(**old)],primary=primary)
        raise
    else:_restore_all([lambda:pipe.update_components(**old)])

def write_av(result,directory):
    """No video/audio resampling, discarded frames, quantization, or fake mux."""
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    frame_records=[]
    for index,frame in enumerate(result['videos'][0]):
        path=directory/f'frame-{index:06d}.png'
        with path.open('xb')as stream:
            frame.save(stream,format='PNG');stream.flush();os.fsync(stream.fileno())
        frame_records.append(record(path))
    # WAVE_FORMAT_IEEE_FLOAT32, original stereo channel ordering interleaved.
    audio=result['audio'].detach().to(device='cpu',dtype=torch.float32)[0].transpose(0,1).contiguous()
    payload=audio.numpy().astype('<f4',copy=False).tobytes();rate=result['sampling_rate'];channels=2
    if len(payload)+36>=2**32:raise ValueError('native output exceeds RIFF32; no truncation allowed')
    path=directory/'audio-float32.wav'
    with path.open('xb')as stream:
        stream.write(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(payload),b'WAVE',b'fmt ',16,3,channels,rate,rate*8,8,32,b'data',len(payload)))
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    receipt=dict(schema='h3-native95-complete-decoded-AV-export-v1',frame_metadata=result['frame_metadata'],
        fps=24,frames=frame_records,audio=record(path),sampling_rate=rate,channels=2,samples=audio.shape[0],
        container='complete PNG sequence + native stereo IEEE-float32 WAV; no mux claimed',selection='raw',ema_decay=None)
    from .journal import write_json
    write_json(directory/'media.json',receipt)
    return receipt

def render_pair(*,artifact,snapshot,output,prompt,seed=7,strength=1.,device='cuda:0',check=lambda:None):
    """Complete RAW→both native denoises→close DiT→both full FP32 VAE exports.

    Persist internal latents before decoding so a decoder failure never requires
    replaying the completed native denoise. This API returns finished AV only.
    """
    from particle_sliders.inference import open_inference
    from particle_sliders.lifecycle import atomic_save
    from .lifecycle import NATIVE_BACKEND
    from .conditioning import ValidatedSnapshot,encode_conditioning
    from .contracts import RELEASE,SOURCE_PROFILE,digest,file_hash
    from .journal import write_json
    import torch
    from .placement import Placement
    Placement().validate()
    output=Path(output)
    if output.exists():raise FileExistsError('retain prior native phases; use decode-saved for completed latents')
    validate_request(prompt,strength,seed,'native_av')
    output.mkdir(parents=True,exist_ok=False)
    source=ValidatedSnapshot(snapshot)
    condition=encode_conditioning(snapshot,[prompt],output/'conditioning.pt',check=check,source=source)
    # Same immutable bytes produce the parsed RAW packet and its provenance.
    raw_bytes=Path(artifact).read_bytes()
    raw_pin=dict(path=str(Path(artifact).absolute()),bytes=len(raw_bytes),sha256=hashlib.sha256(raw_bytes).hexdigest())
    packet=torch.load(io.BytesIO(raw_bytes),map_location='cpu',weights_only=True)
    del raw_bytes
    if not isinstance(packet,dict)or packet.get('selection')!='raw':raise ValueError('exact RAW-only native inference artifact required')
    receipt=dict(schema='h3-phase-native-RAW-two-complete-denoises-v1',status='denoise_incomplete',artifact=raw_pin,
        profile=SOURCE_PROFILE,model_id=RELEASE['model_id'],model_revision=RELEASE['model_revision'],
        release_sha256=digest(RELEASE),conditioning=condition,request=dict(prompt=prompt,seed=seed,strength=strength),
        selection='raw',ema_decay=None,Pika_release_strength=1.,outputs=[],
        decoder_source_sha256=file_hash(Path(__file__)))
    options=dict(snapshot=snapshot,device=device,base_precision='bfloat16',
                 conditioning=condition['path'],conditioning_sha256=condition['sha256'])
    with open_inference(packet,backend=NATIVE_BACKEND,operation='render',options=options,strength=strength,
            output_type='native_av_latents',native_request={'prompt':prompt})as session:
        receipt['origin']=session.manifest['origin']
        for label,alpha in(('OFF',0.),('ON',strength)):
            check()
            result=session.render(prompt=prompt,strength=alpha,seed=seed,output_type='native_av_latents')
            path=output/(label+'-native-latents.pt')
            atomic_save(path,result)
            receipt['outputs'].append(dict(label=label,strength=alpha,latents=record(path)))
            write_json(output/'latents-receipt.json',receipt,replace=True)
            del result
    # The inference observation and factory have now both closed. No common
    # session/buffer snapshot is allowed to retain the DiT into VAE loading.
    del session,packet
    import gc
    gc.collect()
    check();source.check()
    receipt['status']='two_complete_denoises_closed_before_decode'
    write_json(output/'latents-receipt.json',receipt,replace=True)
    return decode_saved(snapshot=snapshot,receipt=output/'latents-receipt.json',output=output/'native-AV',
                        device=device,check=check,source=source)


def decode_saved(*,snapshot,receipt,output,device='cuda:0',check=lambda:None,source=None):
    """Decode-only continuation: never opens RAW/DiT, TE, cache or optimizers."""
    import torch
    from .native import decode_latents
    from .conditioning import ValidatedSnapshot
    from .contracts import RELEASE,SOURCE_PROFILE,digest,file_hash
    from .journal import write_json
    receipt=Path(receipt);receipt_bytes=receipt.read_bytes();bundle=json.loads(receipt_bytes)
    receipt_pin=dict(path=str(receipt.absolute()),bytes=len(receipt_bytes),sha256=hashlib.sha256(receipt_bytes).hexdigest())
    if (bundle.get('schema')!='h3-phase-native-RAW-two-complete-denoises-v1'
            or bundle.get('status')!='two_complete_denoises_closed_before_decode'
            or bundle.get('profile')!=SOURCE_PROFILE or bundle.get('model_id')!=RELEASE['model_id']
            or bundle.get('model_revision')!=RELEASE['model_revision'] or bundle.get('release_sha256')!=digest(RELEASE)
            or bundle.get('selection')!='raw' or bundle.get('ema_decay')is not None
            or bundle.get('Pika_release_strength')!=1. or bundle.get('decoder_source_sha256')!=file_hash(__file__)
            or [x['label']for x in bundle.get('outputs',[])]!=['OFF','ON']):
        raise ValueError('genuine completed exact two-denoise phase receipt required')
    for row in bundle['outputs']:
        if record(row['latents']['path'])!=row['latents']:raise ValueError('completed native latent bytes changed')
    output=Path(output);progress_path=output/'render-receipt.json'
    source=ValidatedSnapshot(snapshot)if source is None else source
    if output.exists():
        if not progress_path.is_file():raise FileExistsError('unreceipted prior output retained; no automatic overwrite/replay')
        progress=json.loads(progress_path.read_text())
        if progress.get('latent_receipt')!=receipt_pin:raise ValueError('same original denoise receipt required')
    else:
        output.mkdir(parents=True,exist_ok=False)
        progress=dict(schema='h3-phase-complete-native-AV-pair-export-v1',status='decode_incomplete',
            latent_receipt=receipt_pin,artifact=bundle['artifact'],origin=bundle['origin'],
            request=bundle['request'],selection='raw',ema_decay=None,profile=SOURCE_PROFILE,outputs=[])
        write_json(progress_path,progress)
    for row in bundle['outputs']:
        previous=[x for x in progress['outputs']if x['label']==row['label']]
        if previous:
            if len(previous)!=1 or previous[0]['latents']!=row['latents']:raise ValueError('same completed decoder branch required')
            media=previous[0]['media']
            for pin in media['frames']+[media['audio']]:
                if record(pin['path'])!=pin:raise ValueError('completed finished native media changed')
            continue
        if (output/row['label']).exists():raise FileExistsError('retain incomplete branch; no silent replacement of partial media')
        check();source.check()
        raw=Path(row['latents']['path']).read_bytes()
        if len(raw)!=row['latents']['bytes']or hashlib.sha256(raw).hexdigest()!=row['latents']['sha256']:raise ValueError('same completed native latent payload required')
        latent=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        del raw
        result=decode_latents(snapshot,latent,device=device,check=check,source=source)
        media=write_av(result,output/row['label'])
        progress['outputs'].append(dict(label=row['label'],strength=row['strength'],latents=row['latents'],media=media))
        write_json(progress_path,progress,replace=True)
        del latent,result
    progress['status']='complete'
    write_json(progress_path,progress,replace=True)
    return progress
