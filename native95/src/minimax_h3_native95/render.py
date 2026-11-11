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
    if output_type!='native_av':raise ValueError('finished AV only; hidden/latent fields are not media')

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

def render_av(runtime,*,prompt,strength,seed,geometry,output_type='native_av'):
    validate_request(prompt,strength,seed,output_type)
    import torch
    from PIL.Image import Image
    with scheduler_state(runtime.pipe):
        result=runtime.render(prompt,strength=strength,seed=seed,duration=geometry['duration'],resolution=geometry['resolution'])
    videos=result.get('videos');audio=result.get('audio');rate=result.get('sampling_rate')
    if (not isinstance(videos,list)or len(videos)!=1 or len(videos[0])!=geometry['prepared_frames']
        or any(not isinstance(frame,Image)or frame.size!=(geometry['width'],geometry['height'])for frame in videos[0])):raise ValueError('every complete native decoded frame at original canvas required')
    if (not torch.is_tensor(audio)or audio.ndim!=3 or tuple(audio.shape[:2])!=(1,2)or audio.shape[-1]<1
        or not bool(torch.isfinite(audio).all())or type(rate)is not int or rate<1):raise ValueError('native complete finite stereo audio and actual sample rate required')
    return result

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

def render_pair(*,artifact,snapshot,output,prompt,seed=7,strength=1.,device='cpu'):
    """Fresh artifact-only factory, exact RAW; no cache/teacher/critic/optimizer."""
    from particle_sliders.inference import open_inference
    from .lifecycle import NATIVE_BACKEND
    output=Path(output)
    if output.exists():raise FileExistsError('keep previous native render; use a fresh destination')
    validate_request(prompt,strength,seed,'native_av')
    # Authenticate and load the SAME original RAW bytes; never hash a later
    # path version as provenance for an already loaded packet.
    import torch
    original=Path(artifact).read_bytes()
    artifact_record=dict(path=str(Path(artifact).absolute()),bytes=len(original),sha256=hashlib.sha256(original).hexdigest())
    packet=torch.load(io.BytesIO(original),map_location='cpu',weights_only=True)
    del original
    if not isinstance(packet,dict)or packet.get('selection')!='raw':raise ValueError('original RAW-only inference artifact required')
    options=dict(snapshot=snapshot,device=device,base_precision='bfloat16')
    with open_inference(packet,backend=NATIVE_BACKEND,operation='render',options=options,strength=strength,
            output_type='native_av',native_request={'prompt':prompt})as session:
        output.mkdir(parents=True,exist_ok=False)
        receipt=dict(schema='h3-native95-matched-RAW-native-AV-pair-v1',artifact=artifact_record,
            origin=session.manifest['origin'],request=dict(prompt=prompt,seed=seed,strength=strength),
            Pika_release_strength=1.,selection='raw',ema_decay=None,teacher_cache_used=False,training_updates=0,outputs=[])
        for label,alpha in(('OFF',0.),('ON',strength)):
            result=session.render(prompt=prompt,strength=alpha,seed=seed,output_type='native_av')
            receipt['outputs'].append(dict(label=label,strength=alpha,media=write_av(result,output/label)))
            from .journal import write_json
            write_json(output/'render-receipt.json',receipt,replace=True)
        return receipt
