"""Allocation-free new-cohort declarations; historical UNI/PG094 stay separate."""
import hashlib,json,math,re
from pathlib import Path
CORE_REVISION='41f6066f2d2644902670d5eb736de052e1a70553'
BACKEND_NAME='minimax_h3_phase_owned_av95'
SOURCE_PROFILE='h3-native-allCUDA-DiT24-26-refiner1-1-PG0-group5-sync-unwind-FF512-v3'
PROTOCOL='h3-full-native-video-then-audio-velocity-raw-v1'
INITIALIZATION_POLICY='h3-kaiming-sqrt5-QR-down-zero-UP-KEEP-HNDL-critic-v1'
WEIGHT_POLICY='raw-no-weight-EMA'
RELEASE=json.loads((Path(__file__).with_name('data')/'pika-h3-flex-release.json').read_text())

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path):
    path=Path(path);return dict(path=str(path.absolute()),bytes=path.stat().st_size,sha256=file_hash(path))

def geometry(options):
    duration,resolution=options.get('duration',5),options.get('resolution','480p')
    if type(duration)is not int or not 5<=duration<=15 or resolution not in ('480p','768p'):
        raise ValueError('native integer5..15seconds and480p/768p required; no reduced geometry')
    t,p=RELEASE['temporal_contract'],RELEASE['packed_geometry']
    requested=duration*t['fps'];frames=((requested-t['latents_per_chunk']+t['frames_per_chunk']-1)//t['frames_per_chunk'])*t['frames_per_chunk']+t['latents_per_chunk']
    height,width=p['canvases'][resolution]
    latent_frames=(frames-t['latents_per_chunk'])//t['frames_per_chunk']*t['latents_per_chunk']+t['latent_frame_bias']
    video_rows=latent_frames*(height//p['row_spatial_multiple'])*(width//p['row_spatial_multiple'])
    audio_rows=round(frames/t['fps']*t['audio_latents_per_second'])*t['audio_channels']
    return dict(duration=duration,resolution=resolution,height=height,width=width,requested_frames=requested,
        prepared_frames=frames,video_rows=video_rows,audio_rows=audio_rows,
        response_shape=[1,video_rows*p['video_features']+audio_rows*p['audio_features']])

def validate_rows(rows):
    if not isinstance(rows,list)or not 2<=len(rows)<=128:raise ValueError('2..128 native matched prompt pairs required')
    for row in rows:
        if not isinstance(row,dict)or set(row)!={'neutral','positive'}or any(not isinstance(v,str)or not v.strip()or len(v)>8000 for v in row.values())or row['neutral']==row['positive']:
            raise ValueError('distinct complete neutral/positive prompts required')

def rates(options):
    if options.get('prior_lr_mult')is not None:raise ValueError('adapter-only native H3 has no bank/prior LR')
    values={k:options[k]for k in('lr','d_lr_mult')if k in options}
    if any(type(v)not in(int,float)or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('literal positive finite G/D common rates required')
    return values

def validate_options(options,*,training=True):
    allowed={'snapshot','device','base_precision','conditioning','conditioning_sha256'}
    if training:allowed|={'cache','cache_sha256','duration','resolution','steps','batch','seed','ema_decay','pg_version','pg_source_profile','penalty_input_domain','source_profile','initialization_policy','lr','d_lr_mult','prior_lr_mult','checkpointing'}
    if set(options)-allowed:raise ValueError('unknown H3 options; no native shape/formula/weight-policy overrides')
    if not isinstance(options.get('snapshot'),str)or not Path(options['snapshot']).is_dir():raise ValueError('explicit locally staged exact Pika snapshot required')
    if not re.fullmatch(r'cpu|cuda:[0-9]+',options.get('device','cpu'))or options.get('base_precision','bfloat16')!='bfloat16':raise ValueError('explicit native device, BF16 base and released FP32 VAE islands required')
    if not isinstance(options.get('conditioning'),str)or not Path(options['conditioning']).is_file()or not re.fullmatch('[0-9a-f]{64}',options.get('conditioning_sha256','')):raise ValueError('explicit complete frozen phase text-conditioning artifact required')
    if options.get('device','cuda:0')!='cuda:0':raise ValueError('resident grouped profile fixes game/entry cuda:0 and blocks24..49 cuda:1')
    if training:
        if 'ema_decay'not in options or options['ema_decay']is not None:raise ValueError('explicit ema_decay=None required; weight EMA refused')
        if tuple(options.get(k)for k in('pg_version','pg_source_profile','penalty_input_domain','source_profile','initialization_policy'))!=('0.9.5','release-0.9.5','continuous',SOURCE_PROFILE,INITIALIZATION_POLICY):raise ValueError('explicit new095 full-vector continuous/fresh-initialization profile required')
        geometry(options);rates(options)
        for key,default,low in(('steps',4,1),('batch',2,2),('seed',7,0)):
            value=options.get(key,default)
            if type(value)is not int or not low<=value<2**31:raise ValueError('invalid '+key)
        if options.get('checkpointing')is not True:raise ValueError('new phase profile explicitly requires scale-aware shared block/refiner checkpointing; native flags stayFalse')
        if not isinstance(options.get('cache'),str)or not Path(options['cache']).is_file()or not re.fullmatch('[0-9a-f]{64}',options.get('cache_sha256','')):raise ValueError('content-bound complete native cache required')

def validate_installed_core():
    from importlib import metadata
    lock=json.loads(Path(__file__).with_name('current_core.json').read_text());dist=metadata.distribution('particle-sliders-core')
    if dist.version!='0.14.1'or lock['commit']!=CORE_REVISION:raise ValueError('exact current core41f source required')
    for name,expected in lock['checked_api_files'].items():
        if file_hash(dist.locate_file('particle_sliders/'+name))!=expected:raise ValueError('core source mismatch:'+name)
