"""Once-prepared, content-bound original full native paired AV teacher data."""
from dataclasses import asdict
from pathlib import Path
import torch
from particle_sliders.lifecycle import atomic_save
from .contracts import geometry,validate_rows,digest,file_hash,record,RELEASE
from .packing import PackedLayout
from .native import PreparedH3Flex
from .adapter import H3Pair,PreparedPair,H3ResponseAdapter

SCHEMA='h3-phase-owned-full-packed-paired-cache-v1'

def _pack(value):
    return {name:getattr(value,name).detach().cpu().clone() for name in PackedLayout.__dataclass_fields__}|{'frame_metadata':dict(value.frame_metadata)}

def _unpack(value,device):
    if set(value)!=set(PackedLayout.__dataclass_fields__)|{'frame_metadata'}:raise ValueError('exact native packed fields required')
    return PreparedH3Flex(**{key:item.to(device) for key,item in value.items() if key!='frame_metadata'},frame_metadata=value['frame_metadata'])

def prepare_cache(runtime,prompts,path,*,duration=5,resolution='480p',seed=7):
    """Commit each new full native teacher once; retain partial/unknown calls.

    The original response/scales/row order are unchanged. New phase cohorts
    have no completed teachers to replay. A started call with no durable output
    is ambiguous and is refused, not silently repeated after interruption.
    """
    validate_rows(prompts);shape=geometry(dict(duration=duration,resolution=resolution))
    path=Path(path)
    if path.exists():raise FileExistsError('reuse completed cache; no repeated teacher preparation')
    if type(seed)is not int or not 0<=seed<2**31:raise ValueError('explicit native seed required')
    from .journal import write_json
    import json
    progress=path.with_name(path.name+'.teacher-parts')
    progress.mkdir(exist_ok=True)
    cohort=dict(schema='h3-phase-complete-teacher-cohort-v1',model_identity=runtime.identity,
        geometry=shape,prompts=prompts,seed=seed,conditioning=record(runtime.conditions.path),
        selection='raw',ema_decay=None,teacher_slider=0.,positive_training_scale=1.)
    declaration=progress/'cohort.json'
    if declaration.exists():
        if json.loads(declaration.read_text())!=cohort:raise ValueError('partial native teacher cohort identity differs')
    else:write_json(declaration,cohort)
    adapter=H3ResponseAdapter(runtime,shape['response_shape'],model_identity=runtime.identity)
    rows=[]
    try:
        for index,prompt in enumerate(prompts):
            kwargs=dict(duration=duration,resolution=resolution,seed=seed+index)
            prepared=progress/f'row-{index:04d}-native-pair.pt'
            if not prepared.exists():
                pair=H3Pair(runtime.prepare_t2va(prompt['neutral'],**kwargs),runtime.prepare_t2va(prompt['positive'],**kwargs))
                for name in('hidden_states','audio_hidden_states','timestep'):
                    if not torch.equal(getattr(pair.neutral,name),getattr(pair.positive,name)):
                        raise ValueError('same native noise/times required in both full teachers')
                atomic_save(prepared,dict(neutral=_pack(pair.neutral),positive=_pack(pair.positive)))
            native=torch.load(prepared,map_location='cpu',weights_only=True)
            values={}
            # Same original positive-then-neutral teacher order and scale0.
            for field,side in(('target','positive'),('neutral_response','neutral')):
                result=progress/f'row-{index:04d}-{field}.pt'
                attempt=progress/f'row-{index:04d}-{field}-attempt.json'
                complete=progress/f'row-{index:04d}-{field}-complete.json'
                binding=dict(row=index,side=side,cohort=record(declaration),native_pair=record(prepared))
                if complete.exists():
                    saved=json.loads(complete.read_text())
                    if saved.get('binding')!=binding or saved.get('result')!=record(result):
                        raise ValueError('durable first native teacher identity changed')
                elif attempt.exists():
                    raise RuntimeError('unknown interrupted native teacher retained; no automatic forward replay')
                else:
                    write_json(attempt,dict(schema='h3-phase-new-native-teacher-attempt-v1',binding=binding))
                    with torch.no_grad():
                        value=adapter._response(_unpack(native[side],runtime.device),0).detach().cpu().clone()
                    atomic_save(result,value)
                    write_json(complete,dict(schema='h3-phase-native-teacher-complete-v1',binding=binding,result=record(result)))
                    del value
                value=torch.load(result,map_location='cpu',weights_only=True)
                if list(value.shape)!=shape['response_shape']or not torch.isfinite(value).all():
                    raise ValueError('durable full finite native teacher vector required')
                values[field]=value
            rows.append(dict(neutral=native['neutral'],positive=native['positive'],**values))
        packet=dict(schema=SCHEMA,model_identity=runtime.identity,release_sha256=digest(RELEASE),
            geometry=shape,prompts=prompts,seed=seed,rows=rows,teacher_slider=0.,positive_training_scale=1.,
            conditioning=record(runtime.conditions.path),
            weight_policy='raw-no-weight-EMA',ema_decay=None)
        atomic_save(path,packet)
        return record(path)
    finally:adapter.close()

class CachedRows:
    def __init__(self,path,sha256,runtime,shape):
        self.path=Path(path);self.sha256=sha256;self.response_shape=tuple(shape['response_shape'])
        if file_hash(self.path)!=sha256:raise ValueError('native cache whole-file hash differs')
        packet=torch.load(self.path,map_location='cpu',weights_only=True)
        if (packet.get('schema')!=SCHEMA or packet.get('model_identity')!=runtime.identity
            or packet.get('release_sha256')!=digest(RELEASE)or packet.get('geometry')!=shape
            or packet.get('conditioning')!=record(runtime.conditions.path)
            or packet.get('ema_decay')is not None or packet.get('weight_policy')!='raw-no-weight-EMA'
            or packet.get('teacher_slider')!=0. or packet.get('positive_training_scale')!=1.):raise ValueError('cache identity/geometry/native policy differs')
        validate_rows(packet['prompts'])
        if len(packet['rows'])!=len(packet['prompts']):raise ValueError('complete frozen row cohort required')
        self.rows=[]
        for row in packet['rows']:
            pair=H3Pair(_unpack(row['neutral'],runtime.device),_unpack(row['positive'],runtime.device))
            target,neutral=(row[k].to(runtime.device,dtype=torch.float32)for k in('target','neutral_response'))
            if any(list(v.shape)!=shape['response_shape']or not bool(torch.isfinite(v).all()) for v in(target,neutral)):raise ValueError('complete finite full native AV teacher vectors required')
            self.rows.append(PreparedPair(pair,target,neutral))
        self.packet_metadata={k:v for k,v in packet.items()if k!='rows'}
    def identity(self):
        if file_hash(self.path)!=self.sha256:raise ValueError('frozen cache changed')
        return dict(schema=SCHEMA,cache_sha256=self.sha256,metadata=self.packet_metadata)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]
    def conditions(self,indices):
        return torch.eye(len(self.rows),device=self.rows[0].target.device)[indices]
