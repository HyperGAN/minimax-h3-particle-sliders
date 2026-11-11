"""Once-prepared, content-bound original full native paired AV teacher data."""
from dataclasses import asdict
from pathlib import Path
import torch
from particle_sliders.lifecycle import atomic_save
from .contracts import geometry,validate_rows,digest,file_hash,record,RELEASE
from .packing import PackedLayout
from .native import PreparedH3Flex
from .adapter import H3Pair,PreparedPair,H3ResponseAdapter

SCHEMA='h3-native95-full-packed-paired-cache-v1'

def _pack(value):
    return {name:getattr(value,name).detach().cpu().clone() for name in PackedLayout.__dataclass_fields__}|{'frame_metadata':dict(value.frame_metadata)}

def _unpack(value,device):
    if set(value)!=set(PackedLayout.__dataclass_fields__)|{'frame_metadata'}:raise ValueError('exact native packed fields required')
    return PreparedH3Flex(**{key:item.to(device) for key,item in value.items() if key!='frame_metadata'},frame_metadata=value['frame_metadata'])

def prepare_cache(runtime,prompts,path,*,duration=5,resolution='480p',seed=7):
    """No optimizers or learned updates; never overwrite a completed cache."""
    validate_rows(prompts);shape=geometry(dict(duration=duration,resolution=resolution))
    path=Path(path)
    if path.exists():raise FileExistsError('reuse completed cache; no repeated teacher preparation')
    if type(seed)is not int or not 0<=seed<2**31:raise ValueError('explicit native seed required')
    adapter=H3ResponseAdapter(runtime,shape['response_shape'],model_identity=runtime.identity)
    rows=[]
    try:
        for index,prompt in enumerate(prompts):
            kwargs=dict(duration=duration,resolution=resolution,seed=seed+index)
            pair=H3Pair(runtime.prepare_t2va(prompt['neutral'],**kwargs),runtime.prepare_t2va(prompt['positive'],**kwargs))
            item=adapter.prepare(pair)
            rows.append(dict(neutral=_pack(pair.neutral),positive=_pack(pair.positive),
                target=item.target.cpu(),neutral_response=item.neutral.cpu()))
        packet=dict(schema=SCHEMA,model_identity=runtime.identity,release_sha256=digest(RELEASE),
            geometry=shape,prompts=prompts,seed=seed,rows=rows,teacher_slider=0.,positive_training_scale=1.,
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
