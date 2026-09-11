"""Final bounded comparison through the reusable single-shot video core."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from h3ce.cache.keys import file_sha256
from h3ce.train.checkpoint import TrainingBudget,CheckpointManager
from h3ce.train.guard import no_training_guard
from h3ce.infer.native_head import FrozenHeadVideoPipeline
from h3ce.vae.bridge import H3VAEBridge,FrameMeta
from h3ce.vae.aitoolkit_h3_backend import AIToolkitH3Backend
from h3ce.vae.decoder_tile_batch import DecoderTileBatchH3Backend
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head256 import NAFHead
from scripts.research_naf_head_inference import NAFHeadInference
from scripts.benchmark_h3ce_native_video import read
from scripts import benchmark_swinir_batch8_strict_video as timing
OUT=ROOT/'logs/native-head-fastpath-20260910-v1'
RUN=ROOT/'runs/naf-head256-train-20260910-v1'
PARENT=ROOT/'logs/architecture-speed-20260910-v1'


def save(path,value):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=scalar),encoding='utf-8')


class OriginalHead(torch.nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,x):
        cache=self.head.features(x)
        return self.head.delta(cache['features'],cache['skip'])


def main():
    assert sys.argv[1:]==['--run'] and not OUT.exists();OUT.mkdir();torch.set_num_threads(4)
    parent=read(PARENT/'protocol.json');checkpoint=parent['checkpoint']
    for f,h in parent['bindings'].items():assert file_sha256(f)==h,f
    files=[Path(__file__),ROOT/'h3ce/infer/native_head.py',ROOT/'h3ce/vae/decoder_tile_batch.py',
        ROOT/'h3ce/vae/tile_batch.py',ROOT/'scripts/research_naf_head_inference.py',ROOT/'docs/NATIVE_HEAD_FASTPATH_PROTOCOL_20260910.md',
        PARENT/'protocol.json',ROOT/'logs/decoder-tile-speed-20260910-v1/protocol.json',ROOT/'logs/decoder-tile-speed-20260910-v1/failure.json']
    protocol={'cases':parent['cases'],'gates':parent['gates'],'checkpoint':checkpoint,'head_side':256,'batch_size':8,
        'scope':'GPU resident frames to returned float prediction; includes geometry validation/pad/H3/condition/spatial/paste; excludes loading/detection/media/storage',
        'counts_max':{'encodes':4,'decodes':4,'serial_encoder':32,'fast_encoder':32,'serial_decoder':16,'fast_decoder':4,'backbone':12,'tail':12,'reference':6},
        'warmup_forwards':0,'updates':0,'bindings':{str(f):file_sha256(f) for f in files}}
    save(OUT/'protocol.json',protocol);timing.OUT=OUT;telemetry=timing.Telemetry();counts={k:0 for k in protocol['counts_max']};rows=[]
    def add(key):counts[key]+=1;assert counts[key]<=protocol['counts_max'][key],counts
    def wrap(key,fn):
        def call(*a,**kw):add(key);return fn(*a,**kw)
        return call
    with TrainingBudget(ROOT/'runs',172800,phase='native_head_fastpath_combined') as budget,no_training_guard() as guard:
        timing.wait_idle('before_load');budget.check()
        base,info=load_model();head=NAFHead(base)
        state=CheckpointManager(ROOT/'runs',RUN,contract=read(RUN/'training_contract.json')).read(Path(checkpoint['path']))
        assert state['step']==32 and file_sha256(checkpoint['path'])==checkpoint['sha256']
        head.tail.load_state_dict(state['model'],strict=True);del state
        head.cuda().eval().requires_grad_(False)
        bridges={name:H3VAEBridge(cls.from_locked(project_root=ROOT,weights='models/minimax_h3_video_vae_fp16.safetensors')) for name,cls in [('serial',AIToolkitH3Backend),('fast',DecoderTileBatchH3Backend)]}
        pipelines={'serial':FrozenHeadVideoPipeline(bridges['serial'],OriginalHead(head)),
                   'fast':FrozenHeadVideoPipeline(bridges['fast'],NAFHeadInference(head))}
        models=[head]+[b.backend.model for b in bridges.values()]
        versions=[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models]
        for key,module in [('backbone',head.backbone),('tail',head.tail),('reference',head.reference)]:
            module.register_forward_pre_hook(lambda m,a,key=key:add(key))
        for name,b in bridges.items():
            b.backend.encode_mean_raw=wrap('encodes',b.backend.encode_mean_raw);b.backend.decode_raw=wrap('decodes',b.backend.decode_raw)
            for component in ('encoder','decoder'):
                getattr(b.backend.model,component).register_forward_pre_hook(lambda m,a,key=name+'_'+component:add(key))
        save(OUT/'load.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'naf_cpu_constructor_forwards':1,
            'contracts':{name:{'encoder':b.encoder_id,'decoder':b.current_codec_pack().effective_decoder_hash,'native':b.inspect_contract().native} for name,b in bridges.items()}})
        timing.wait_idle('before_inference');telemetry.start()
        try:
            for case in protocol['cases']:
                kind=case['kind'];folder=OUT/kind;folder.mkdir();stages={}
                arr=np.load(case['input']);full=torch.from_numpy(arr).permute(0,3,1,2).cuda()
                gs=read(RUN/'video_check'/(kind+'_geometry.json'));meta=FrameMeta('video',tuple(case['pts']),real_video=True,shot_id='0')
                for name in (['serial','fast'] if kind=='degraded' else ['fast','serial']):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    result=pipelines[name](full,meta,gs);torch.cuda.synchronize()
                    stage={'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                    assert result['pts']==meta.pts
                    for key in ('prediction','delta','native','latent'):np.save(folder/(name+'_'+key+'.npy'),result[key].cpu().numpy())
                    del result
                    stages[name]=stage;print(kind+' '+name+' '+str(stage),flush=True)
                    save(folder/'timing.json',stages)
                rows.append({'kind':kind,'stages':stages,'pts':case['pts']});del full;budget.check()
            assert counts==protocol['counts_max'],counts
            assert versions==[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models]
            assert all(not v.requires_grad and v.grad is None for m in models for v in m.parameters())
            save(OUT/'summary.json',{'status':'combined_frozen_core_benchmark_complete','rows':rows,'counts':counts,
                'guard':dict(guard),'weights_unchanged':True,'budget':budget.snapshot(),
                'core_ratio':sum(r['stages']['fast']['seconds'] for r in rows)/sum(r['stages']['serial']['seconds'] for r in rows),
                'quality_acceptance':False,'end_to_end_acceptance':False,'artifact_hashes':{str(f):file_sha256(f) for f in OUT.glob('*/*.npy')}})
        finally:telemetry.finish()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():save(OUT/'failure.json',{'error':repr(exc),'automatic_restart':False})
        raise
