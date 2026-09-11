"""Frozen-H3 float-output diagnostic; no optimization or training launch capability."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
import numpy as np
from functools import lru_cache
import torch
from h3ce.cache.keys import canonical_json,file_sha256
from h3ce.cache.store import atomic_write
from h3ce.train.guard import no_training_guard
from h3ce.train.checkpoint import TrainingBudget
from h3ce.train.preflight import open_dataset,load_bridge,execution_contract
from h3ce.train.pipeline import move_sample,refine_latent,restore_pixels
from h3ce.train.losses import ApplicationLoss
from h3ce.model import SpatialRefinerV2
from scripts.evaluate_bootstrap_checkpoint import read_committed,verify_subset,verify_compatibility,measure
from scripts.run_heavy_degradation_probe import read,require,check
from scripts.verify_scaled_length_probe import saved_at

BANDS=['dc','low','mid','high']

@lru_cache(maxsize=4)
def dct_basis(n):
    basis=np.cos(np.pi/n*np.arange(n)[:,None]*(np.arange(n)[None,:]+.5))*np.sqrt(2/n)
    basis[0]/=np.sqrt(2)
    return basis

def dctn(value,axes=(0,1),norm='ortho'):
    require(axes==(0,1) and norm=='ortho','Only orthonormal spatial DCT supported')
    basis=dct_basis(value.shape[0])
    return np.einsum('ki,ijc,lj->klc',basis,value,basis,optimize=True)

def band_masks(n):
    yy,xx=np.meshgrid(np.arange(n)/(2*n),np.arange(n)/(2*n),indexing='ij')
    r=np.hypot(xx,yy)
    return {'dc':r==0,'low':(r>0)&(r<=1/32),'mid':(r>1/32)&(r<=1/8),'high':r>1/8}

def frequency_metrics(x,y,pred):
    x,y,pred=[np.asarray(v,dtype=np.float64) for v in [x,y,pred]]
    require(x.shape==y.shape==pred.shape and x.ndim==3 and x.shape[0]==x.shape[1] and x.shape[-1]==3,'Square aligned RGB required')
    require(all(np.isfinite(v).all() for v in [x,y,pred]),'Nonfinite pixels')
    desired=dctn(y-x,axes=(0,1),norm='ortho');delta=dctn(pred-x,axes=(0,1),norm='ortho')
    tx=dctn(x,axes=(0,1),norm='ortho');ty=dctn(y,axes=(0,1),norm='ortho')
    result={};den=x.size
    for name,mask in band_masks(x.shape[0]).items():
        a,b=desired[mask],delta[mask];need=float(np.sum(a*a)/den);change=float(np.sum(b*b)/den)
        dot=float(np.sum(a*b)/den);error=float(np.sum((a-b)**2)/den)
        result[name]={'input_error_mse':need,'output_error_mse':error,'mse_gain':need-error,'change_energy':change,
            'aligned_fraction':dot/need if need>1e-20 else None,
            'amplitude_ratio':float(np.sqrt(change/need)) if need>1e-20 else None,
            'target_energy':float(np.sum(ty[mask]**2)/den),'input_energy':float(np.sum(tx[mask]**2)/den)}
    require(np.isclose(sum(v['input_error_mse'] for v in result.values()),np.mean((y-x)**2),rtol=1e-10,atol=1e-15),'Input Parseval mismatch')
    require(np.isclose(sum(v['output_error_mse'] for v in result.values()),np.mean((y-pred)**2),rtol=1e-10,atol=1e-15),'Output Parseval mismatch')
    return result

def declare(output):
    require(output.is_relative_to(ROOT/'logs') and not output.exists(),'New logs directory required')
    prior=ROOT/'logs/heavy-degradation-20260908';check(read(prior/'protocol.json'))
    run=Path(read(prior/'execution.json')['run']);cp=saved_at(run,64)
    paths=[cp,run/'resolved.yaml',run/'training_contract.json',prior/'protocol.json',prior/'evaluations.json',prior/'visual_protocol.json',
        ROOT/'runs/quality-calibration-20260908/encoded/heavy/overfit_manifest.jsonl',ROOT/'scripts/evaluate_bootstrap_checkpoint.py']
    p={'status':'declared_before_float_forward','run':str(run),'checkpoint':str(cp),'manifest':str(paths[-2]),
       'script_sha256':file_sha256(Path(__file__)),'sources':{str(f):file_sha256(f) for f in paths},
       'rois':read(prior/'visual_protocol.json')['rois'],'method':'Orthonormal 2-D DCT-II per RGB channel, fixed 128px ROI, no window, no clamp, no quantization',
       'bands_cycles_per_pixel':{'dc':'0','low':'(0,1/32]','mid':'(1/32,1/8]','high':'>1/8'},
       'metric':'Additive squared-error decomposition; not decomposition of training Charbonnier loss',
       'oracle':'X + feather*(D(E(Y))-D(E(X))); target latent used only for diagnosis, not deployable, not a theoretical quality bound',
       'chart_contract':{'renderer':'standalone matplotlib','question':'Which frequency bands contribute correction and error reduction?',
           'family':'grouped bar','grain':'8 sources equal weight per degraded/clean cohort','palette':'blue model; gray hatched oracle',
           'takeaway':'Unknown before measurement','output':'frequency_diagnostic.png'},
       'optimizer_updates':0,'independent_validation':False,'trained_base_accepted':False}
    output.mkdir();atomic_write(output/'protocol.json',canonical_json(p))

def verify_protocol(p):
    require(p['script_sha256']==file_sha256(Path(__file__)),'Diagnostic code changed')
    for f,h in p['sources'].items():require(file_sha256(Path(f))==h,'Bound source changed: '+f)

def run_diagnostic(output):
    p=read(output/'protocol.json');verify_protocol(p);require(not (output/'float_outputs').exists(),'Do not repeat existing forward outputs')
    run=Path(p['run']);prior=ROOT/'logs/heavy-degradation-20260908'
    with no_training_guard() as guard,TrainingBudget(ROOT/'runs',172800,phase='detail_frequency_forward_diagnostic') as budget:
        cfg,state,cp=read_committed(ROOT,run,p['checkpoint']);budget.validate_resume_snapshot(state['budget'])
        dataset=open_dataset(cfg,ROOT,Path(p['manifest']));verify_subset(run,state,dataset);verify_compatibility(ROOT,cfg,state,dataset)
        audit=dataset.audit(budget_check=budget.check);require(len(dataset)==16,'Keep all sixteen pairs')
        bridge=load_bridge(cfg,ROOT,dataset);model=SpatialRefinerV2.from_config(cfg.model).to('cuda').eval().requires_grad_(False)
        model.load_state_dict(state['model'],strict=True);models=[model,bridge.backend.model]
        versions=[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models]
        loss=ApplicationLoss(cfg.training.losses);rois={r['source']:r for r in p['rois']};records=[]
        old={r['kind']:read(Path(r['report'])) for r in read(prior/'evaluations.json') if r['step']==64}
        folder=output/'float_outputs';folder.mkdir();torch.cuda.reset_peak_memory_stats()
        for i,view in enumerate(dataset.views):
            budget.check();sample=move_sample(dataset[i],'cuda');source=Path(dataset.sources[sample['asset_id']]['path']).name
            kind='clean' if dataset.variants[view['variant_id']]['clean_pair'] else 'degraded'
            zp,_=refine_latent(model,sample,autocast_enabled=torch.cuda.is_bf16_supported())
            pred,pack=restore_pixels(bridge,sample,zp,grad=False,strength=cfg.model.output.strength)
            oracle,_=restore_pixels(bridge,sample,sample['z_target'],grad=False,strength=cfg.model.output.strength)
            actual=measure(sample,pred,zp,loss);reference=next(c for c in old[kind]['cases'] if c['view_id']==view['view_id'])
            mae_delta=actual['metrics']['rgb_global_mae']-reference['current']['metrics']['rgb_global_mae']
            require(abs(mae_delta)<1e-7,'Rerun deviates from the bound fixed evaluation')
            x0,y0,x1,y1=rois[source]['roi_xyxy'];mask=np.load(view['pad_valid_map']);require(mask[y0:y1,x0:x1].all(),'ROI intersects padding')
            def array(t):return t[0,:,0].detach().float().cpu().permute(1,2,0).numpy()[y0:y1,x0:x1].copy()
            arrays={'input':array(sample['x']),'target':array(sample['y']),'model':array(pred),'oracle':array(oracle)}
            if kind=='clean':require(np.array_equal(arrays['input'],arrays['oracle']),'Clean target-latent oracle must be identity')
            path=folder/f'{kind}_{Path(source).stem}.npz';np.savez_compressed(path,**arrays)
            record={'source':source,'kind':kind,'view_id':view['view_id'],'roi_xyxy':[x0,y0,x1,y1],
                'float_path':str(path),'float_sha256':file_sha256(path),'full_model':actual,
                'full_oracle':measure(sample,oracle,sample['z_target'],loss),'rerun_global_mae_delta':mae_delta,
                'bands':{label:frequency_metrics(arrays['input'],arrays['target'],arrays[label]) for label in ['model','oracle']}}
            records.append(record);print(canonical_json({'event':'float_case_completed','index':i,'kind':kind,'source':source}).decode(),flush=True)
        require(versions==[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models],'Parameters changed')
        require(not any(v.requires_grad or v.grad is not None for m in models for v in m.parameters()),'Trainable parameter or gradient detected')
        verify_protocol(p);torch.cuda.synchronize()
        result={'status':'completed_frozen_float_frequency_diagnostic','cases':records,'audit':audit,'runtime':execution_contract(),
            'optimizer_updates':0,'execution_guard':dict(guard),'all_parameters_frozen_and_unchanged':True,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'budget':budget.snapshot(),
            'protocol_sha256':file_sha256(output/'protocol.json'),'independent_validation':False,'trained_base_accepted':False}
        atomic_write(output/'results.json',canonical_json(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['declare','run'],required=True);a=parser.parse_args()
    if a.action=='declare':declare(a.output.resolve())
    else:run_diagnostic(a.output.resolve())
