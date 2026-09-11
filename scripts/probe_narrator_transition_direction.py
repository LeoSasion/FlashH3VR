"""Finite real-video spatial/motion gradient audit; no optimizer or weight writes."""
from pathlib import Path
import sys,json,math,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from h3ce.components import sha256_file
from h3ce.config import load_config
from h3ce.train.checkpoint import TrainingBudget,CheckpointManager
from h3ce.train.perceptual import load_perceptual
from h3ce.train.head_video_loss import target_transition_loss
from h3ce.data.continuous_head import paste_delta
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head3 import NAFHead3
from scripts.research_head_tail2 import strict_spatial,metrics
from scripts.train_head_condition_int8 import objective
from scripts import benchmark_swinir_batch8_strict_video as timing

DATA=ROOT/'runs/narrator-video-data-20260910-v2'
PARENT=ROOT/'runs/head-condition-int8-20260910-v1'
OUT=ROOT/'logs/narrator-transition-direction-20260910-v1'
WEIGHT=.1
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def load_case(case,device):
    dest=Path(case['directory'])
    x=torch.from_numpy(np.load(case['input'])).permute(0,3,1,2).to(device)
    y=torch.from_numpy(np.load(case['target'])).permute(0,3,1,2).to(device)
    support=torch.from_numpy(np.load(dest/'support.npy')).to(device)
    plan=read(dest/'geometry.json');det=read(dest/'detections.json');samples=[]
    for i,g in enumerate(plan['transforms']):
        a,b,c,d=g['paste_xyxy'];fa,fb,fc,fd=det[i]['face_xyxy']
        face=torch.zeros(1,1,1,d-b,c-a,device=device)
        face[...,max(0,math.floor(fb)-b):min(d-b,math.ceil(fd)-b),max(0,math.floor(fa)-a):min(c-a,math.ceil(fc)-a)]=1
        assert face.sum()>0
        samples.append({'x':x[i:i+1,:,None,b:d,a:c],'y':y[i:i+1,:,None,b:d,a:c],
            'valid':torch.ones_like(face),'face_mask':face,'clean':case['clean'],'geometry':g})
    return x,y,support,plan,samples

def evaluate_delta(delta,case,metric):
    x,y,support,plan,samples=load_case(case,delta.device)
    prediction=torch.cat([paste_delta(x[i:i+1],delta[i:i+1],g) for i,g in enumerate(plan['transforms'])])
    terms=[];values=[];input_values=[]
    for i,s in enumerate(samples):
        a,b,c,d=s['geometry']['paste_xyxy'];pred=prediction[i:i+1,:,None,b:d,a:c]
        terms.append(objective(pred,s,metric)['total']);values.append(metrics(pred,s));input_values.append(metrics(s['x'],s))
    spatial=torch.stack(terms).mean()
    temporal=target_transition_loss(prediction,y,support,case['pts'],[0]*22)['total']*(2. if case['clean'] else 1.)
    row={'clip_start':case['clip_start'],'kind':case['kind'],'clean':case['clean'],'purpose':case['purpose'],
         'spatial_loss':float(spatial.detach()),'transition_loss':float(temporal.detach()),
         'joint_loss':float((spatial+WEIGHT*temporal).detach()),
         'metrics':{k:float(np.mean([v[k] for v in values])) for k in values[0]},
         'input_metrics':{k:float(np.mean([v[k] for v in input_values])) for k in input_values[0]}}
    return spatial,temporal,row

def main():
    assert sys.argv[1:]==['--run'] and not OUT.exists()
    p=read(DATA/'data_protocol.json');review=ROOT/'logs/narrator-video-data-20260910-v2/review.json'
    assert read(review)['status']=='passed66_source132_conditions264_buckets126_transitions_cpu_review'
    parent=read(PARENT/'training_report.json');checkpoint=parent['checkpoint']
    assert parent['steps']==64 and sha256_file(checkpoint['path'])==checkpoint['sha256']
    for case in p['cases']:
        for f,h in case['artifacts'].items():assert sha256_file(f)==h,f
    OUT.mkdir();torch.set_num_threads(4);timing.OUT=OUT
    files=[Path(__file__),DATA/'data_protocol.json',DATA/'h3_contract.json',review,PARENT/'training_report.json',PARENT/'training_contract.json',
        ROOT/'h3ce/train/head_video_loss.py',ROOT/'h3ce/data/continuous_head.py',ROOT/'scripts/research_naf_head3.py',
        ROOT/'scripts/research_nafnet_gopro32.py',ROOT/'scripts/train_head_condition_int8.py',ROOT/'scripts/research_head_tail2.py']
    protocol={'authorization':'Active ongoing optimization; new real-video direction precheck, no completed64 run resumed',
        'cases':p['cases'],'checkpoint':checkpoint,'head_side':256,'source_groups':1,'updates':0,
        'loss':'Per-frame original ROI RGB1+light.2+detail.5+LPIPS.05 plus0.1 original-canvas target-transition Charbonnier; clean2 for both',
        'fit_cases':4,'same_source_check_cases':2,'gradient':'Separate spatial and temporal autograd.grad per fit clip, each mean over4 fit clips',
        'hypothesis':'One functional first freshAdam step, LR1e-5 clip1 epsilon1e-8, from frozen-recorded static64 candidate; no parameter writes, no optimizer',
        'gates':'Joint direction descent dot products positive for spatial and temporal; actual hypothesis mean-fit spatial and transition both decrease; clean mean <=.001 and every degraded RGB <=1.05 input; same-source check reported separately',
        'counts_max':{'backbone':6,'reference':6,'tail':16,'lpips':352,'autograd_grad':8,'h3':0,'updates':0},
        'cache':'Only fixed input H3/NAF conditions with exact binding; no target supplied to features',
        'bindings':{str(f):sha256_file(f) for f in files}}
    save(OUT/'protocol.json',protocol);counts={k:0 for k in protocol['counts_max']}
    def inc(k):counts[k]+=1;assert counts[k]<=protocol['counts_max'][k],counts
    with TrainingBudget(ROOT/'runs',172800,phase='real_video_transition_direction') as budget:
        timing.wait_idle('before_load');cfg=load_config(ROOT/'configs/project.int8.yaml')
        base,info=load_model();model=NAFHead3(base).cuda().eval();metric=load_perceptual(cfg,ROOT,force=True)
        manager=CheckpointManager(ROOT/'runs',PARENT,contract=read(PARENT/'training_contract.json'))
        state=manager.read(Path(checkpoint['path']));assert state['step']==64
        model.tail.load_state_dict(state['model'],strict=True);del state
        params=dict(model.tail.named_parameters());assert sum(p.numel() for p in params.values())==199747
        before={k:v.detach().clone() for k,v in params.items()}
        versions=[{n:(id(p),p._version) for n,p in m.named_parameters()} for m in (base,model.reference,metric)]
        for k,m in [('backbone',base),('reference',model.reference),('tail',model.tail),('lpips',metric)]:
            m.register_forward_pre_hook(lambda m,a,k=k:inc(k))
        cache=[];baseline=[];negative=[]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),strict_spatial():
            for i,case in enumerate(p['cases']):
                condition=torch.from_numpy(np.load(Path(case['directory'])/'condition.npy')).cuda()
                c=model.features(condition);ref=model.reference(c['features'],c['skip_mid'],c['skip_full'])
                assert torch.equal(ref,c['official_ending'])
                c={k:c[k] for k in ('features','skip_mid','skip_full')};cache.append(c);negative.append(ref)
                torch.save({k:v.cpu() for k,v in c.items()},OUT/f'features_{i}.pt');np.save(OUT/f'reference_{i}.npy',ref.cpu().numpy())
                delta=model.tail(**c)-ref;sp,motion,row=evaluate_delta(delta,case,metric);baseline.append(row)
                np.save(OUT/f'initial_delta_{i}.npy',delta.cpu().numpy());del condition,delta,sp,motion
                print(f'Initial real-video case{i+1}/6 checked',flush=True);budget.check()
        save(OUT/'initial.json',baseline)
        gp={k:torch.zeros_like(v) for k,v in params.items()};gm={k:torch.zeros_like(v) for k,v in params.items()}
        with strict_spatial():
            for i,case in enumerate(p['cases']):
                if case['purpose']!='fit':continue
                delta=model.tail(**cache[i])-negative[i];sp,motion,_=evaluate_delta(delta,case,metric)
                ap=torch.autograd.grad(sp,tuple(params.values()),retain_graph=True);inc('autograd_grad')
                am=torch.autograd.grad(motion,tuple(params.values()));inc('autograd_grad')
                for (k,_),a,b in zip(params.items(),ap,am):
                    assert torch.isfinite(a).all() and torch.isfinite(b).all();gp[k]+=a.detach()/4;gm[k]+=b.detach()/4
                del delta,sp,motion,ap,am;budget.check()
        joint={k:gp[k]+WEIGHT*gm[k] for k in gp};norm=torch.sqrt(sum(g.square().sum() for g in joint.values()))
        assert torch.isfinite(norm) and norm>0;clip=min(1.,1./(float(norm)+1e-6))
        hypothesis={k:before[k]-1e-5*(joint[k]*clip)/((joint[k]*clip).abs()+1e-8) for k in params}
        torch.save({'spatial':{k:v.cpu() for k,v in gp.items()},'temporal':{k:v.cpu() for k,v in gm.items()},
                    'initial':{k:v.cpu() for k,v in before.items()},'hypothesis':{k:v.cpu() for k,v in hypothesis.items()}},OUT/'gradient_and_hypothesis.pt')
        proposed=[]
        with torch.no_grad(),strict_spatial():
            for i,case in enumerate(p['cases']):
                delta=torch.func.functional_call(model.tail,hypothesis,(),cache[i])-negative[i]
                sp,motion,row=evaluate_delta(delta,case,metric);proposed.append(row)
                np.save(OUT/f'hypothesis_delta_{i}.npy',delta.cpu().numpy());del delta,sp,motion
                print(f'Functional one-step hypothesis{i+1}/6 checked',flush=True);budget.check()
        save(OUT/'hypothesis.json',proposed)
        def avg(rows,k):return float(np.mean([r[k] for r in rows if r['purpose']=='fit']))
        dp=float(sum((gp[k]*joint[k]).sum() for k in gp));dm=float(sum((gm[k]*joint[k]).sum() for k in gm))
        clean=float(np.mean([r['metrics']['rgb_mae'] for r in proposed if r['clean']]))
        gates={'spatial_direction':dp>0,'temporal_direction':dm>0,
            'fit_spatial_decrease':avg(proposed,'spatial_loss')<avg(baseline,'spatial_loss'),
            'fit_temporal_decrease':avg(proposed,'transition_loss')<avg(baseline,'transition_loss'),
            'clean_mae':clean<=.001,'all_degraded_rgb':all(r['clean'] or r['metrics']['rgb_mae']<=1.05*r['input_metrics']['rgb_mae'] for r in proposed)}
        assert all(torch.equal(v,before[k]) and v.grad is None for k,v in params.items())
        assert versions==[{n:(id(p),p._version) for n,p in m.named_parameters()} for m in (base,model.reference,metric)]
        assert counts==protocol['counts_max'],counts
        files=[f for f in OUT.iterdir() if f.is_file()]
        save(OUT/'summary.json',{'status':'completed_zero_update_real_video_direction_precheck','counts':counts,'gates':gates,'passed':all(gates.values()),
            'spatial_dot_joint':dp,'temporal_dot_joint':dm,'joint_gradient_norm':float(norm),'clean_mae':clean,
            'initial_fit':{k:avg(baseline,k) for k in ('spatial_loss','transition_loss','joint_loss')},
            'hypothesis_fit':{k:avg(proposed,k) for k in ('spatial_loss','transition_loss','joint_loss')},
            'all_weights_unchanged':True,'h3_cache_reused':True,'h3_calls':0,'optimizer_constructions':0,'updates':0,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'accepted_base':False,'budget':budget.snapshot(),
            'artifacts':{str(f):sha256_file(f) for f in files}})
        print({'gates':gates,'updates':0,'counts':counts},flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'automatic_restart':False,'updates':0})
        raise
