"""New finite real-video adaptation with audited frozen INT8 H3 conditions."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from h3ce.config import load_config,write_resolved
from h3ce.components import sha256_file
from h3ce.train.checkpoint import CheckpointManager,TrainingBudget
from h3ce.train.preflight import seed_model
from h3ce.train.perceptual import load_perceptual
from h3ce.train.head_video_loss import target_transition_loss
from h3ce.data.continuous_head import paste_delta
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head3 import Tail3
from scripts.research_head_tail2 import strict_spatial,metrics
from scripts.train_head_condition_int8 import objective
from scripts.probe_narrator_transition_direction import load_case
from scripts.full_batch_solver import FullBatchState
from scripts import benchmark_swinir_batch8_strict_video as timing

PREFLIGHT=ROOT/'logs/narrator-transition-direction-20260910-v1'
RUN=ROOT/'runs/narrator-transition32-20260910-v1'
OUT=ROOT/'logs/narrator-transition32-20260910-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def calculate(delta,data,case,metric):
    x,y,support,plan,samples=data
    pred=torch.cat([paste_delta(x[i:i+1],delta[i:i+1],g) for i,g in enumerate(plan['transforms'])])
    terms=[]
    for i,s in enumerate(samples):
        a,b,c,d=s['geometry']['paste_xyxy'];terms.append(objective(pred[i:i+1,:,None,b:d,a:c],s,metric))
    spatial=torch.stack([t['total'] for t in terms]).mean()
    temporal=target_transition_loss(pred,y,support,case['pts'],[0]*22)['total']*(2. if case['clean'] else 1.)
    def avg(k):return torch.stack([t[k]*t['sample_weight'] for t in terms]).mean()
    effective={'rgb':avg('rgb'),'lighting':.2*avg('lighting'),'detail':.5*avg('detail'),
               'lpips':.05*avg('perceptual'),'motion':.1*temporal}
    return spatial+.1*temporal,spatial,temporal,effective,pred

def plot_losses():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(v) for v in (RUN/'metrics.jsonl').read_text().splitlines()]
    steps=[r['step'] for r in rows];values=np.array([r['total'] for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(steps,values,label='Actual fixed4-clip loss')
    smooth=np.array([np.mean(values[max(0,i-3):i+1]) for i in range(len(values))])
    axes[0].plot(steps,smooth,label='Trailing4 updates');axes[0].legend()
    for k in rows[0]['effective']:axes[1].plot(steps,[r['effective'][k] for r in rows],label=k)
    axes[1].legend();axes[1].set_yscale('log');axes[1].set_title('Actual weighted components')
    fixed=[]
    for p in sorted(RUN.glob('fixed_step_*/report.json')):
        o=read(p);rr=[r for r in o['rows'] if r['purpose']=='fit' and not r['clean']]
        fixed.append((o['step'],np.mean([r['transition_loss'] for r in rr])))
    axes[2].plot([x[0] for x in fixed],[x[1] for x in fixed],'o-');axes[2].set_title('Measured fixed degraded-fit motion loss')
    for ax in axes:ax.set_xlabel('Update');ax.grid(alpha=.2)
    fig.suptitle('Real-video256 head adaptation; frozen INT8 H3 conditions; no interpolated evaluations')
    fig.tight_layout();fig.savefig(RUN/'losses.png',dpi=150);plt.close(fig)

def main():
    assert sys.argv[1:]==['--run'] and not RUN.exists() and not OUT.exists()
    audit=read(PREFLIGHT/'cpu_review/review.json');assert audit['passed']
    pre=read(PREFLIGHT/'protocol.json');result=read(PREFLIGHT/'summary.json');assert result['passed'] and result['updates']==0
    for f,h in result['artifacts'].items():assert sha256_file(f)==h,f
    for f,h in audit['bindings'].items():assert sha256_file(f)==h,f
    RUN.mkdir();OUT.mkdir();torch.set_num_threads(4);seed_model(42);timing.OUT=OUT
    files=[Path(__file__),PREFLIGHT/'protocol.json',PREFLIGHT/'summary.json',PREFLIGHT/'cpu_review/review.json',
        ROOT/'docs/NARRATOR_VIDEO_TRANSITION32_PROTOCOL_20260910.md',ROOT/'h3ce/train/head_video_loss.py',
        ROOT/'scripts/probe_narrator_transition_direction.py',ROOT/'scripts/research_naf_head3.py',ROOT/'scripts/train_head_condition_int8.py']
    protocol={'authorization':'Active ongoing research; audited new32-step video supervision protocol; no old optimizer resumed',
        'cases':pre['cases'],'source_groups':1,'steps_max':32,'head_side':256,'working_hw':[810,1536],
        'loss':'spatial RGB1+light.2+detail.5+LPIPS.05, original-coordinate temporal.1, clean2, four complete fit clips equally weighted',
        'checkpoint_start':pre['checkpoint'],'preflight_sha256':sha256_file(PREFLIGHT/'summary.json'),
        'counts_max':{'tail':146,'lpips':3212,'backward':128,'updates':32,'h3':0,'backbone_gpu':0},
        'evaluation_steps':[0,8,32],'cpu_constructor_shape_forwards':1,'bindings':{str(f):sha256_file(f) for f in files}}
    save(OUT/'protocol.json',protocol);counts={k:0 for k in protocol['counts_max']};step=0;optimizer=None
    def inc(k):counts[k]+=1;assert counts[k]<=protocol['counts_max'][k],counts
    cfg=load_config(ROOT/'configs/project.int8.yaml');write_resolved(cfg,RUN/'resolved.yaml')
    save(RUN/'execution_override.json',{'source_config':'resolved.yaml','actual_model':'199747 existing NAF tail parameters',
        'actual_h3':'Frozen upstream INT8 condition cache from real22 videos; no new H3 calls','h3_contract':read(ROOT/'runs/narrator-video-data-20260910-v2/h3_contract.json'),
        'loss_override':protocol['loss'],'learning_rate':1e-5,'steps_max':32,'latent_loss_enabled':False})
    contract={'kind':'naf3_real_video_transition32','protocol_sha256':sha256_file(OUT/'protocol.json'),'parameters':199747}
    save(RUN/'training_contract.json',contract);manager=CheckpointManager(ROOT/'runs',RUN,contract=contract)
    with TrainingBudget(ROOT/'runs',172800,phase='real_video_transition32') as budget:
      try:
        timing.wait_idle('before_load');base,_=load_model();tail=Tail3(base).cuda().eval().requires_grad_(True);del base
        gradient=torch.load(PREFLIGHT/'gradient_and_hypothesis.pt',map_location='cpu',weights_only=True)
        tail.load_state_dict(gradient['initial'],strict=True);params=dict(tail.named_parameters());assert sum(p.numel() for p in params.values())==199747
        metric=load_perceptual(cfg,ROOT,force=True)
        tail.register_forward_pre_hook(lambda m,a:inc('tail'));metric.register_forward_pre_hook(lambda m,a:inc('lpips'))
        metric_versions={n:(id(p),p._version) for n,p in metric.named_parameters()}
        cases=protocol['cases'];cache=[];negative=[];data=[]
        for i,case in enumerate(cases):
            cache.append({k:v.cuda() for k,v in torch.load(PREFLIGHT/f'features_{i}.pt',map_location='cpu',weights_only=True).items()})
            negative.append(torch.from_numpy(np.load(PREFLIGHT/f'reference_{i}.npy')).cuda());data.append(load_case(case,'cuda'))
        fit=[i for i,c in enumerate(cases) if c['purpose']=='fit'];assert len(fit)==4;sampler=FullBatchState(fit)
        def checkpoint(complete=False):
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            return manager.save(model=tail,optimizer=optimizer,scheduler=None,scaler=None,sampler=sampler,stage='video_transition32',step=step,
                budget=budget.snapshot(),resolved_config=cfg.model_dump(mode='json'),extra={'counts':dict(counts),'phase_complete':complete,'accepted_base':False})
        cp=checkpoint();save(RUN/'initial_checkpoint.json',{'path':str(cp),'sha256':sha256_file(cp)})
        def fixed():
            dest=RUN/f'fixed_step_{step:04d}';dest.mkdir();rows=[]
            with torch.no_grad(),strict_spatial():
                for i,case in enumerate(cases):
                    delta=tail(**cache[i])-negative[i];total,sp,tm,parts,pred=calculate(delta,data[i],case,metric)
                    vals=[];iv=[]
                    for j,s in enumerate(data[i][4]):
                        a,b,c,d=s['geometry']['paste_xyxy'];vals.append(metrics(pred[j:j+1,:,None,b:d,a:c],s));iv.append(metrics(s['x'],s))
                    row={'index':i,'clip_start':case['clip_start'],'kind':case['kind'],'purpose':case['purpose'],'clean':case['clean'],
                        'joint_loss':float(total),'spatial_loss':float(sp),'transition_loss':float(tm),
                        'metrics':{k:float(np.mean([v[k] for v in vals])) for k in vals[0]},
                        'input_metrics':{k:float(np.mean([v[k] for v in iv])) for k in iv[0]}}
                    rows.append(row);np.save(dest/f'delta_{i}.npy',delta.cpu().numpy());del delta,total,sp,tm,parts,pred
            save(dest/'report.json',{'step':step,'rows':rows,'checkpoint':str(cp),'checkpoint_sha256':sha256_file(cp)})
            return rows
        def average(rows,k):return float(np.mean([r[k] for r in rows if r['purpose']=='fit']))
        initial=fixed();old=read(PREFLIGHT/'initial.json')
        for a,b in zip(initial,old):
            for k in ('joint_loss','spatial_loss','transition_loss'):assert abs(a[k]-b[k])<=2e-7,(k,a[k],b[k])
        optimizer=torch.optim.AdamW(params.values(),lr=1e-5,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();stopped=False
        for step in range(1,33):
            optimizer.zero_grad(set_to_none=True);records=[]
            with strict_spatial():
                for i in fit:
                    delta=tail(**cache[i])-negative[i];loss,sp,tm,parts,pred=calculate(delta,data[i],cases[i],metric)
                    assert torch.isfinite(loss);(loss/4).backward();inc('backward')
                    records.append({'total':float(loss.detach()),'effective':{k:float(v.detach()) for k,v in parts.items()}})
                    del delta,loss,sp,tm,parts,pred
            if step==1:
                expected={k:gradient['spatial'][k]+.1*gradient['temporal'][k] for k in params}
                error={k:params[k].grad.cpu()-expected[k] for k in params}
                rel=float(torch.sqrt(sum(e.square().sum() for e in error.values())/sum(e.square().sum() for e in expected.values())))
                maximum=max(float(e.abs().max()) for e in error.values());assert rel<=1e-5 and maximum<=1e-6,(rel,maximum)
                save(RUN/'first_gradient_check.json',{'relative_l2':rel,'maximum':maximum,'preflight_gradient_matched':True})
            norm=torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
            optimizer.step();inc('updates');sampler.completed_calls=step
            if step==1:
                maximum=max(float(abs(params[k].detach().cpu()-gradient['hypothesis'][k]).max()) for k in params)
                assert maximum<=1e-7,maximum
                save(RUN/'first_update_check.json',{'maximum_difference_from_audited_hypothesis':maximum,'passed':True})
            rec={'step':step,'total':float(np.mean([r['total'] for r in records])),
                'effective':{k:float(np.mean([r['effective'][k] for r in records])) for k in records[0]['effective']},
                'gradient_norm_before_clip':float(norm),'optimizer_updated':True,'fit_clips':4,'real_frames_per_update':88}
            assert abs(rec['total']-sum(rec['effective'].values()))<=2e-7
            with (RUN/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
            if step in (8,32):
                cp=checkpoint(complete=step==32);rows=fixed()
                clean=float(np.mean([r['metrics']['rgb_mae'] for r in rows if r['clean']]))
                safe=clean<=.001 and all(r['clean'] or r['metrics']['rgb_mae']<=1.05*r['input_metrics']['rgb_mae'] for r in rows)
                safe=safe and average(rows,'joint_loss')<average(initial,'joint_loss')
                plot_losses();print(f'Update{step}: joint={average(rows,"joint_loss"):.8f}, clean={clean:.8f}, protection={safe}',flush=True)
                if not safe:stopped=True;break
            budget.check()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert metric_versions=={n:(id(p),p._version) for n,p in metric.named_parameters()}
        if not stopped:assert counts==protocol['counts_max'],counts
        save(RUN/'training_report.json',{'status':'stopped_at_protection' if stopped else 'completed32','steps':step,'counts':counts,
            'training_loop_with_evaluation_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'checkpoint':{'path':str(cp),'sha256':sha256_file(cp)},'new_h3_calls':0,'new_backbone_gpu_calls':0,
            'accepted_base':False,'budget':budget.snapshot(),'artifacts':{str(p):sha256_file(p) for p in RUN.rglob('*') if p.is_file()}})
      except BaseException as e:
        if 'tail' in locals() and 'sampler' in locals():
            try:cp=checkpoint()
            except Exception:pass
        save(RUN/'failure.json',{'error':repr(e),'steps':counts['updates'],'counts':counts,'automatic_restart':False})
        raise

if __name__=='__main__':main()
