"""New bounded static head-H3 condition adaptation from official pretrained NAF."""
from pathlib import Path
import sys,time,json,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from h3ce.cache.keys import file_sha256
from h3ce.config import load_config,write_resolved
from h3ce.data.continuous_head import head_transform,pack_frame,inverse_delta,crop_feather
from h3ce.vae.int8_kitchen_backend import Int8KitchenH3Backend,runtime
from h3ce.vae.bridge import FrameMeta,H3VAEBridge
from h3ce.train.checkpoint import TrainingBudget,CheckpointManager
from h3ce.train.preflight import seed_model
from h3ce.train.perceptual import load_perceptual
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head3 import NAFHead3
from scripts.research_head_tail2 import strict_spatial,metrics,objective as pixel_objective
from scripts.full_batch_solver import FullBatchState
from scripts.plot_training_losses import plot
from scripts.benchmark_h3ce_native_video import read
from scripts.benchmark_native_head_fastpath import save
from scripts import benchmark_swinir_batch8_strict_video as timing

DATA=ROOT/'runs/naf-head256-data-20260910-v1'
RUN=ROOT/'runs/head-condition-int8-20260910-v1'
OUT=ROOT/'logs/head-condition-int8-20260910-v1'

def sample(case,device):
    with np.load(case['path']) as z:
        fullx,fully=(torch.from_numpy(z[k].copy()).permute(2,0,1)[None].to(device) for k in ('x','y'))
        face=torch.from_numpy(z['face_mask'].copy()).to(device)
    g=head_transform(case['geometry']['face_xyxy'],tuple(fullx.shape[-2:]),frame_index=0,pts=0.,side=256,provenance='source_detector')
    a,b,c,d=g['paste_xyxy']
    return {'fullx':fullx,'fully':fully,'x':fullx[...,b:d,a:c][:,:,None],'y':fully[...,b:d,a:c][:,:,None],
        'valid':torch.ones(1,1,1,d-b,c-a,device=device),'face_mask':face[b:d,a:c][None,None,None],
        'clean':case['clean'],'geometry':g}

def predict(delta,s):
    return s['x']+(inverse_delta(delta,s['geometry'])*crop_feather(s['geometry'],device=delta.device))[:,:,None]

def objective(pred,s,metric):
    terms=pixel_objective(pred,s)
    perceptual=metric(pred[:,:,0].clamp(0,1)*2-1,s['y'][:,:,0].clamp(0,1)*2-1).mean()
    raw=terms['unweighted_total']+.05*perceptual
    return {**terms,'perceptual':perceptual,'pixel_unweighted_total':terms['unweighted_total'],
        'unweighted_total':raw,'total':raw*terms['sample_weight']}

def mean_fit(rows):return float(np.mean([r['losses']['total'] for r in rows if r['purpose']=='fit']))

def protects(rows):
    return all(np.mean([r['metrics']['rgb_mae'] for r in rows if r['purpose']==g and r['clean']])<=.001
        for g in ('fit','unfitted_same_episode_check')) and all(r['clean'] or r['metrics']['rgb_mae']<=1.05*r['input_metrics']['rgb_mae'] for r in rows)

def main():
    assert sys.argv[1:]==['--run'] and not RUN.exists() and not OUT.exists()
    RUN.mkdir();OUT.mkdir();torch.set_num_threads(4);seed_model(42)
    data=read(DATA/'protocol.json');cases=data['cases'];assert len(cases)==32
    for f,h in data['bindings'].items():assert file_sha256(f)==h,f
    for c in cases:assert file_sha256(c['path'])==c['sha256'] and c['source']['split']=='train'
    assert len({c['source']['source_group'] for c in cases})==1
    files=[Path(__file__),DATA/'protocol.json',ROOT/'scripts/research_naf_head3.py',ROOT/'scripts/research_nafnet_gopro32.py',
        ROOT/'h3ce/vae/int8_kitchen_backend.py',ROOT/'h3ce/vae/convrot_cuda.py',ROOT/'configs/convrot_cuda_runtime.lock.json',ROOT/'logs/kijai-int8-runtime-20260910-v1/acquisition.json',ROOT/'h3ce/data/continuous_head.py',ROOT/'h3ce/vae/current.py',ROOT/'h3ce/vae/factory.py',ROOT/'h3ce/vae/int8_convrot_backend.py',
        ROOT/'h3ce/vae/bridge.py',ROOT/'configs/project.int8.yaml',ROOT/'configs/components.int8.lock.json',
        ROOT/'scripts/research_head_tail2.py',ROOT/'scripts/detail_supervision_math.py',ROOT/'scripts/plot_training_losses.py',
        ROOT/'docs/HEAD_CONDITION_INT8_PROTOCOL_20260910.md',ROOT/'models/nafnet_gopro32_baseline/provenance.json']+[Path(c['path']) for c in cases]
    p={'authorization':'Active ongoing experiments/optimization goal; new finite protocol, no old run resumed',
        'cases':cases,'steps_max':64,'learning_rate':1e-5,'head_side':256,'train_originals':12,'same_episode_check_originals':4,
        'source_groups':1,'ai':0,'h3':'User INT8 ConvRot source, full pinned Comfy Kitchen CUDA INT8 frozen image native H3 on continuous256 head buckets; research override recorded separately from shared configuration',
        'architecture':'Official pretrained last three decoder levels, last two ups and ending; frozen same-shape reference',
        'loss':'RGB1+light.2+original_detail.5+LPIPS.05, clean2, latent0; original-coordinate ROI without artificial padding',
        'evaluation_steps':[0,8,32,64],'counts_max':{'encode':32,'decode':32,'backbone':4,'tail':70,'reference':74,'lpips':1720,'vjp':1,'backward':64,'updates':64},
        'bindings':{str(f):file_sha256(f) for f in files}}
    save(OUT/'protocol.json',p);counts={k:0 for k in p['counts_max']};step=0;checkpoint=None;timing.OUT=OUT
    def inc(k):counts[k]+=1;assert counts[k]<=p['counts_max'][k],counts
    def wrap(k,fn):
        def call(*a,**kw):inc(k);return fn(*a,**kw)
        return call
    with TrainingBudget(ROOT/'runs',172800,phase='head_condition_int8_adaptation64') as budget:
      try:
        timing.wait_idle('before_load');start=time.perf_counter();cfg=load_config(ROOT/'configs/project.int8.yaml')
        cd=cfg.model_dump(mode='json');cd['training']['losses'].update(latent=0.,perceptual=.05,lighting_target=.2);cfg=type(cfg).model_validate(cd)
        write_resolved(cfg,RUN/'resolved.yaml');cuda,_,runtime_identity=runtime()
        bridge=H3VAEBridge(Int8KitchenH3Backend.from_locked(project_root=ROOT,weights='models/minimax_h3_video_vae_int8_convrot.safetensors',components_lock='configs/components.int8.lock.json'));codec=bridge.current_codec_pack()
        save(RUN/'execution_override.json',{'base_resolved_config':'resolved.yaml','actual_h3_backend':'Int8KitchenH3Backend','compute_mode':'comfy_kitchen_cuda_int8','runtime':runtime_identity,'reason':'Explicit research provider not yet in general CLI factory; H3 frozen, no surrogate backward','native_contract':bridge.inspect_contract().native})
        base,info=load_model();model=NAFHead3(base).cuda().eval();metric=load_perceptual(cfg,ROOT,force=True)
        nparams=sum(v.numel() for v in model.tail.parameters())
        for k,m in [('backbone',base),('tail',model.tail),('reference',model.reference),('lpips',metric)]:m.register_forward_pre_hook(lambda m,a,k=k:inc(k))
        bridge.backend.encode_mean_raw=wrap('encode',bridge.backend.encode_mean_raw);bridge.backend.decode_raw=wrap('decode',bridge.backend.decode_raw)
        frozen=[base,model.reference,metric];versions=lambda:[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in frozen]
        initial_versions=versions();h3_versions={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
        save(OUT/'load.json',{'seconds':time.perf_counter()-start,'trainable_parameters':nparams,'base_parameters':info['parameters'],
            'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'encoder_id':bridge.encoder_id,
            'decoder_id':codec.effective_decoder_hash,'h3_contract':bridge.inspect_contract().native,'constructor_cpu_shape_forwards':1})
        timing.wait_idle('before_compute');samples=[sample(c,'cuda') for c in cases];conditions=[];native_rows=[]
        cache_dir=RUN/'native_cache';cache_dir.mkdir();save(RUN/'geometry.json',[s['geometry'] for s in samples])
        for i,s in enumerate(samples):
            bucket=pack_frame(s['fullx'],s['geometry']);z=bridge.encode_rgb(bucket[:,:,None],FrameMeta('image',(0.,)))
            n=bridge.decode_latent(z,grad=False,codec_pack=codec);condition=.5*(bucket+n[:,:,0].clamp(0,1));conditions.append(condition)
            f=cache_dir/f'case_{i:02d}.npz';np.savez_compressed(f,bucket=bucket.cpu().numpy(),native=n.cpu().numpy(),condition=condition.cpu().numpy())
            native_rows.append({'index':i,'path':str(f),'sha256':file_sha256(f)})
            del z,n,bucket,condition;budget.check()
            if (i+1)%8==0:print(f'Fresh head-H3 image conditions {i+1}/32',flush=True)
        assert h3_versions=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
        assert all(not v.requires_grad and v.grad is None for v in bridge.backend.model.parameters())
        linears=[m for m in bridge.backend.model.modules() if type(m).__name__=='KitchenInt8Linear'];assert len(linears)==144 and sum(m.calls for m in linears)==4608
        save(RUN/'h3_compute_counts.json',{'encode':counts['encode'],'decode':counts['decode'],'actual_upstream_int8_linears':sum(m.calls for m in linears),'input_gradient_requested':False,'h3_updates':0})
        del linears
        save(RUN/'native_cache.json',{'files':native_rows,'encoder_id':bridge.encoder_id,'decoder_id':codec.effective_decoder_hash,'h3_frozen_unchanged':True})
        del bridge;gc.collect();conditions=torch.cat(conditions);cache={k:[] for k in ('features','skip_mid','skip_full')}
        with torch.no_grad(),strict_spatial():
            for i in range(0,32,8):
                ca=model.features(conditions[i:i+8]);ref=model.reference(ca['features'],ca['skip_mid'],ca['skip_full'])
                assert torch.equal(ref,ca['official_ending']),'Copied tail must exactly reproduce official pretrained output'
                for k in cache:cache[k].append(ca[k])
            cache={k:torch.cat(v) for k,v in cache.items()}
        torch.save({k:v.cpu() for k,v in cache.items()},RUN/'frozen_features.pt');del conditions,ca,ref
        train=[i for i,c in enumerate(cases) if c['purpose']=='fit'];assert len(train)==24
        params=tuple(model.tail.parameters());optimizer=None;sampler=FullBatchState(train)
        contract={'kind':'naf_head_three_level_head_h3_condition256','protocol_sha256':file_sha256(OUT/'protocol.json'),'trainable_parameters':nparams}
        save(RUN/'training_contract.json',contract);manager=CheckpointManager(ROOT/'runs',RUN,contract=contract)
        def ckpt(complete=False):
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            return manager.save(model=model.tail,optimizer=optimizer,scheduler=None,scaler=None,sampler=sampler,
                stage='head_condition64',step=step,budget=budget.snapshot(),resolved_config=cfg.model_dump(mode='json'),
                extra={'counts':dict(counts),'phase_complete':complete,'accepted_base':False})
        checkpoint=ckpt();save(RUN/'initial_checkpoint.json',{'path':str(checkpoint),'sha256':file_sha256(checkpoint)})
        def fixed(label,trial=None):
            dest=RUN/label;dest.mkdir();rows=[]
            with torch.no_grad(),strict_spatial():
                ds=model.delta(**cache,parameters=trial)
                for i,s in enumerate(samples):
                    pred=predict(ds[i:i+1],s);assert torch.isfinite(pred).all()
                    if label=='fixed_step_0000':assert torch.equal(pred,s['x']) and torch.count_nonzero(ds[i])==0
                    terms=objective(pred,s,metric)
                    rows.append({'index':i,'head_index':cases[i]['head_index'],'purpose':cases[i]['purpose'],'clean':s['clean'],
                        'metrics':metrics(pred,s),'input_metrics':metrics(s['x'],s),'losses':{k:float(v) for k,v in terms.items()}})
                    np.savez_compressed(dest/f'case_{i:02d}.npz',prediction=pred.cpu().numpy(),delta=ds[i:i+1].cpu().numpy())
            save(dest/'report.json',{'step':step,'rows':rows,'checkpoint_sha256':file_sha256(checkpoint)});return rows
        def training_loss():
            ds=model.delta(**{k:v[train] for k,v in cache.items()});terms=[objective(predict(ds[j:j+1],samples[i]),samples[i],metric) for j,i in enumerate(train)]
            return sum(t['total'] for t in terms)/24,terms
        with strict_spatial():
            initial=fixed('fixed_step_0000');before={n:v.detach().clone() for n,v in model.tail.named_parameters()}
            loss,terms=training_loss();grads=torch.autograd.grad(loss,params);inc('vjp')
            assert all(torch.isfinite(g).all() for g in grads)
            norm=torch.sqrt(sum(g.square().sum() for g in grads));assert norm>0;clip=min(1.,1./(float(norm)+1e-6))
            trial={n:v.detach()-1e-5*(g*clip)/((g*clip).abs()+1e-8) for (n,v),g in zip(model.tail.named_parameters(),grads)}
            torch.save({'gradient':{n:g.cpu() for (n,_),g in zip(model.tail.named_parameters(),grads)},'trial':{n:v.cpu() for n,v in trial.items()}},RUN/'precheck_gradient.pt')
            hypothetical=fixed('precheck_hypothetical',trial);passed=mean_fit(hypothetical)<mean_fit(initial) and protects(hypothetical)
            assert all(torch.equal(v,before[n]) for n,v in model.tail.named_parameters())
            save(RUN/'precheck.json',{'passed':passed,'actual_parameter_vjp':1,'functional_hypotheses':1,'optimizer_updates':0,
                'gradient_norm':float(norm),'fit_initial_loss':mean_fit(initial),'fit_hypothetical_loss':mean_fit(hypothetical),
                'weights_unchanged':True,'protection_passed':protects(hypothetical)})
            del loss,terms,grads,trial,before
            if not passed:
                save(RUN/'training_report.json',{'status':'precheck_failed_no_training','steps':0,'counts':counts,'accepted_base':False,'budget':budget.snapshot()})
                print('Precheck failed, no updates',flush=True);return
            print(f'Pretrained {nparams} parameters: identity, VJP and first-step protections passed',flush=True)
            optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.);torch.cuda.synchronize();train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats();stopped=False
            for step in range(1,65):
                optimizer.zero_grad(set_to_none=True);loss,terms=training_loss();assert torch.isfinite(loss)
                loss.backward();inc('backward');gradnorm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();inc('updates');sampler.completed_calls=step
                def avg(k):return sum(float(t[k].detach()*t['sample_weight']) for t in terms)/24
                rgb,detail=avg('rgb'),avg('detail')
                record={'step':step,'phase':'pixel','optimizer_updated':True,'losses':{'total':float(loss.detach()),
                    'rgb':rgb+.5*detail,'lighting_target':avg('lighting'),'perceptual':avg('perceptual'),'latent':0.},
                    'component_details':{'sample_weighted_rgb':rgb,'sample_weighted_detail':detail,'detail_coefficient':.5,'latent_enabled':False},
                    'gradient_norm_before_clip':float(gradnorm)}
                with (RUN/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n');f.flush()
                del loss,terms;budget.check()
                if step in (8,32,64):
                    checkpoint=ckpt();evaluated=fixed(f'fixed_step_{step:04d}');plot([RUN],RUN/'losses.png')
                    print(f'Head condition adaptation {step}/64 fit_loss={mean_fit(evaluated):.8f}',flush=True)
                    if mean_fit(evaluated)>=mean_fit(initial) or not protects(evaluated):stopped=True;print('Stopped at declared protection',flush=True);break
            checkpoint=ckpt(True);plot([RUN],RUN/'losses.png');torch.cuda.synchronize()
            seconds=time.perf_counter()-train_start;peak=torch.cuda.max_memory_allocated()
        assert versions()==initial_versions and all(not v.requires_grad and v.grad is None for m in frozen for v in m.parameters())
        if not stopped:assert counts==p['counts_max'],counts
        report={'status':'stopped_declared_protection' if stopped else 'completed64','steps':step,'counts':counts,
            'trainable_parameters':nparams,'checkpoint':{'path':str(checkpoint),'sha256':file_sha256(checkpoint)},
            'frozen_models_unchanged':True,'training_loop_with_evaluation_seconds':seconds,'training_peak_allocated_bytes':peak,
            'accepted_base':False,'new_ai_images':0,'budget':budget.snapshot()}
        report['artifacts']={str(f):file_sha256(f) for f in RUN.rglob('*') if f.is_file()};save(RUN/'training_report.json',report)
        print({'status':report['status'],'steps':step,'counts':counts,'seconds':seconds},flush=True)
      except BaseException as exc:
        save(RUN/'failure.json',{'error':repr(exc),'step':step,'counts':counts,'last_checkpoint':str(checkpoint),'automatic_restart':False});raise

if __name__=='__main__':main()
