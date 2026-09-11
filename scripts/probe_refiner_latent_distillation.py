"""Bounded R fitting: original encoded targets versus C4 optimized targets.

Six training views, unchanged input-only SpatialRefinerV2, independent fresh arms.
This diagnostic does not satisfy the 8-16 pair overfit or useful-base gate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None, ""): sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from h3ce.cache.keys import canonical_json, file_sha256, digest
from h3ce.cache.store import atomic_write, assert_no_links
from h3ce.config import load_config, write_resolved
from h3ce.model import SpatialRefinerV2
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget, capture_rng_state, restore_rng_state
from h3ce.train.engine import make_contract
from h3ce.train.guard import no_training_guard
from h3ce.train.losses import region_error
from h3ce.train.perceptual import application_loss
from h3ce.train.pipeline import move_sample, refine_latent, restore_pixels
from h3ce.train.precision import make_scaler
from h3ce.train.preflight import open_dataset, load_bridge, require, seed_model
from h3ce.train.sampler import StatefulSampler
from scripts.gpt_clarity_probe_common import bound, read, code_hashes, update_window
from scripts.diagnose_supervision_gradients import gradient_only_guard
from scripts.evaluate_gpt_clarity_probe import fixed_roi, edge_mse
from scripts.evaluate_bootstrap_checkpoint import measure
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.plot_training_losses import plot
from scripts.verify_loss_balance_initialization import assert_equal, state_hash

KIND="refiner_latent_distillation_probe"
SOURCE=ROOT/"logs/target-start-latent-20260909-v1"
ARMS=("original_latent","optimized_latent")
TRAIN_INDICES=(0,1,5,6,14,15)
STEPS=384
EVALUATIONS=(0,48,192,384)
LR=.0002
GATE={"train_degraded_min_cases":2,"high_relative_input_max":-.05,
      "edge_relative_input_max":-.05,"global_relative_input_max":0.,
      "clean_mean_global_mae_max":.001,"single_train_degraded_global_regression_max":.05,
      "visual_required":True,"independent_validation":False}


def codes():
    result=code_hashes()
    for name in ("probe_target_start_latent.py","plot_target_start_latent_losses.py",
                 "probe_refiner_latent_distillation.py","summarize_target_start_latent.py",
                 "diagnose_supervision_gradients.py","verify_loss_balance_initialization.py"):
        result["scripts/"+name]=file_sha256(ROOT/"scripts"/name)
    return result


def latent_mse(prediction,target,valid):
    require(prediction.shape==target.shape and prediction.ndim==5 and prediction.shape[1]==24,
            "Expected matching native 24-channel normalized latents")
    require(prediction.shape[2]==valid.shape[2] and valid.ndim==5 and valid.shape[1]==1
            and torch.isfinite(valid).all() and (valid>=0).all() and (valid<=1).all() and valid.sum()>0,
            "Invalid image latent mask")
    require(torch.isfinite(prediction).all() and torch.isfinite(target).all(),"Nonfinite latent target")
    mask=F.interpolate(valid.float(),size=prediction.shape[2:],mode="area")
    return region_error((prediction.float()-target.float()).square(),mask)


def target_for(sample,index,arm,targets):
    require(arm in ARMS and index in TRAIN_INDICES,"Unknown arm or training view")
    require(sample["supervision_group"] in ("original_degraded","original_clean"),"AI target forbidden")
    if sample["clean_pair"]:
        require(sample["supervision_group"]=="original_clean" and torch.equal(sample["z_input"],sample["z_target"]),"Clean pair is not identity")
        return sample["z_target"]
    require(sample["supervision_group"]=="original_degraded","Wrong degraded target")
    return targets[index].to(sample["z_input"].device) if arm=="optimized_latent" else sample["z_target"]


def check(p):
    from scripts.probe_target_start_latent import check as source_check
    source_check(read(SOURCE/"protocol.json"))
    require(p["kind"]==KIND and p["steps"]==STEPS and p["maximum_optimizer_updates"]==2*STEPS
            and p["evaluation_steps"]==list(EVALUATIONS) and p["train_indices"]==list(TRAIN_INDICES)
            and p["learning_rate"]==LR and p["accumulation"]==1 and p["gate"]==GATE
            and p["code_sha256"]==codes() and p["maximum_fixed_predictions"]==128 and p["maximum_native_decodes"]==256
            and p["maximum_training_vjps"]==12 and p["full_frame_tolerance"]=={"rtol":2e-4,"atol":2e-7},"R distillation scope or code changed")
    require(set(p["arms"])==set(ARMS) and p["objective"]=="area-valid masked normalized latent MSE; coefficient 1"
            and p["trainable"]=="Existing SpatialRefinerV2 including scene; native H3 entirely frozen"
            and p["initialization"]=="Fresh seed42, identical model/optimizer/scheduler/scaler/sampler/RNG in both arms"
            and p["refiner_autocast"]=="bfloat16_if_supported" and not p["deployable"] and not p["trained_base_accepted"],"R fitting contract changed")
    for item in p["evidence"]: require(file_sha256(item["path"])==item["sha256"],"Bound R fitting evidence changed")
    folder=Path(p["config"]).resolve().parent
    require(folder.is_relative_to(ROOT/"logs") and all(Path(p["arms"][arm]).resolve()==ROOT/"runs"/folder.name/arm for arm in ARMS),"Run path differs")
    require(load_config(p["config"]).model_dump(mode="json")==load_config(read(SOURCE/"protocol.json")["config"]).model_dump(mode="json"),"Base config changed")


def declare(output):
    assert_no_links(output)
    require(output.is_relative_to(ROOT/"logs") and not output.exists(),"Use a new experiment folder")
    source=read(SOURCE/"protocol.json"); summary=read(SOURCE/"summary/summary.json"); review=read(SOURCE/"final_review.json")
    require(summary["status"]=="verified_completed_target_start_sample_latent_probe"
            and summary["source_optimizer_updates"]==96 and summary["gate"]["retention_gate_met"]
            and review["visual_detail_retention_supported"] is True,"Completed numerically and visually reviewed C4 required")
    config=load_config(source["config"])
    with no_training_guard(),TrainingBudget(ROOT/"runs",config.project.budget_seconds,phase=KIND+"_declare") as budget:
        dataset=open_dataset(config,ROOT,Path(source["manifest"]));audit=dataset.audit(budget_check=budget.check)
        evaluation_indices=[i for i,v in enumerate(dataset.views) if v["supervision_group"]!="ai_paired"]
        require(len(evaluation_indices)==16 and len(TRAIN_INDICES)==6,"Expected 16 original evaluation views and six fitting views")
        targets={};bindings=[]
        for case in source["cases"]:
            run=Path(case["run"]);report=read(run/"training_report.json")
            manager=CheckpointManager(ROOT/"runs",run,contract=read(run/"training_contract.json"))
            state=manager.read(Path(report["checkpoint"]));budget.validate_resume_snapshot(state["budget"])
            require(state["step"]==32 and state["extra"]["phase_complete"],"Incomplete C4 teacher")
            sample=dataset[case["index"]]
            targets[case["index"]]=sample["z_input"]+state["model"]["delta_latent"]
            bindings.extend((bound(Path(report["checkpoint"])),bound(Path(report["checkpoint"]).with_suffix(".json"))))
        output.mkdir();write_resolved(config,output/"resolved.yaml")
        target_path=output/"targets.npz";np.savez_compressed(target_path,**{str(k):v.numpy() for k,v in targets.items()})
        evidence=[bound(path) for path in (SOURCE/"protocol.json",SOURCE/"summary/summary.json",SOURCE/"final_review.json",
                  Path(source["manifest"]),output/"resolved.yaml",target_path,ROOT/config.paths.components_lock)]
        evidence.extend(bindings)
        evidence.extend(item for c in summary["cases"] for item in c["evidence"])
        p={"kind":KIND,"status":"declared_before_training","created_utc":datetime.now(timezone.utc).isoformat(),
           "authorization":"Four-hour autonomous GPU research; C4 motivates a bounded input-only R fit of optimized latents",
           "steps":STEPS,"maximum_optimizer_updates":2*STEPS,"evaluation_steps":list(EVALUATIONS),
           "maximum_fixed_predictions":128,"maximum_native_decodes":256,"maximum_training_vjps":12,
           "train_indices":list(TRAIN_INDICES),"evaluation_indices":evaluation_indices,"learning_rate":LR,"accumulation":1,
           "arms":{a:str(ROOT/"runs"/output.name/a) for a in ARMS},"config":str(output/"resolved.yaml"),
           "manifest":source["manifest"],"targets":bound(target_path),"target_hashes":{str(k):state_hash(v) for k,v in targets.items()},
           "views":[{"index":i,"view_id":v["view_id"],"asset_id":v["asset_id"],"group":v["supervision_group"]} for i,v in enumerate(dataset.views)],
           "rois":{str(i):fixed_roi(dataset[i]) for i in evaluation_indices},"source_protocol":bound(SOURCE/"protocol.json"),
           "objective":"area-valid masked normalized latent MSE; coefficient 1",
           "changed_factor":"Three degraded training latent targets: original encoding versus C4 fixed step32; clean targets identical",
           "trainable":"Existing SpatialRefinerV2 including scene; native H3 entirely frozen",
           "initialization":"Fresh seed42, identical model/optimizer/scheduler/scaler/sampler/RNG in both arms",
           "refiner_autocast":"bfloat16_if_supported","gate":GATE,"data_audit":audit,"evidence":evidence,"code_sha256":codes(),
           "full_frame_tolerance":{"rtol":2e-4,"atol":2e-7},"automatic_extension":False,"deployable":False,"trained_base_accepted":False,
           "independent_validation":False,"overfit_gate_satisfied":False,"new_ai_images":0,
           "scope":"Six-view capacity/target diagnostic, not a long bootstrap or source-independent validation; other ten views are same-source unseen during this fit"}
        atomic_write(output/"protocol.json",canonical_json(p));check(p)
    print(canonical_json({"event":"refiner_distillation_declared","output":str(output)}).decode(),flush=True)


def load_targets(p):
    require(file_sha256(p["targets"]["path"])==p["targets"]["sha256"],"Teacher file changed")
    with np.load(p["targets"]["path"],allow_pickle=False) as archive: targets={int(k):torch.from_numpy(archive[k].copy()) for k in archive.files}
    require(set(targets)=={0,5,14} and all(state_hash(v)==p["target_hashes"][str(k)] for k,v in targets.items()),"Optimized latent hashes differ")
    return targets


def append(path,row):
    with path.open("ab") as stream: stream.write(canonical_json(row)+b"\n");stream.flush()


def gradient_preflight(model,samples,targets,arm,run,p):
    rng=capture_rng_state(); before=state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()})
    rows=[]
    try:
        with gradient_only_guard() as guard:
            for index in TRAIN_INDICES:
                sample=samples[index]
                zp,delta=refine_latent(model,sample,autocast_enabled=torch.cuda.is_bf16_supported())
                require(torch.count_nonzero(delta)==0 and torch.equal(zp,sample["z_input"]),"Fresh R is not zero")
                target=target_for(sample,index,arm,targets);term=latent_mse(zp,target,sample["valid"])
                grads=torch.autograd.grad(term,tuple(model.parameters()),allow_unused=False)
                require(all(torch.isfinite(g).all() for g in grads),"Nonfinite R latent VJP")
                norms={n:float(g.double().norm()) for (n,_),g in zip(model.named_parameters(),grads)}
                total=sum(v*v for v in norms.values())**.5
                require(total>0 or sample["clean_pair"],"Degraded latent target has no parameter gradient")
                rows.append({"index":index,"target_sha256":state_hash(target.detach().cpu()),"loss":float(term.detach()),"gradient_norm":total,"parameter_gradient_norms":norms})
            require(dict(guard)=={"optimizer_constructions":0,"backward_calls":0,"autograd_grad_calls":6},"Unexpected R preflight operations")
            require(before==state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()}) and not any(v.grad is not None for v in model.parameters()),"R changed in preflight")
            result={"status":"passed_six_view_latent_vjp_preflight","arm":arm,"protocol_sha256":file_sha256(Path(p["config"]).parent/"protocol.json"),"guard":dict(guard),"cases":rows,"optimizer_updates":0,"initial_model_sha256":before}
            atomic_write(run/"preflight.json",canonical_json(result))
    finally: restore_rng_state(rng)
    return result


def fixed(model,bridge,dataset,loss,config,p,budget,run,step,checkpoint,targets,arm):
    folder=run/f"fixed_step_{step:04d}";require(not folder.exists(),"Fixed evaluation cannot repeat");folder.mkdir()
    rng=capture_rng_state();mode=model.training;rows=[]
    versions=[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in (model,bridge.backend.model)]
    try:
        with no_training_guard() as guard:
            model.eval()
            for index in p["evaluation_indices"]:
                budget.check();sample=move_sample(dataset[index],"cuda")
                zp,delta=refine_latent(model,sample,autocast_enabled=torch.cuda.is_bf16_supported())
                pred,_=restore_pixels(bridge,sample,zp,grad=False,strength=1.)
                if step==0:require(torch.equal(pred,sample["x"]),"Initial pixel output differs from X")
                x0,y0,x1,y1=p["rois"][str(index)]
                def roi(t):return t[0,:,0,y0:y1,x0:x1].detach().float().cpu().permute(1,2,0).numpy().copy()
                arrays={"input":roi(sample["x"]),"target":roi(sample["y"]),"prediction":roi(pred),
                        "full_prediction":pred.detach().float().cpu().numpy(),"z_prediction":zp.detach().float().cpu().numpy()}
                path=folder/f"case_{index:02d}.npz";np.savez_compressed(path,**arrays)
                current=measure(sample,pred,zp,loss);baseline=measure(sample,sample["x"],sample["z_input"],loss)
                frequency=frequency_metrics(arrays["input"],arrays["target"],arrays["prediction"])
                target=target_for(sample,index,arm,targets) if index in TRAIN_INDICES else sample["z_target"]
                rows.append({"index":index,"view_id":sample["view_id"],"asset_id":sample["asset_id"],"group":sample["supervision_group"],
                             "trained_view":index in TRAIN_INDICES,"current":current,"input":baseline,"frequency":frequency,
                             "edge_mse":{k:edge_mse(arrays[k],arrays["target"]) for k in ("input","prediction")},
                             "latent_mse_to_training_target":float(latent_mse(zp,target,sample["valid"])),
                             "input_latent_mse_to_training_target":float(latent_mse(sample["z_input"],target,sample["valid"])),
                             "target_role":"arm_training_target" if index in TRAIN_INDICES else "untrained_original_reference",
                             "float_arrays":bound(path)})
            require(versions==[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in (model,bridge.backend.model)],"Fixed evaluation changed parameters")
            result={"status":"completed_refiner_distillation_fixed","step":step,"arm":arm,"checkpoint":bound(checkpoint),"cases":rows,
                    "guard":dict(guard),"native_decodes":2*len(rows),"optimizer_updates":0,"budget":budget.snapshot(),"independent_validation":False}
            atomic_write(folder/"metrics.json",canonical_json(result))
    finally:model.train(mode);restore_rng_state(rng)
    print(canonical_json({"event":"refiner_distillation_fixed","arm":arm,"step":step,"cases":len(rows)}).decode(),flush=True)
    return result


def train(output,arm,resume=False):
    p=read(output/"protocol.json");check(p);require(arm in ARMS,"Unknown arm")
    run=Path(p["arms"][arm]);assert_no_links(run)
    require(not (run/"training_report.json").exists() and run.exists()==resume,"New arm requires a new directory; interrupted arm requires explicit resume")
    if resume:
        interruption=read(run/"training_interruption.json")
        require(interruption["optimizer_may_have_partially_updated"] is False,"Cannot automatically recover an uncertain optimizer transaction")
    config=load_config(p["config"])
    if not resume:run.mkdir(parents=True);write_resolved(config,run/"resolved.yaml")
    step=0;manager=None;checkpoint=None;transaction={"in_optimizer":False};started=time.monotonic()
    with TrainingBudget(ROOT/"runs",config.project.budget_seconds,phase=KIND+"_"+arm) as budget:
        try:
            dataset=open_dataset(config,ROOT,Path(p["manifest"]));dataset.audit(budget_check=budget.check)
            targets=load_targets(p);seed_model(42);bridge=load_bridge(config,ROOT,dataset)
            model=SpatialRefinerV2.from_config(config.model).to("cuda").train()
            samples={i:move_sample(dataset[i],"cuda") for i in TRAIN_INDICES}
            if not resume:preflight=gradient_preflight(model,samples,targets,arm,run,p)
            else:
                preflight=read(run/"preflight.json")
                require(preflight["status"]=="passed_six_view_latent_vjp_preflight" and preflight["protocol_sha256"]==file_sha256(output/"protocol.json") and preflight["initial_model_sha256"]==state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()}),"Initial VJP binding changed")
            optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
            scaler=make_scaler("latent");sampler=StatefulSampler(len(TRAIN_INDICES),seed=42)
            loss=application_loss(config,ROOT);contract=make_contract(config,ROOT,dataset,"latent",STEPS)
            contract["experiment"]={"kind":KIND,"arm":arm,"protocol_sha256":file_sha256(output/"protocol.json"),"code_sha256":p["code_sha256"],"train_indices":list(TRAIN_INDICES),"target_hashes":p["target_hashes"],"deployable":False}
            if resume:require(read(run/"training_contract.json")==contract,"Resume contract changed")
            else:atomic_write(run/"training_contract.json",canonical_json(contract))
            manager=CheckpointManager(ROOT/"runs",run,contract=contract)
            extra={"kind":KIND,"slot_visits":[0]*6,"overflow_attempts":0,"consecutive_overflows":0,"phase_complete":False,"deployable":False}
            def save():
                optimizer.zero_grad(set_to_none=True)
                return manager.save(model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler,stage="latent",step=step,budget=budget.snapshot(),resolved_config=config.model_dump(mode="json"),extra=extra)
            if not resume:
                checkpoint=save();atomic_write(run/"initial_checkpoint.json",canonical_json(bound(checkpoint)))
            else:
                candidates=[]
                for receipt_file in manager.directory.glob("checkpoint-*.json"):
                    path,receipt=manager._receipt(receipt_file.with_suffix(".pt"))
                    require(receipt["contract_id"]==manager.contract_id,"Wrong resume receipt")
                    candidates.append((receipt["created_ns"],path))
                require(candidates,"No committed recovery state")
                checkpoint=max(candidates,key=lambda item:(item[0],str(item[1])))[1]
                state=manager.restore(checkpoint,model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler)
                budget.validate_resume_snapshot(state["budget"]);step=state["step"];extra=state["extra"]
                require(state["stage"]=="latent" and 0<=step<=STEPS and not extra["phase_complete"],"Invalid recovery phase")
                path=run/"metrics.jsonl";raw=path.read_bytes() if path.exists() else b"";rows=[json.loads(line) for line in raw.splitlines()]
                require([r["step"] for r in rows[:step]]==list(range(1,step+1)),"Recovery lacks committed records")
                if len(rows)>step:
                    atomic_write(run/f"metrics_before_resume_{time.time_ns()}.jsonl",raw);atomic_write(path,b"".join(canonical_json(r)+b"\n" for r in rows[:step]))
            if arm==ARMS[1] and not resume:
                prior_run=Path(p["arms"][ARMS[0]]);binding=read(prior_run/"initial_checkpoint.json")
                require(file_sha256(binding["path"])==binding["sha256"],"First arm initial checkpoint changed")
                prior=CheckpointManager(ROOT/"runs",prior_run,contract=read(prior_run/"training_contract.json")).read(Path(binding["path"]))
                current=manager.read(checkpoint)
                for key in ("model","optimizer","scheduler","scaler","sampler","rng"):assert_equal(prior[key],current[key],"Both R arms initialization: "+key)
                atomic_write(run/"same_initialization.json",canonical_json({"all_six_equal":True,"first":binding,"second":bound(checkpoint)}))
            frozen={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()};torch.cuda.reset_peak_memory_stats()
            def evaluate_current():
                path=run/f"fixed_step_{step:04d}"/"metrics.json"
                if path.exists():
                    previous=read(path);require(previous["step"]==step and previous["arm"]==arm,"Fixed recovery identity differs")
                    require(file_sha256(previous["checkpoint"]["path"])==previous["checkpoint"]["sha256"],"Fixed checkpoint changed")
                    saved=manager.read(Path(previous["checkpoint"]["path"]))
                    assert_equal(saved["model"],{n:v.detach().cpu() for n,v in model.state_dict().items()},"Recovery fixed R model")
                    for case in previous["cases"]:require(file_sha256(case["float_arrays"]["path"])==case["float_arrays"]["sha256"],"Fixed floats changed")
                else:fixed(model,bridge,dataset,loss,config,p,budget,run,step,checkpoint,targets,arm)
            if step in EVALUATIONS:evaluate_current()
            def get_terms(slot):
                budget.check();index=TRAIN_INDICES[slot];sample=samples[index]
                zp,_=refine_latent(model,sample,autocast_enabled=torch.cuda.is_bf16_supported())
                target=target_for(sample,index,arm,targets);value=latent_mse(zp,target,sample["valid"])
                return {"total":value,"latent":value},{"index":index,"view_id":sample["view_id"],"asset_id":sample["asset_id"],"supervision_group":sample["supervision_group"],"clean_pair":sample["clean_pair"],"target_sha256":state_hash(target.detach().cpu())}
            while step<STEPS:
                record=update_window(model,optimizer,scheduler,scaler,sampler,get_terms,1,1.,transaction)
                require(record["optimizer_updated"],"Latent BF16 nonfinite update requires explicit investigation")
                step+=1
                for sample in record["samples"]:extra["slot_visits"][sample["slot"]]+=1
                record.update(step=step,phase="latent",diagnostic_kind=KIND,arm=arm,learning_rate=LR,used_seconds=budget.used)
                append(run/"metrics.jsonl",record)
                if step%48==0:checkpoint=save()
                if step==1 or step%48==0:
                    print(canonical_json({"event":"refiner_latent_update","arm":arm,"step":step,"latent_mse":record["losses"]["total"]}).decode(),flush=True)
                    rng=capture_rng_state()
                    try:plot([run],run/"training_losses.png",window=24)
                    finally:restore_rng_state(rng)
                if step in EVALUATIONS:evaluate_current()
            require(extra["slot_visits"]==[STEPS//6]*6,"Unbalanced fixed six-view exposure")
            require(frozen=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
                    and not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 changed")
            check(p);extra["phase_complete"]=True;checkpoint=save()
            report={"status":"completed_refiner_latent_distillation_probe","kind":KIND,"arm":arm,"optimizer_steps":step,"checkpoint":bound(checkpoint),"extra":extra,
                    "source_preflight_vjps":6,"source_fixed_predictions":64,"native_decodes":128,"h3_weight_updates":0,"trained_base_accepted":False,"deployable":False,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.monotonic()-started,"budget":budget.snapshot(),"preflight":bound(run/"preflight.json")}
            atomic_write(run/"training_report.json",canonical_json(report))
        except BaseException as exc:
            if manager is not None and 'save' in locals() and not transaction["in_optimizer"]:checkpoint=save()
            atomic_write(run/"training_interruption.json",canonical_json({"step":step,"checkpoint":str(checkpoint),"optimizer_may_have_partially_updated":transaction["in_optimizer"],"error":type(exc).__name__,"message":str(exc),"budget":budget.snapshot(),"automatic_restart":False}))
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',required=True,choices=('declare','run'));parser.add_argument('--output',required=True,type=Path);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='declare':require(args.arm is None and not args.resume,"Declaration has no arm or resume");declare(output)
    else:
        require(not args.resume or args.arm in ARMS,"Explicit resume requires one arm")
        for arm in ([args.arm] if args.arm else ARMS):train(output,arm,args.resume)
