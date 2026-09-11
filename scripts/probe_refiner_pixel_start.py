"""Bounded cold versus latent-fitted warm R starts under identical original pixel/detail loss.

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
from scripts.plot_refiner_pixel_losses import plot
from scripts.detail_supervision_math import detail_loss
from h3ce.train.precision import POLICY
from scripts.verify_loss_balance_initialization import assert_equal, state_hash

KIND="refiner_pixel_start_probe"
SOURCE=ROOT/"logs/refiner-latent-distillation-20260909-v1"
ARMS=("cold_pixel","warm_pixel")
TRAIN_INDICES=(0,1,5,6,14,15)
STEPS=192
EVALUATIONS=(0,48,192)
LR=.00001
GATE={"train_degraded_min_cases":2,"high_relative_input_max":-.05,
      "edge_relative_input_max":-.05,"global_relative_input_max":0.,
      "clean_mean_global_mae_max":.001,"single_train_degraded_global_regression_max":.05,
      "visual_required":True,"independent_validation":False}


def codes():
    result=code_hashes()
    for name in ("probe_refiner_latent_distillation.py","summarize_refiner_latent_distillation.py",
                 "probe_refiner_pixel_start.py","plot_refiner_pixel_losses.py","detail_supervision_math.py",
                 "diagnose_supervision_gradients.py","verify_loss_balance_initialization.py"):
        result["scripts/"+name]=file_sha256(ROOT/"scripts"/name)
    return result


PIXEL_LOSSES={"rgb":1.,"lighting_target":.2,"detail":.5,"latent":0.,"perceptual":0.}


def latent_mse(prediction,target,valid):
    from scripts.probe_refiner_latent_distillation import latent_mse as measure_latent
    return measure_latent(prediction,target,valid)


def target_for(sample,index,arm,targets):
    require(arm in ARMS and index in TRAIN_INDICES,"Wrong R pixel view")
    require(sample["supervision_group"] in ("original_degraded","original_clean") and sample["clean_pair"]==(sample["supervision_group"]=="original_clean"),"R pixel target must be original-only")
    return sample["z_target"]


def pixel_terms(model,bridge,sample,loss):
    require(sample["supervision_group"] in ("original_degraded","original_clean") and sample["clean_pair"]==(sample["supervision_group"]=="original_clean"),"Original pixel supervision only")
    zp,_=refine_latent(model,sample,autocast_enabled=torch.cuda.is_bf16_supported())
    pred,_=restore_pixels(bridge,sample,zp,grad=True,strength=1.)
    terms=dict(loss(pred,sample["y"],zp,sample["z_target"],sample["valid"],sample["person_mask"],sample["face_mask"]))
    terms["application_total"]=terms["total"]
    terms["detail"]=detail_loss(pred,sample["y"],sample["valid"],sample["person_mask"],sample["face_mask"])
    terms["total"]=terms["application_total"]+.5*terms["detail"]
    return terms


def check(p):
    from scripts.probe_refiner_latent_distillation import check as prior_check
    prior_check(read(SOURCE/"protocol.json"))
    require(p["kind"]==KIND and p["steps"]==STEPS and p["maximum_optimizer_updates"]==384
            and p["evaluation_steps"]==list(EVALUATIONS) and p["train_indices"]==list(TRAIN_INDICES)
            and p["learning_rate"]==LR and p["accumulation"]==1 and p["gate"]==GATE and p["pixel_losses"]==PIXEL_LOSSES
            and p["code_sha256"]==codes() and p["maximum_fixed_predictions"]==96 and p["maximum_native_decodes"]==1048 and p["maximum_training_attempts"]==416
            and p["maximum_training_vjps"]==12 and p["full_frame_tolerance"]=={"rtol":2e-4,"atol":2e-7},"R pixel scope changed")
    require(set(p["arms"])==set(ARMS) and not p["deployable"] and not p["trained_base_accepted"] and not p["automatic_extension"],"Wrong pixel arms or acceptance")
    for item in p["evidence"]:require(file_sha256(item["path"])==item["sha256"],"R pixel source evidence changed")
    folder=Path(p["config"]).resolve().parent
    require(folder.is_relative_to(ROOT/"logs") and all(Path(p["arms"][arm]).resolve()==ROOT/"runs"/folder.name/arm for arm in ARMS),"R pixel paths changed")
    require(load_config(p["config"]).model_dump(mode="json")==load_config(read(SOURCE/"protocol.json")["config"]).model_dump(mode="json"),"R pixel configuration changed")


def declare(output):
    assert_no_links(output);require(output.is_relative_to(ROOT/"logs") and not output.exists(),"Use a fresh pixel experiment folder")
    source=read(SOURCE/"protocol.json");summary=read(SOURCE/"summary/summary.json")
    require(summary["status"]=="verified_completed_refiner_latent_distillation" and summary["source_optimizer_updates"]==768,"Completed independently audited latent R fit required")
    prior_run=Path(source["arms"]["optimized_latent"]);prior_report=read(prior_run/"training_report.json")
    prior_manager=CheckpointManager(ROOT/"runs",prior_run,contract=read(prior_run/"training_contract.json"))
    prior=prior_manager.read(Path(prior_report["checkpoint"]["path"]));require(prior["step"]==384 and prior["extra"]["phase_complete"],"Warm R source incomplete")
    config=load_config(source["config"])
    with no_training_guard(),TrainingBudget(ROOT/"runs",config.project.budget_seconds,phase=KIND+"_declare") as budget:
        budget.validate_resume_snapshot(prior["budget"]);dataset=open_dataset(config,ROOT,Path(source["manifest"]));audit=dataset.audit(budget_check=budget.check)
        output.mkdir();write_resolved(config,output/"resolved.yaml")
        evidence=[bound(path) for path in (SOURCE/"protocol.json",SOURCE/"summary/summary.json",SOURCE/"final_review.json",SOURCE/"integrity_review.json",
                  prior_run/"training_report.json",prior_run/"training_contract.json",Path(prior_report["checkpoint"]["path"]),Path(prior_report["checkpoint"]["path"]).with_suffix(".json"),
                  Path(source["manifest"]),output/"resolved.yaml",ROOT/config.paths.components_lock)]
        evidence.extend(summary["evidence"])
        p={"kind":KIND,"status":"declared_before_training","created_utc":datetime.now(timezone.utc).isoformat(),
           "authorization":"Four-hour autonomous GPU research; isolate whether an existing latent-fitted R start benefits original pixel/detail optimization",
           "steps":STEPS,"maximum_optimizer_updates":384,"evaluation_steps":list(EVALUATIONS),"maximum_fixed_predictions":96,"maximum_native_decodes":1048,"maximum_training_attempts":416,
           "maximum_training_vjps":12,"train_indices":list(TRAIN_INDICES),"evaluation_indices":source["evaluation_indices"],"learning_rate":LR,"accumulation":1,
           "arms":{a:str(ROOT/"runs"/output.name/a) for a in ARMS},"config":str(output/"resolved.yaml"),"manifest":source["manifest"],
           "warm_run":str(prior_run),"warm_checkpoint":prior_report["checkpoint"],"warm_model_sha256":state_hash(prior["model"]),
           "views":source["views"],"rois":source["rois"],"source_protocol":bound(SOURCE/"protocol.json"),"pixel_losses":PIXEL_LOSSES,
           "changed_factor":"Initial R weights only: fresh seed42 versus completed optimized-latent R step384; five other states identical and new",
           "trainable":"Existing R/scene; native H3 frozen; no new architecture or temporal processing",
           "refiner_autocast":"bfloat16_if_supported","gradient_precision_policy":POLICY,"gate":GATE,"data_audit":audit,"evidence":evidence,"code_sha256":codes(),
           "full_frame_tolerance":{"rtol":2e-4,"atol":2e-7},"automatic_extension":False,"deployable":False,"trained_base_accepted":False,"independent_validation":False,
           "overfit_gate_satisfied":False,"new_ai_images":0,"scope":"Six-view pixel-start diagnostic, not useful-base or source-independent acceptance"}
        atomic_write(output/"protocol.json",canonical_json(p));check(p)
    print(canonical_json({"event":"refiner_pixel_start_declared","output":str(output)}).decode(),flush=True)


def load_targets(p):return {}


def append(path,row):
    with path.open("ab") as stream: stream.write(canonical_json(row)+b"\n");stream.flush()


def gradient_preflight(model,bridge,loss,samples,targets,arm,run,p):
    rng=capture_rng_state(); before=state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()})
    rows=[]
    h3_before={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
    try:
        with gradient_only_guard() as guard:
            for index in TRAIN_INDICES:
                sample=samples[index]
                target=target_for(sample,index,arm,targets)
                terms=pixel_terms(model,bridge,sample,loss);term=terms["total"]
                grads=torch.autograd.grad(term*POLICY["init_scale"],tuple(model.parameters()),allow_unused=False)
                grads=tuple(g.detach().float()/POLICY["init_scale"] for g in grads)
                require(all(torch.isfinite(g).all() for g in grads),"Nonfinite R pixel VJP")
                norms={n:float(g.double().norm()) for (n,_),g in zip(model.named_parameters(),grads)}
                total=sum(v*v for v in norms.values())**.5
                require(total>0 or sample["clean_pair"],"Degraded original pixel target has no parameter gradient")
                rows.append({"index":index,"target_sha256":state_hash(target.detach().cpu()),"loss":float(term.detach()),"gradient_norm":total,"parameter_gradient_norms":norms})
            require(dict(guard)=={"optimizer_constructions":0,"backward_calls":0,"autograd_grad_calls":6},"Unexpected R preflight operations")
            require(before==state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()}) and not any(v.grad is not None for v in model.parameters()),"R changed in preflight")
            require(h3_before=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()} and not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 changed in pixel preflight")
            result={"h3_frozen_unchanged":True,"status":"passed_six_view_pixel_vjp_preflight","arm":arm,"protocol_sha256":file_sha256(Path(p["config"]).parent/"protocol.json"),"guard":dict(guard),"cases":rows,"optimizer_updates":0,"initial_model_sha256":before,"scale":POLICY["init_scale"]}
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
                if step==0 and arm=="cold_pixel":require(torch.equal(pred,sample["x"]),"Initial cold pixel output differs from X")
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
                             "float_arrays":bound(path),"detail_objective":{"application_total":current["metrics"]["total"],"detail":float(detail_loss(pred,sample["y"],sample["valid"],sample["person_mask"],sample["face_mask"]))},})
            require(versions==[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in (model,bridge.backend.model)],"Fixed evaluation changed parameters")
            result={"status":"completed_refiner_pixel_start_fixed","step":step,"arm":arm,"checkpoint":bound(checkpoint),"cases":rows,
                    "guard":dict(guard),"native_decodes":2*len(rows),"optimizer_updates":0,"budget":budget.snapshot(),"independent_validation":False}
            atomic_write(folder/"metrics.json",canonical_json(result))
    finally:model.train(mode);restore_rng_state(rng)
    print(canonical_json({"event":"refiner_pixel_fixed","arm":arm,"step":step,"cases":len(rows)}).decode(),flush=True)
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
            if arm=="warm_pixel":
                prior_run=Path(p["warm_run"]);prior=CheckpointManager(ROOT/"runs",prior_run,contract=read(prior_run/"training_contract.json")).read(Path(p["warm_checkpoint"]["path"]))
                require(state_hash(prior["model"])==p["warm_model_sha256"],"Warm source weights changed")
                model.load_state_dict(prior["model"],strict=True)
            loss=application_loss(config,ROOT)
            samples={i:move_sample(dataset[i],"cuda") for i in TRAIN_INDICES}
            if not resume:preflight=gradient_preflight(model,bridge,loss,samples,targets,arm,run,p)
            else:
                preflight=read(run/"preflight.json")
                require(preflight["status"]=="passed_six_view_pixel_vjp_preflight" and preflight["protocol_sha256"]==file_sha256(output/"protocol.json") and preflight["initial_model_sha256"]==state_hash({n:v.detach().cpu() for n,v in model.state_dict().items()}),"Initial VJP binding changed")
            optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
            scaler=make_scaler("pixel");sampler=StatefulSampler(len(TRAIN_INDICES),seed=42)
            loss=application_loss(config,ROOT);contract=make_contract(config,ROOT,dataset,"pixel",STEPS)
            contract["experiment"]={"kind":KIND,"arm":arm,"protocol_sha256":file_sha256(output/"protocol.json"),"code_sha256":p["code_sha256"],"train_indices":list(TRAIN_INDICES),"pixel_losses":PIXEL_LOSSES,"deployable":False}
            if resume:require(read(run/"training_contract.json")==contract,"Resume contract changed")
            else:atomic_write(run/"training_contract.json",canonical_json(contract))
            manager=CheckpointManager(ROOT/"runs",run,contract=contract)
            extra={"kind":KIND,"slot_visits":[0]*6,"overflow_attempts":0,"consecutive_overflows":0,"phase_complete":False,"deployable":False}
            def save():
                optimizer.zero_grad(set_to_none=True)
                return manager.save(model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler,stage="pixel",step=step,budget=budget.snapshot(),resolved_config=config.model_dump(mode="json"),extra=extra)
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
                require(state["stage"]=="pixel" and 0<=step<=STEPS and not extra["phase_complete"],"Invalid recovery phase")
                path=run/"metrics.jsonl";raw=path.read_bytes() if path.exists() else b"";rows=[json.loads(line) for line in raw.splitlines()]
                require([r["step"] for r in rows[:step]]==list(range(1,step+1)),"Recovery lacks committed records")
                if len(rows)>step:
                    atomic_write(run/f"metrics_before_resume_{time.time_ns()}.jsonl",raw);atomic_write(path,b"".join(canonical_json(r)+b"\n" for r in rows[:step]))
            if arm==ARMS[1] and not resume:
                prior_run=Path(p["arms"][ARMS[0]]);binding=read(prior_run/"initial_checkpoint.json")
                require(file_sha256(binding["path"])==binding["sha256"],"First arm initial checkpoint changed")
                prior=CheckpointManager(ROOT/"runs",prior_run,contract=read(prior_run/"training_contract.json")).read(Path(binding["path"]))
                current=manager.read(checkpoint)
                for key in ("optimizer","scheduler","scaler","sampler","rng"):assert_equal(prior[key],current[key],"Both R arms initialization: "+key)
                atomic_write(run/"same_initialization.json",canonical_json({"all_five_nonmodel_equal":True,"first":binding,"second":bound(checkpoint)}))
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
                target=target_for(sample,index,arm,targets);terms=pixel_terms(model,bridge,sample,loss)
                return terms,{"index":index,"view_id":sample["view_id"],"asset_id":sample["asset_id"],"supervision_group":sample["supervision_group"],"clean_pair":sample["clean_pair"],"target_sha256":state_hash(target.detach().cpu())}
            while step<STEPS:
                record=update_window(model,optimizer,scheduler,scaler,sampler,get_terms,1,1.,transaction)
                if not record["optimizer_updated"]:
                    extra["overflow_attempts"]+=1;extra["consecutive_overflows"]+=1;checkpoint=save()
                    append(run/"skipped_updates.jsonl",{"completed_step":step,**record})
                    require(extra["consecutive_overflows"]<POLICY["max_consecutive_overflows"] and extra["overflow_attempts"]<16,"Repeated native FP16 overflow")
                    continue
                extra["consecutive_overflows"]=0
                step+=1
                for sample in record["samples"]:extra["slot_visits"][sample["slot"]]+=1
                record.update(step=step,phase="pixel",diagnostic_kind=KIND,arm=arm,learning_rate=LR,used_seconds=budget.used)
                append(run/"metrics.jsonl",record)
                if step%48==0:checkpoint=save()
                if step==1 or step%48==0:
                    print(canonical_json({"event":"refiner_pixel_update","arm":arm,"step":step,"total":record["losses"]["total"],"application_total":record["losses"]["application_total"],"detail":record["losses"]["detail"]}).decode(),flush=True)
                    rng=capture_rng_state()
                    try:plot([run],run/"training_losses.png",window=24)
                    finally:restore_rng_state(rng)
                if step in EVALUATIONS:evaluate_current()
            require(extra["slot_visits"]==[STEPS//6]*6,"Unbalanced fixed six-view exposure")
            require(frozen=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
                    and not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 changed")
            check(p);extra["phase_complete"]=True;checkpoint=save()
            report={"status":"completed_refiner_pixel_start_probe","kind":KIND,"arm":arm,"optimizer_steps":step,"checkpoint":bound(checkpoint),"extra":extra,
                    "source_preflight_vjps":6,"source_fixed_predictions":48,"native_decodes":492+2*extra["overflow_attempts"],"h3_weight_updates":0,"trained_base_accepted":False,"deployable":False,
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
