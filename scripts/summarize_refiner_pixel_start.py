"""Independently audit saved R fitting checkpoints and full-frame predictions on CPU."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from h3ce.cache.keys import canonical_json,file_sha256
from h3ce.cache.store import atomic_write
from h3ce.config import load_config
from h3ce.train.checkpoint import CheckpointManager,TrainingBudget
from h3ce.train.guard import no_training_guard
from h3ce.train.preflight import open_dataset,require
from h3ce.train.perceptual import application_loss
from h3ce.train.sampler import StatefulSampler
from scripts.gpt_clarity_probe_common import read,bound
from scripts.probe_refiner_pixel_start import (KIND,ARMS,STEPS,LR,EVALUATIONS,TRAIN_INDICES,GATE,check,latent_mse,load_targets,target_for)
from scripts.summarize_latent_reachability import binding,relative_change,GUARD_ZERO
from scripts.summarize_supervision_gradients import near,finite_tree
from scripts.evaluate_bootstrap_checkpoint import measure
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.evaluate_gpt_clarity_probe import edge_mse
from scripts.plot_training_losses import read_records
from scripts.plot_refiner_pixel_losses import contributions,plot
from scripts.detail_supervision_math import detail_loss
from scripts.verify_loss_balance_initialization import assert_equal,check_optimizer,state_hash


def decision(rows,preset=GATE):
    degraded=[r for r in rows if r["trained_view"] and r["group"]=="original_degraded"]
    clean=[r for r in rows if r["group"]=="original_clean"]
    require(len(degraded)==3 and len(clean)==8,"All training originals and clean checks required")
    cases=[]
    for row in degraded:
        changes={k:relative_change(row[k],row[k+"_input"]) for k in ("global_mae","high_mse","edge_mse")}
        passed=all(changes[k] is not None and changes[k]<=preset[threshold] for k,threshold in
                   (("global_mae","global_relative_input_max"),("high_mse","high_relative_input_max"),("edge_mse","edge_relative_input_max")))
        cases.append({"index":row["index"],"changes":changes,"numerical_pass":passed})
    clean_mean=float(np.mean([r["global_mae"] for r in clean]));count=sum(r["numerical_pass"] for r in cases)
    regression_guard=all(c["changes"]["global_mae"]<=preset["single_train_degraded_global_regression_max"] for c in cases)
    return {"cases":cases,"passing_train_degraded_cases":count,"all_clean_mean_global_mae":clean_mean,
            "clean_guard_pass":clean_mean<=preset["clean_mean_global_mae_max"],"all_train_degraded_regression_guard":regression_guard,
            "numeric_gate_met":count>=preset["train_degraded_min_cases"] and regression_guard and clean_mean<=preset["clean_mean_global_mae_max"],
            "visual_pass":None,"trained_base_accepted":False}


def replay_sampler(state,steps,rows):
    sampler=StatefulSampler(6,seed=42);visits=[0]*6
    for row in rows[:steps]:
        sampler.begin_window();slot=sampler.next_index()
        require(row["samples"][0]["slot"]==slot and row["samples"][0]["index"]==TRAIN_INDICES[slot],"Logged sample differs from replayed sampler")
        visits[slot]+=1;sampler.commit_window()
    assert_equal(sampler.state_dict(),state,"Replayed six-view sampler")
    return visits


def verify_arm(folder,p,arm,dataset,config,budget,targets):
    run=Path(p["arms"][arm]);report=read(run/"training_report.json");contract=read(run/"training_contract.json")
    require(report["status"]=="completed_refiner_pixel_start_probe" and report["kind"]==KIND
            and report["arm"]==arm and report["optimizer_steps"]==STEPS and report["h3_weight_updates"]==0
            and not report["trained_base_accepted"] and not report["deployable"],"Incomplete R fitting report")
    require(contract["phase"]=="pixel" and contract["max_steps"]==STEPS and contract["experiment"]["protocol_sha256"]==file_sha256(folder/"protocol.json")
            and contract["experiment"]["code_sha256"]==p["code_sha256"] and contract["experiment"]["arm"]==arm,"R fitting checkpoint contract changed")
    require(load_config(run/"resolved.yaml").model_dump(mode="json")==config.model_dump(mode="json"),"Arm config changed")
    evidence=[bound(run/name) for name in ("training_report.json","training_contract.json","initial_checkpoint.json","preflight.json","resolved.yaml","metrics.jsonl")]
    manager=CheckpointManager(ROOT/"runs",run,contract=contract)
    initial_path=binding(read(run/"initial_checkpoint.json"));initial=manager.read(initial_path)
    final_path=binding(report["checkpoint"]);final=manager.read(final_path)
    require(initial["step"]==0 and final["step"]==STEPS and final["extra"]["phase_complete"] and final["extra"]==report["extra"],"R endpoint differs")
    if arm=="cold_pixel":require(all(torch.count_nonzero(initial["model"][k])==0 for k in ("output_projection.weight","output_projection.bias")),"Cold R initial output head is not zero")
    else:require(state_hash(initial["model"])==p["warm_model_sha256"],"Warm R initial weights differ from source")
    rows,_=read_records(run/"metrics.jsonl");require([r["step"] for r in rows]==list(range(1,STEPS+1)),"Missing actual R updates")
    weights,_=contributions(rows,config.model_dump(mode="json"));require(weights=={"rgb":1.,"latent":0.,"perceptual":0.,"lighting_target":.2,"detail":.5},"Pixel detail objective coefficients differ")
    training_samples={i:dataset[i] for i in TRAIN_INDICES}
    for row in rows:
        finite_tree(row,"R update")
        require(row["phase"]=="pixel" and row["diagnostic_kind"]==KIND and row["arm"]==arm and row["optimizer_updated"]
                and row["learning_rate"]==LR and len(row["samples"])==1 and row["loss_scale_before"]>0 and row["loss_scale_after"]>0,"R update precision or source differs")
        sample=row["samples"][0];index=sample["index"];data=training_samples[index]
        require(sample["view_id"]==data["view_id"] and sample["asset_id"]==data["asset_id"] and sample["clean_pair"]==data["clean_pair"]
                and sample["supervision_group"]==data["supervision_group"] and sample["target_sha256"]==state_hash(target_for(data,index,arm,targets)),"Wrong R training data/target")
        near(row["losses"],sample["losses"],"Single sample loss accounting")
    loss=application_loss(config,ROOT);fixed_rows=[];arrays_by_step={}
    def checkpoint(path,step):
        state=manager.read(path);budget.validate_resume_snapshot(state["budget"])
        require(state["stage"]=="pixel" and state["step"]==step and state["scheduler"]["last_epoch"]==step and state["scaler"]["scale"]>0,"Checkpoint scheduler/scaler differs")
        require(state["resolved_config"]==config.model_dump(mode="json"),"Checkpoint config differs")
        visits=replay_sampler(state["sampler"],step,rows);require(visits==state["extra"]["slot_visits"],"Checkpoint exposure differs")
        if step:check_optimizer(initial["optimizer"],state["optimizer"],state["model"],step)
        evidence.extend((bound(path),bound(path.with_suffix(".json"))))
        return state
    checkpoint(initial_path,0);checkpoint(final_path,STEPS)
    require(final["extra"]["slot_visits"]==[32]*6 and final["extra"]["consecutive_overflows"]==0,"Final R exposure or overflow differs")
    skipped_path=run/"skipped_updates.jsonl"
    skipped=[json.loads(line) for line in skipped_path.read_bytes().splitlines()] if skipped_path.exists() else []
    require(len(skipped)==final["extra"]["overflow_attempts"]<16,"Pixel overflow accounting differs")
    for row in skipped:require(not row["optimizer_updated"] and row["loss_scale_after"]<row["loss_scale_before"],"Invalid skipped pixel update")
    if skipped_path.exists():evidence.append(bound(skipped_path))
    preflight=read(run/"preflight.json")
    require(report["preflight"]==bound(run/"preflight.json") and preflight["guard"]=={**GUARD_ZERO,"autograd_grad_calls":6}
            and preflight["optimizer_updates"]==0 and preflight["scale"]==65536. and preflight["h3_frozen_unchanged"] and preflight["initial_model_sha256"]==state_hash(initial["model"]),"R preflight receipt differs")
    for row,index in zip(preflight["cases"],TRAIN_INDICES):
        require(row["index"]==index and row["target_sha256"]==state_hash(target_for(dataset[index],index,arm,targets)),"R preflight target differs")
        near(row["gradient_norm"],sum(v*v for v in row["parameter_gradient_norms"].values())**.5,"Recorded VJP norm accounting")
    for step in EVALUATIONS:
        path=run/f"fixed_step_{step:04d}"/"metrics.json";value=read(path);evidence.append(bound(path))
        require(value["status"]=="completed_refiner_pixel_start_fixed" and value["step"]==step and value["arm"]==arm
                and value["guard"]==GUARD_ZERO and value["optimizer_updates"]==0 and value["native_decodes"]==32,"Fixed execution scope differs")
        state=checkpoint(binding(value["checkpoint"]),step)
        if step==STEPS:assert_equal(state["model"],final["model"],"Final fixed/committed R")
        require([c["index"] for c in value["cases"]]==p["evaluation_indices"],"Fixed evaluation coverage differs")
        arrays_by_step[step]={}
        for saved in value["cases"]:
            budget.check();index=saved["index"];sample=dataset[index]
            require(saved["view_id"]==sample["view_id"] and saved["asset_id"]==sample["asset_id"] and saved["group"]==sample["supervision_group"]
                    and saved["trained_view"]==(index in TRAIN_INDICES),"Fixed view mapping differs")
            file=binding(saved["float_arrays"]);evidence.append(saved["float_arrays"])
            with np.load(file,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
            require(set(arrays)=={"input","target","prediction","full_prediction","z_prediction"}
                    and all(a.dtype==np.float32 and np.isfinite(a).all() for a in arrays.values()),"Invalid full float evidence")
            pred=torch.from_numpy(arrays["full_prediction"]);zp=torch.from_numpy(arrays["z_prediction"])
            require(pred.shape==sample["y"].shape and zp.shape==sample["z_input"].shape,"Wrong full shape")
            if step==0 and arm=="cold_pixel":require(torch.equal(pred,sample["x"]) and torch.equal(zp,sample["z_input"]),"Initial cold R output is not identity")
            if step==0 and arm=="warm_pixel":
                source_path=Path(p["warm_run"])/f"fixed_step_0384/case_{index:02d}.npz"
                with np.load(source_path,allow_pickle=False) as source:
                    require(np.array_equal(source["z_prediction"],arrays["z_prediction"]),"Warm R latent differs from parent checkpoint output")
                    near(float(np.max(np.abs(source["full_prediction"]-arrays["full_prediction"]))),0.,"Warm source full prediction",rtol=0,atol=2e-7)
            x0,y0,x1,y1=p["rois"][str(index)]
            for key,t in (("input",sample["x"]),("target",sample["y"]),("prediction",pred)):
                require(np.array_equal(arrays[key],t[0,:,0,y0:y1,x0:x1].permute(1,2,0).numpy()),"Full/ROI mismatch")
            current=measure(sample,pred,zp,loss);baseline=measure(sample,sample["x"],sample["z_input"],loss)
            near(saved["current"],current,"Recomputed full R metrics",**p["full_frame_tolerance"])
            near(saved["input"],baseline,"Recomputed full baseline",**p["full_frame_tolerance"])
            detail=float(detail_loss(pred,sample["y"],sample["valid"],sample["person_mask"],sample["face_mask"]))
            near(saved["detail_objective"],{"application_total":current["metrics"]["total"],"detail":detail},"Recomputed pixel detail objective",**p["full_frame_tolerance"])
            freq=frequency_metrics(arrays["input"],arrays["target"],arrays["prediction"]);near(saved["frequency"],freq,"Recomputed R frequencies")
            edges={k:edge_mse(arrays[k],arrays["target"]) for k in ("input","prediction")};near(saved["edge_mse"],edges,"Recomputed R edges")
            target=target_for(sample,index,arm,targets) if index in TRAIN_INDICES else sample["z_target"]
            for key,z in (("latent_mse_to_training_target",zp),("input_latent_mse_to_training_target",sample["z_input"])):
                near(saved[key],float(latent_mse(z,target,sample["valid"])),"Recomputed normalized latent MSE",**p["full_frame_tolerance"])
            fixed_rows.append({"arm":arm,"step":step,"index":index,"group":saved["group"],"trained_view":saved["trained_view"],
                               "global_mae":current["metrics"]["rgb_global_mae"],"global_mae_input":baseline["metrics"]["rgb_global_mae"],
                               "high_mse":freq["high"]["output_error_mse"],"high_mse_input":freq["high"]["input_error_mse"],
                               "edge_mse":edges["prediction"],"edge_mse_input":edges["input"],
                               "application_total":current["metrics"]["total"],"detail":detail,"combined_total":current["metrics"]["total"]+.5*detail,"latent_mse":saved["latent_mse_to_training_target"],"input_latent_mse":saved["input_latent_mse_to_training_target"]})
            arrays_by_step[step][index]={k:arrays[k] for k in ("input","target","prediction")}
    return {"arm":arm,"evidence":evidence,"source_optimizer_updates":STEPS,"fixed_rows":fixed_rows,"gate":decision([r for r in fixed_rows if r["step"]==STEPS]),
            "final_model_sha256":state_hash(final["model"]),"source_native_decodes":report["native_decodes"],"overflow_attempts":len(skipped),"elapsed_seconds":report["elapsed_seconds"],"peak_allocated_bytes":report["peak_allocated_bytes"]},arrays_by_step,initial


def render(p,verified,arrays,folder):
    paths=[]
    for group in ("original_degraded","original_clean"):
        indices=[r["index"] for r in verified[0]["fixed_rows"] if r["step"]==STEPS and r["group"]==group]
        for offset in (0,4):
            fig,axes=plt.subplots(4,4,figsize=(11,11))
            for row,index in zip(axes,indices[offset:offset+4]):
                a=arrays[0][STEPS][index];b=arrays[1][STEPS][index]
                for ax,im,label in zip(row,(a["target"],a["input"],a["prediction"],b["prediction"]),("Original","Input","Cold pixel start","Warm pixel start")):
                    ax.imshow(np.clip(im,0,1));ax.set_title(f"view {index}: {label}",fontsize=8);ax.axis('off')
            fig.suptitle(f"{group}: fixed step192, cold/warm R pixel fits; display clipped only",fontsize=10);fig.tight_layout()
            path=folder/f"{group}_{offset//4+1}.png";fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(3,3,figsize=(14,9))
    for row,index in zip(axes,(0,5,14)):
        for ax,key in zip(row,("global_mae","high_mse","latent_mse")):
            for result in verified:
                points=[r for r in result["fixed_rows"] if r["index"]==index]
                ax.plot([r["step"] for r in points],[r[key] for r in points],'o-',label=result["arm"])
            ax.set(title=f"View {index}: {key}",xlabel='Completed R updates');ax.set_xticks(EVALUATIONS);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Training cases: actual fixed evaluations only; pixel/detail objective is identical across starts; latent is diagnostic only');fig.tight_layout()
    path=folder/'fixed_training_metrics.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    return paths


def summarize(folder):
    folder=folder.resolve();p=read(folder/'protocol.json');check(p);destination=folder/'summary';require(not destination.exists(),"Existing summary cannot be overwritten")
    config=load_config(p['config']);torch.set_num_threads(4)
    with no_training_guard() as guard,TrainingBudget(ROOT/'runs',config.project.budget_seconds,phase=KIND+'_cpu_audit') as budget:
        dataset=open_dataset(config,ROOT,Path(p['manifest']));dataset.audit(budget_check=budget.check);targets=load_targets(p)
        verified=[];arrays=[];initials=[]
        for arm in ARMS:
            result,archive,initial=verify_arm(folder,p,arm,dataset,config,budget,targets);verified.append(result);arrays.append(archive);initials.append(initial)
        for key in ('optimizer','scheduler','scaler','sampler','rng'):assert_equal(initials[0][key],initials[1][key],'Independent R arm initialization '+key)
        destination.mkdir();rows=[r for v in verified for r in v['fixed_rows']]
        path=destination/'fixed_metrics.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        artifacts=[path]+render(p,verified,arrays,destination)
        plot([Path(p['arms'][arm]) for arm in ARMS],destination/'losses.png',window=24);artifacts.extend((destination/'losses.png',destination/'losses.json'))
        evidence=[bound(folder/'protocol.json'),bound(Path(__file__))]+[b for v in verified for b in v['evidence']]
        require(all(file_sha256(b['path'])==b['sha256'] for b in evidence),'Evidence changed during CPU audit');check(p)
        result={'status':'verified_completed_refiner_pixel_start','kind':KIND,'arms':verified,'all_five_nonmodel_initial_states_equal':True,
                'source_optimizer_updates':384,'source_pixel_preflight_vjps':12,'source_fixed_predictions':96,'source_native_decodes':sum(v['source_native_decodes'] for v in verified),
                'new_optimizer_updates':0,'new_native_decodes':0,'new_parameter_forward_passes':0,'execution_guard':dict(guard),
                'verification':'Full-frame application and original-detail metrics, diagnostic latent masked MSE, all ROI metrics, committed optimizer steps and sampling independently checked. Saved R predictions are bound to GPU runner and checkpoints; R/H3 were not rerun on CPU.',
                'evidence':evidence,'artifacts':[bound(a) for a in artifacts],'trained_base_accepted':False,'independent_validation':False,'budget':budget.snapshot()}
        atomic_write(destination/'summary.json',canonical_json(result))
    print(canonical_json({'event':'refiner_pixel_start_cpu_verified','path':str(destination/'summary.json')}).decode(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);summarize(parser.parse_args().output)
