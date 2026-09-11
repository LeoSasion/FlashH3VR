"""Optimize only three independent sample latents, never a restoration network.

Target-assisted diagnostics cannot be deployed or registered as a useful base.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from h3ce.cache.keys import canonical_json, file_sha256, digest
from h3ce.cache.store import atomic_write, assert_no_links
from h3ce.config import load_config, write_resolved
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget, capture_rng_state, restore_rng_state
from h3ce.train.engine import make_contract
from h3ce.train.guard import no_training_guard
from h3ce.train.perceptual import application_loss
from h3ce.train.pipeline import move_sample, restore_pixels
from h3ce.train.precision import POLICY, make_scaler
from h3ce.train.preflight import open_dataset, load_bridge, require, seed_model
from h3ce.train.sampler import StatefulSampler
from scripts.gpt_clarity_probe_common import read, bound, code_hashes, check_protocol, update_window
from scripts.diagnose_supervision_gradients import check as check_gradients
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.evaluate_gpt_clarity_probe import edge_mse
from scripts.evaluate_bootstrap_checkpoint import measure
from scripts.plot_training_losses import plot

OLD = ROOT / "logs/gpt-clarity-probe-20260909"
AUDIT = ROOT / "logs/supervision-gradient-20260909-v1"
STEPS, LR, EVALUATIONS = 32, .001, (0, 8, 16, 32)
KIND = "target_assisted_sample_latent_diagnostic"


class SampleLatent(torch.nn.Module):
    """A checkpointable free tensor, with no layers or learned temporal processing."""
    def __init__(self, reference):
        super().__init__()
        self.delta_latent = torch.nn.Parameter(torch.zeros_like(reference, dtype=torch.float32))


def codes():
    result = code_hashes()
    for name in ("probe_latent_reachability.py", "diagnose_supervision_gradients.py", "detail_supervision_math.py"):
        result["scripts/"+name] = file_sha256(ROOT / "scripts" / name)
    return result


def check(p):
    require(p["kind"] == KIND and p["steps_per_case"] == STEPS and p["maximum_optimizer_updates"] == 96
            and p["learning_rate"] == LR and p["code_sha256"] == codes(), "Latent diagnostic scope changed")
    for item in p["evidence"]:
        require(file_sha256(item["path"]) == item["sha256"], "Bound diagnostic evidence changed")
    check_protocol(read(OLD / "protocol.json")); check_gradients(read(AUDIT / "protocol.json"))


def declare(output):
    assert_no_links(output)
    require(output.is_relative_to(ROOT / "logs") and not output.exists(), "Use new diagnostic directory")
    old = read(OLD / "protocol.json"); check_protocol(old)
    gp = read(AUDIT / "protocol.json"); check_gradients(gp)
    summary = read(AUDIT / "summary/summary.json")
    require(summary["status"] == "verified_completed_supervision_gradient_audit" and summary["new_optimizer_updates"] == 0,
            "Verified real H3 gradient audit required")
    config = load_config(old["config"])
    config.training.gradient_accumulation = 1
    config.training.stages.bootstrap_pixel.lr = LR
    config.training.stages.bootstrap_pixel.max_steps = STEPS
    with no_training_guard(), TrainingBudget(ROOT / "runs", config.project.budget_seconds, phase="latent_reachability_declare") as budget:
        dataset = open_dataset(config, ROOT, Path(old["manifest"])); audit = dataset.audit(budget_check=budget.check)
        cases = []
        for case in gp["cases"]:
            sample = dataset[case["original"]]
            require(sample["supervision_group"] == "original_degraded" and not sample["clean_pair"], "Use original target O only")
            name = f"original_{case['original']:02d}"
            run = ROOT / "runs" / output.name / name
            require(not run.exists(), "Diagnostic run exists")
            cases.append({"name": name, "index": case["original"], "run": str(run), "view_id": sample["view_id"],
                "source": dataset.sources[sample["asset_id"]]["path"], "latent_shape": list(sample["z_input"].shape),
                "z_input_key": dataset.views[case["original"]]["z_input_key"],
                "z_target_key": dataset.views[case["original"]]["z_target_key"],
                "roi": old["rois"][sample["view_id"]]})
        output.mkdir(); write_resolved(config, output / "resolved.yaml")
        protocol = {"status": "declared_before_optimizer_updates", "kind": KIND,
            "created_utc": datetime.now(timezone.utc).isoformat(), "steps_per_case": STEPS,
            "maximum_optimizer_updates": 96, "learning_rate": LR, "accumulation": 1, "evaluation_steps": list(EVALUATIONS),
            "authorization": "User authorized ordered development, then continued; weak-detail candidate failed direction gate, so diagnose local latent optimizability",
            "config": str(output / "resolved.yaml"), "manifest": old["manifest"], "cases": cases,
            "objective": "Unchanged original-target application loss RGB=1, linear-RGB lighting=0.2, latent/perceptual=0",
            "variables": "One zero-initialized float32 delta_latent per sample; separate AdamW/scaler/sampler per sample",
            "forward": "X + feather*(D(zX+delta)-D(zX)), same frozen native H3, strength=1",
            "oracle": "X + feather*(D(E(O))-D(zX)); target-assisted reference only, not an optimum or prediction",
            "refiner_loaded": False, "h3_trainable": False, "generates_new_images": False,
            "independent_validation": False, "trained_base_accepted": False, "deployable": False,
            "question": "Can the existing output path and original objective exploit free per-sample latents? Not a comparison of R and latent parameter efficiency.",
            "pilot_criterion": {"high_mse_relative_to_input_max": -.005, "edge_mse_relative_to_input_max": -.005,
                "global_mae_relative_to_input_max": 0., "case_count_min": 2,
                "visual": "Review all three fixed cases; do not count noise, halos, or geometry/tone drift as restored detail",
                "failure_scope": "A bounded failure cannot prove decoder incapacity; inspect precision/local optimization next"},
            "automatic_extension": False, "data_audit": audit, "code_sha256": codes(),
            "evidence": [bound(path) for path in (output / "resolved.yaml", Path(old["manifest"]), OLD / "protocol.json",
                AUDIT / "protocol.json", AUDIT / "results.json", AUDIT / "summary/summary.json", ROOT / config.paths.components_lock)]}
        atomic_write(output / "protocol.json", canonical_json(protocol)); check(protocol)
    print(canonical_json({"event": "latent_diagnostic_declared", "output": str(output), "maximum_optimizer_updates": 96}).decode(), flush=True)


def append(path, data):
    with path.open("ab") as f:
        f.write(canonical_json(data)+b"\n"); f.flush()


def latest_case_checkpoint(manager):
    """Find receipts only in this case; historical auto lookup scans one run level."""
    candidates = []
    for receipt_file in manager.directory.glob("checkpoint-*.json"):
        path, receipt = manager._receipt(receipt_file.with_suffix(".pt"))
        require(receipt["contract_id"] == manager.contract_id, "Another case's receipt entered this directory")
        candidates.append((receipt["created_ns"], path))
    require(candidates, "No committed case checkpoint")
    path = max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    manager.read(path)
    return path


def fixed(container, bridge, sample, loss, case, run, step, checkpoint, budget, protocol):
    folder = run / f"fixed_step_{step:04d}"
    require(not folder.exists(), "Fixed output already exists")
    folder.mkdir()
    rng = capture_rng_state()
    try:
        with no_training_guard() as guard:
            zp = sample["z_input"] + container.delta_latent
            pred, _ = restore_pixels(bridge, sample, zp, grad=False, strength=1.)
            if step == 0: require(torch.equal(pred, sample["x"]), "Initial diagnostic must equal X")
            x0,y0,x1,y1 = case["roi"]
            def arr(t): return t[0,:,0,y0:y1,x0:x1].detach().float().cpu().permute(1,2,0).numpy().copy()
            arrays = {"input": arr(sample["x"]), "target": arr(sample["y"]), "prediction": arr(pred)}
            result = {"status": "completed_target_assisted_latent_evaluation", "step": step, "view_id": sample["view_id"],
                "checkpoint": bound(checkpoint), "current": measure(sample,pred,zp,loss),
                "input": measure(sample,sample["x"],sample["z_input"],loss),
                "latent_delta_l2": float(container.delta_latent.float().norm()), "latent_delta_absmax": float(container.delta_latent.abs().max())}
            if step == 0:
                oracle, _ = restore_pixels(bridge, sample, sample["z_target"], grad=False, strength=1.)
                arrays["oracle"] = arr(oracle)
                result["oracle"] = measure(sample,oracle,sample["z_target"],loss)
                result["oracle_frequency"] = frequency_metrics(arrays["input"],arrays["target"],arrays["oracle"])
                result["oracle_edge_mse"] = edge_mse(arrays["oracle"],arrays["target"])
            path = folder / "roi.npz"; np.savez_compressed(path,**arrays)
            result.update(float_arrays=bound(path), frequency=frequency_metrics(arrays["input"],arrays["target"],arrays["prediction"]),
                edge_mse={k:edge_mse(arrays[k],arrays["target"]) for k in ("input","prediction")},
                guard=dict(guard), new_optimizer_updates=0, budget=budget.snapshot(), deployable=False)
            atomic_write(folder / "metrics.json", canonical_json(result))
    finally: restore_rng_state(rng)
    return result


def train_case(output, case_name, resume=False):
    p = read(output / "protocol.json"); check(p)
    case = next(c for c in p["cases"] if c["name"] == case_name)
    run = Path(case["run"]); assert_no_links(run)
    require(not (run / "training_report.json").exists(), "Completed diagnostic cannot extend")
    require(run.exists() == resume, "New case needs new directory; interrupted case requires --resume")
    config = load_config(p["config"])
    if not resume: run.mkdir(parents=True); write_resolved(config,run/"resolved.yaml")
    step, manager, checkpoint = 0, None, None
    transaction = {"in_optimizer":False}; started=time.monotonic()
    with TrainingBudget(ROOT / "runs",config.project.budget_seconds,phase="latent_reachability_"+case_name) as budget:
        try:
            dataset=open_dataset(config,ROOT,Path(p["manifest"]));dataset.audit(budget_check=budget.check)
            seed_model(42);bridge=load_bridge(config,ROOT,dataset)
            sample=move_sample(dataset[case["index"]],"cuda")
            require(sample["view_id"]==case["view_id"] and list(sample["z_input"].shape)==case["latent_shape"],"Diagnostic sample changed")
            container=SampleLatent(sample["z_input"])
            optimizer=torch.optim.AdamW(container.parameters(),lr=LR,weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
            scaler=make_scaler("pixel");sampler=StatefulSampler(1,seed=42,shuffle=False);loss=application_loss(config,ROOT)
            contract=make_contract(config,ROOT,dataset,"pixel",STEPS)
            contract["compatibility"]["architecture"]={"kind":"free_per_sample_latent_tensor","shape":case["latent_shape"]}
            contract["experiment"]={"kind":KIND,"protocol_sha256":file_sha256(output/"protocol.json"),"case":case,
                "code_sha256":p["code_sha256"],"refiner_loaded":False,"deployable":False}
            if resume: require(read(run/"training_contract.json")==contract,"Resume contract changed")
            else:
                atomic_write(run/"training_contract.json",canonical_json(contract))
                atomic_write(run/"training_request.json",canonical_json({"protocol":str(output/"protocol.json"),"case":case_name}))
            manager=CheckpointManager(ROOT/"runs",run,contract=contract)
            extra={"overflow_attempts":0,"consecutive_overflows":0,"successful_case_visits":0,"kind":KIND,"deployable":False}
            frozen={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
            if resume:
                checkpoint=latest_case_checkpoint(manager)
                state=manager.restore(checkpoint,model=container,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler)
                budget.validate_resume_snapshot(state["budget"]);step,extra=state["step"],state["extra"]
                path=run/"metrics.jsonl";raw=path.read_bytes() if path.exists() else b""
                rows=[__import__('json').loads(line) for line in raw.splitlines()]
                require([r["step"] for r in rows[:step]]==list(range(1,step+1)),"Missing committed records")
                if len(rows)>step:
                    atomic_write(run/f"metrics_before_resume_{time.time_ns()}.jsonl",raw)
                    atomic_write(path,b"".join(canonical_json(r)+b"\n" for r in rows[:step]))
            def save():
                optimizer.zero_grad(set_to_none=True)
                return manager.save(model=container,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler,
                    stage="pixel",step=step,budget=budget.snapshot(),resolved_config=config.model_dump(mode="json"),extra=extra)
            if checkpoint is None:
                checkpoint=save();atomic_write(run/"initial_checkpoint.json",canonical_json(bound(checkpoint)))
            torch.cuda.reset_peak_memory_stats()
            def evaluate_current():
                folder=run/f"fixed_step_{step:04d}";path=folder/"metrics.json"
                if path.exists():
                    saved=read(path);require(saved["step"]==step and saved["status"]=="completed_target_assisted_latent_evaluation","Bad existing fixed state")
                    state=manager.read(Path(saved["checkpoint"]["path"]))
                    require(torch.equal(state["model"]["delta_latent"],container.delta_latent.detach().cpu()),"Fixed output checkpoint differs")
                    return
                if folder.exists():
                    destination=run/f"incomplete_fixed_{step:04d}_{time.time_ns()}";assert_no_links(folder)
                    require(folder.resolve().is_relative_to(run.resolve()) and destination.resolve().is_relative_to(run.resolve()),"Unsafe archive path")
                    folder.rename(destination)
                fixed(container,bridge,sample,loss,case,run,step,checkpoint,budget,p)
            if step in EVALUATIONS:evaluate_current()
            def get_terms(index):
                require(index==0,"Single sample diagnostic sampler changed");budget.check()
                zp=sample["z_input"]+container.delta_latent
                pred,_=restore_pixels(bridge,sample,zp,grad=True,strength=1.)
                terms=loss(pred,sample["y"],zp,sample["z_target"],sample["valid"],sample["person_mask"],sample["face_mask"])
                return terms,{"view_id":sample["view_id"],"asset_id":sample["asset_id"],"clean_pair":False,
                    "supervision_group":"original_degraded","optimization_variable":"sample_delta_latent"}
            while step<STEPS:
                record=update_window(container,optimizer,scheduler,scaler,sampler,get_terms,1,1.,transaction)
                require(not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 acquired gradients")
                if not record["optimizer_updated"]:
                    extra["overflow_attempts"]+=1;extra["consecutive_overflows"]+=1;checkpoint=save()
                    append(run/"skipped_updates.jsonl",{"completed_step":step,**record})
                    require(extra["consecutive_overflows"]<POLICY["max_consecutive_overflows"],"Repeated overflow")
                    continue
                # The shared updater groups non-FFN/non-scene parameters as output. Here there is only one free tensor.
                norms=record.pop("gradient_norms")
                require(norms["spatial"]==norms["scene"]==0.,"Unexpected network parameter group")
                record["gradient_norms"]={"sample_latent":norms["output"]}
                step+=1;extra["successful_case_visits"]+=1;extra["consecutive_overflows"]=0
                record.update(step=step,phase="pixel",diagnostic_kind=KIND,case=case_name,learning_rate=LR,used_seconds=budget.used)
                append(run/"metrics.jsonl",record)
                if step%8==0:checkpoint=save()
                print(canonical_json({"event":"latent_diagnostic_update","case":case_name,"step":step,"total":record["losses"]["total"]}).decode(),flush=True)
                if step==1 or step%8==0:
                    rng=capture_rng_state()
                    try:plot([run],run/"training_losses.png",window=4)
                    finally:restore_rng_state(rng)
                if step in EVALUATIONS:evaluate_current()
            require(extra["successful_case_visits"]==STEPS,"Incomplete per-case optimizer count")
            require(frozen=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()},"H3 changed")
            check(p);extra["phase_complete"]=True;checkpoint=save()
            report={"status":"completed_target_assisted_sample_latent_diagnostic","kind":KIND,"phase":"pixel",
                "optimizer_steps":step,"checkpoint":str(checkpoint),"case":case,"extra":extra,"refiner_loaded":False,
                "h3_frozen_unchanged":True,"elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "budget":budget.snapshot(),"trained_base_accepted":False,"deployable":False,"independent_validation":False}
            atomic_write(run/"training_report.json",canonical_json(report))
        except BaseException as exc:
            if manager is not None and 'save' in locals() and not transaction["in_optimizer"]:checkpoint=save()
            atomic_write(run/"training_interruption.json",canonical_json({"status":"interrupted","last_completed_step":step,
                "checkpoint":str(checkpoint) if checkpoint else None,"optimizer_may_have_partially_updated":transaction["in_optimizer"],
                "error":type(exc).__name__,"message":str(exc),"automatic_restart":False,"budget":budget.snapshot()}))
            if (run/"metrics.jsonl").exists():plot([run],run/"training_losses.png",window=4)
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',required=True,choices=('declare','run'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--case')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=Path(args.output).resolve()
    if args.action=='declare':
        require(not args.case and not args.resume,'Declaration cannot resume');declare(output)
    else:
        p=read(output/'protocol.json');names=[c['name'] for c in p['cases']]
        require(not args.resume or args.case in names,'Resume requires one explicit case')
        require(args.case is None or args.case in names,'Unknown case')
        for name in ([args.case] if args.case else names):train_case(output,name,args.resume)
