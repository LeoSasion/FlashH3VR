"""C4: target-latent starting point, unchanged original-detail objective and optimization.

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
from scripts.plot_target_start_latent_losses import plot_run
from scripts.probe_original_detail_latent import check as check_control
from scripts.diagnose_supervision_gradients import gradient_only_guard
from scripts.detail_supervision_math import detail_loss, gradient_pair_statistics
from scripts.verify_loss_balance_initialization import assert_equal, state_hash

OLD = ROOT / "logs/gpt-clarity-probe-20260909"
AUDIT = ROOT / "logs/supervision-gradient-20260909-v1"
STEPS, LR, EVALUATIONS = 32, .001, (0, 8, 16, 32)
KIND = "target_start_sample_latent_probe"
CONTROL = ROOT / "logs/original-detail-latent-20260909-v1"
DETAIL_WEIGHT = .5
INIT_FIELDS = ("optimizer", "scheduler", "scaler", "sampler", "rng")
RETENTION = {"high_relative_to_start_max":.01,"edge_relative_to_start_max":.01,
    "global_mae_relative_to_start_max":-.005,"case_count_min":2,"visual_required":True}
INITIALIZATION = "float32 delta = cached z_target - cached z_input; only model state changes from C2"


class SampleLatent(torch.nn.Module):
    """A checkpointable free tensor, with no layers or learned temporal processing."""
    def __init__(self, reference):
        super().__init__()
        self.delta_latent = torch.nn.Parameter(reference.detach().float().clone())


def codes():
    result = code_hashes()
    for name in ("probe_latent_reachability.py", "probe_original_detail_latent.py", "plot_original_detail_latent_losses.py", "diagnose_supervision_gradients.py", "detail_supervision_math.py", "probe_target_start_latent.py", "plot_target_start_latent_losses.py", "verify_loss_balance_initialization.py"):
        result["scripts/"+name] = file_sha256(ROOT / "scripts" / name)
    return result


def check(p):
    require(p["initialization"] == INITIALIZATION and p["retention_criterion"] == RETENTION, "C4 initialization or retention gate changed")
    require(p["kind"] == KIND and p["steps_per_case"] == STEPS and p["maximum_optimizer_updates"] == 96
            and p["learning_rate"] == LR and p["accumulation"] == 1 and p["detail_weight"] == DETAIL_WEIGHT
            and p["evaluation_steps"] == list(EVALUATIONS) and p["code_sha256"] == codes(), "Detail latent scope changed")
    for item in p["evidence"]:
        require(file_sha256(item["path"]) == item["sha256"], "Bound diagnostic evidence changed")
    require(set(p["initial_delta_sha256"]) == {c["name"] for c in p["cases"]}
            and all(len(v)==64 for v in p["initial_delta_sha256"].values())
            and p["full_frame_verification_tolerance"] == {"rtol":2e-4,"atol":2e-7}, "Initial tensor bindings or CPU tolerance changed")
    cp = read(CONTROL / "protocol.json"); check_control(cp)
    require(p["control_protocol"] == bound(CONTROL / "protocol.json") and p["control_cases"] == cp["cases"],
            "Control protocol or cases changed")
    require(p["manifest"] == cp["manifest"], "Control manifest path changed")
    require(p["pilot_criterion"] == cp["pilot_criterion"] and p["comparison_criterion"] == {
        "high_mse_relative_to_control_max":-.005,"edge_mse_relative_to_control_max":-.005,
        "global_mae_relative_to_control_max":.005,"case_count_min":2,"visual_required":True,
        "single_case_high_or_edge_regression_max":.01,
        "purpose":"Pilot continuation, not useful-base or independent-source acceptance"}, "Quality thresholds changed")
    require(p["detail_definition"] == read(AUDIT / "protocol.json")["detail"], "Detail definition changed")
    require(p["preflight"] == {"required":True,"forward_graphs":3,"autograd_grad_calls":9,"optimizer_updates":0,
        "scale":POLICY["init_scale"],"gate":"Finite nonzero app/detail/combined latent gradients; combined cosine positive with app and detail"},
        "Declared preflight changed")
    require(len(p["cases"]) == 3 and all({k:v for k,v in case.items() if k != "run"} ==
            {k:v for k,v in control.items() if k != "run"} for case,control in zip(p["cases"],cp["cases"])),
            "Only the case run directory may differ from control")
    folder = Path(p["config"]).resolve().parent
    require(folder.is_relative_to(ROOT / "logs") and all(Path(case["run"]).resolve() ==
            ROOT / "runs" / folder.name / case["name"] for case in p["cases"]), "Case run path is outside the new experiment")
    require(load_config(p["config"]).model_dump(mode="json") == load_config(cp["config"]).model_dump(mode="json"),
            "Single-factor comparison cannot change base configuration")


def declare(output):
    assert_no_links(output)
    require(output.is_relative_to(ROOT / "logs") and not output.exists(), "Use new diagnostic directory")
    control = read(CONTROL / "protocol.json"); check_control(control)
    summary = read(CONTROL / "summary/summary.json")
    require(summary["status"] == "verified_completed_original_detail_sample_latent_probe"
            and summary["source_optimizer_updates"] == 96 and summary["new_optimizer_updates"] == 0
            and bound(CONTROL / "protocol.json") in summary["inputs"], "Verified completed control required")
    config = load_config(control["config"])
    with no_training_guard(), TrainingBudget(ROOT / "runs", config.project.budget_seconds, phase="target_start_latent_declare") as budget:
        dataset = open_dataset(config, ROOT, Path(control["manifest"])); audit = dataset.audit(budget_check=budget.check)
        cases = [{**case, "run": str(ROOT / "runs" / output.name / case["name"])} for case in control["cases"]]
        for case in cases:
            validate_sample(dataset[case["index"]], case)
            require(not Path(case["run"]).exists(), "Diagnostic run exists")
        initial_hashes = {case["name"]: state_hash((dataset[case["index"]]["z_target"] - dataset[case["index"]]["z_input"]).float()) for case in cases}
        output.mkdir(); write_resolved(config, output / "resolved.yaml")
        protocol = {"status": "declared_before_optimizer_updates", "kind": KIND,
            "created_utc": datetime.now(timezone.utc).isoformat(), "steps_per_case": STEPS,
            "maximum_optimizer_updates": 96, "learning_rate": LR, "accumulation": 1, "evaluation_steps": list(EVALUATIONS),
            "authorization": "User authorized four hours autonomous GPU research; test the proposed target-latent starting point",
            "initialization": INITIALIZATION, "retention_criterion": RETENTION,
            "initial_delta_sha256": initial_hashes, "full_frame_verification_tolerance": {"rtol": 2e-4, "atol": 2e-7},
            "config": str(output / "resolved.yaml"), "manifest": control["manifest"], "cases": cases,
            "control_protocol": bound(CONTROL / "protocol.json"), "control_cases": control["cases"],
            "detail_weight": DETAIL_WEIGHT, "detail_definition": read(AUDIT / "protocol.json")["detail"],
            "objective": "Original application RGB=1, lighting=0.2, latent/perceptual=0; plus 0.5*original detail only",
            "variables": "One original-target-initialized FP32 delta per sample; optimizer/scheduler/scaler/sampler/RNG match C2 zero-step state",
            "forward": "X + feather*(D(zX+delta)-D(zX)), same frozen native H3, strength=1",
            "oracle": "X + feather*(D(E(O))-D(zX)); target-assisted reference only, not an optimum or prediction",
            "refiner_loaded": False, "h3_trainable": False, "generates_new_images": False,
            "independent_validation": False, "trained_base_accepted": False, "deployable": False,
            "question": "Can the unchanged C2 objective preserve and improve a clearer target-latent start? No R or AI changes.",
            "pilot_criterion": control["pilot_criterion"],
            "comparison_criterion": {"high_mse_relative_to_control_max": -.005, "edge_mse_relative_to_control_max": -.005,
                "global_mae_relative_to_control_max": .005, "case_count_min": 2,
                "visual_required": True, "single_case_high_or_edge_regression_max": .01,
                "purpose": "Pilot continuation, not useful-base or independent-source acceptance"},
            "preflight": {"required": True, "forward_graphs": 3, "autograd_grad_calls": 9, "optimizer_updates": 0,
                "scale": POLICY["init_scale"], "gate": "Finite nonzero app/detail/combined latent gradients; combined cosine positive with app and detail"},
            "automatic_extension": False, "data_audit": audit, "code_sha256": codes(),
            "evidence": [bound(path) for path in (output / "resolved.yaml", Path(control["manifest"]), CONTROL / "protocol.json",
                CONTROL / "summary/summary.json", CONTROL / "final_review.json", CONTROL / "integrity_review.json",
                AUDIT / "protocol.json", AUDIT / "results.json", ROOT / config.paths.components_lock)]}
        # Reuse the completed control only while every saved numeric/checkpoint input remains identical.
        protocol["evidence"].extend(item for case in summary["cases"] for item in case["evidence"])
        atomic_write(output / "protocol.json", canonical_json(protocol)); check(protocol)
    print(canonical_json({"event": "target_start_latent_declared", "output": str(output), "maximum_optimizer_updates": 96}).decode(), flush=True)


def append(path, data):
    with path.open("ab") as f:
        f.write(canonical_json(data)+b"\n"); f.flush()


def validate_sample(sample, case):
    require(sample["supervision_group"] == "original_degraded" and sample["clean_pair"] is False,
            "Only original degraded targets are allowed")
    require(sample["view_id"] == case["view_id"] and list(sample["z_input"].shape) == case["latent_shape"],
            "Original input identity or latent shape changed")


def objective_terms(prediction, zp, sample, loss):
    require(sample["supervision_group"] == "original_degraded" and sample["clean_pair"] is False,
            "Detail supervision must use the unenhanced original target")
    terms = dict(loss(prediction, sample["y"], zp, sample["z_target"], sample["valid"],
                      sample["person_mask"], sample["face_mask"]))
    terms["application_total"] = terms["total"]
    terms["detail"] = detail_loss(prediction, sample["y"], sample["valid"], sample["person_mask"], sample["face_mask"])
    terms["total"] = terms["application_total"] + DETAIL_WEIGHT * terms["detail"]
    return terms


def assert_same_initialization(control, current):
    require(control["step"] == current["step"] == 0 and control["stage"] == current["stage"] == "pixel",
            "Initialization comparison requires two step-zero pixel checkpoints")
    require(set(control["model"]) == set(current["model"]) == {"delta_latent"}
            and control["model"]["delta_latent"].dtype == torch.float32
            and torch.count_nonzero(control["model"]["delta_latent"]) == 0
            and current["model"]["delta_latent"].dtype == torch.float32
            and current["model"]["delta_latent"].shape == control["model"]["delta_latent"].shape
            and torch.isfinite(current["model"]["delta_latent"]).all(),
            "Initial state must contain one finite FP32 target-start delta")
    for name in INIT_FIELDS:
        assert_equal(control[name], current[name], "same per-case initialization " + name)


def check_initialization_record(run, control_run, budget):
    record = read(run / "same_initialization.json")
    require(record["all_equal"] is True and record["equal_fields"] == list(INIT_FIELDS)
            and record["current_checkpoint"] == read(run / "initial_checkpoint.json")
            and record["control_checkpoint"] == read(control_run / "initial_checkpoint.json"),
            "Same-initialization evidence changed")
    states = []
    for directory, key in ((control_run,"control_checkpoint"),(run,"current_checkpoint")):
        item=record[key];path=Path(item["path"]);assert_no_links(path)
        require(path.parent.resolve()==(directory/"checkpoints").resolve() and file_sha256(path)==item["sha256"],
                "Initial checkpoint binding or case scope changed")
        state=CheckpointManager(ROOT/"runs",directory,contract=read(directory/"training_contract.json")).read(path)
        budget.validate_resume_snapshot(state["budget"]);states.append(state)
    assert_same_initialization(*states)
    require(record["changed_field"] == "model" and record["initial_delta_sha256"] == state_hash(states[1]["model"]["delta_latent"]), "Target initialization record changed")
    return record


def require_new_or_resume(run, resume):
    assert_no_links(run)
    require(not (run / "training_report.json").exists(), "Completed diagnostic cannot extend")
    require(run.exists() == resume, "New case needs new directory; interrupted case requires --resume")


def archive_interruption(run):
    marker = run / "training_interruption.json"
    assert_no_links(marker)
    if marker.exists():
        destination = run / f"training_interruption_before_resume_{time.time_ns()}.json"
        require(run.resolve().is_relative_to((ROOT / "runs").resolve()) and marker.resolve().parent == run.resolve()
                and destination.resolve().parent == run.resolve() and not destination.exists(), "Unsafe interruption archive")
        marker.rename(destination)


def check_preflight(output, p):
    path = output / "preflight.json"; assert_no_links(path)
    require(path.is_file(), "Completed real H3 preflight required before any optimization")
    result = read(path)
    require(result["status"] == "completed_target_start_latent_preflight"
            and result["protocol"] == bound(output / "protocol.json")
            and result["autograd_grad_calls"] == 9 and result["optimizer_updates"] == 0
            and result["forward_graphs"] == 3 and result["scale"] == POLICY["init_scale"]
            and result["guard"] == {"optimizer_constructions":0,"backward_calls":0,"autograd_grad_calls":9}
            and result["h3_frozen_unchanged"] is True and result["delta_unchanged"] is True,
            "Preflight status, binding or zero-update evidence differs")
    require(len(result["cases"]) == 3, "Preflight requires all three originals")
    for saved, case in zip(result["cases"], p["cases"]):
        require(saved["case"] == case["name"] and saved["view_id"] == case["view_id"]
                and saved["original_index"] == case["index"] and saved["source"] == case["source"]
                and saved["delta_unchanged"] and saved["target_start_matches"] and saved["param_grad_none"],
                "Preflight case or state changed")
        arrays = saved["float_arrays"]
        require(Path(arrays["path"]).resolve().is_relative_to(output.resolve())
                and file_sha256(arrays["path"]) == arrays["sha256"], "Preflight arrays changed")
        require(set(saved["gradients"]["norms"]) == {"original_app","original_detail","total"}
                and all(np.isfinite(v) and v > 0 for v in saved["gradients"]["norms"].values()), "Missing finite gradient")
        for key in ("original_app__total", "original_detail__total"):
            cosine = saved["gradients"]["pairs"][key]["cosine"]
            require(cosine is not None and np.isfinite(cosine) and cosine > 0, "Combined loss opposes original objective")
    return result


def preflight(output):
    p = read(output / "protocol.json"); check(p)
    require(not (output / "preflight.json").exists() and not (output / "preflight").exists(), "Preflight already started")
    require(not any(Path(case["run"]).exists() for case in p["cases"]), "Preflight must precede every optimizer run")
    config = load_config(p["config"]); started = time.monotonic()
    with TrainingBudget(ROOT / "runs", config.project.budget_seconds, phase="target_start_latent_preflight") as budget, gradient_only_guard() as guard:
        rng = capture_rng_state()
        try:
            dataset = open_dataset(config,ROOT,Path(p["manifest"])); dataset.audit(budget_check=budget.check)
            seed_model(42); bridge = load_bridge(config,ROOT,dataset); loss = application_loss(config,ROOT)
            frozen = {n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
            torch.cuda.reset_peak_memory_stats()
            folder = output / "preflight"; folder.mkdir(); rows = []
            for case in p["cases"]:
                budget.check(); sample = move_sample(dataset[case["index"]],"cuda"); validate_sample(sample,case)
                container = SampleLatent(sample["z_target"]-sample["z_input"])
                require(state_hash(container.delta_latent.detach().cpu()) == p["initial_delta_sha256"][case["name"]], "Preflight initializer hash mismatch")
                zp = sample["z_input"] + container.delta_latent
                prediction,_ = restore_pixels(bridge,sample,zp,grad=True,strength=1.)
                require(torch.equal(container.delta_latent, sample["z_target"]-sample["z_input"]), "Target-start initialization changed")
                terms = objective_terms(prediction,zp,sample,loss)
                require(all(torch.isfinite(v).all() for v in terms.values()), "Nonfinite preflight objective")
                vectors = {}
                for i,(label,key) in enumerate((("original_app","application_total"),("original_detail","detail"),("total","total"))):
                    g, = torch.autograd.grad(terms[key]*POLICY["init_scale"],container.delta_latent,retain_graph=i<2)
                    vectors[label] = g.detach().float().cpu()/POLICY["init_scale"]
                    require(torch.isfinite(vectors[label]).all() and torch.count_nonzero(vectors[label]) > 0,
                            "Invalid real H3 loss gradient")
                pairs = {a+"__"+b:gradient_pair_statistics(vectors[a],vectors[b]) for a,b in
                         (("original_app","original_detail"),("original_app","total"),("original_detail","total"))}
                require(all(pairs[key]["cosine"] > 0 for key in ("original_app__total","original_detail__total")),
                        "Combined loss has no common descent direction at initialization")
                recombined = vectors["original_app"].double()+DETAIL_WEIGHT*vectors["original_detail"].double()
                direct = vectors["total"].double()
                x0,y0,x1,y1 = case["roi"]
                def arr(t): return t[0,:,0,y0:y1,x0:x1].detach().float().cpu().permute(1,2,0).numpy().copy()
                arrays = {"grad_"+n:v.numpy() for n,v in vectors.items()}
                arrays.update(delta=container.delta_latent.detach().cpu().numpy(),prediction_roi=arr(prediction),
                              input_roi=arr(sample["x"]),original_roi=arr(sample["y"]))
                path = folder / (case["name"]+".npz"); np.savez_compressed(path,**arrays)
                require(container.delta_latent.grad is None and torch.equal(container.delta_latent, sample["z_target"]-sample["z_input"]),
                        "Preflight mutated a sample latent")
                rows.append({"case":case["name"],"view_id":case["view_id"],"original_index":case["index"],"source":case["source"],
                    "objectives":{n:float(v.detach()) for n,v in terms.items()},"float_arrays":bound(path),
                    "gradients":{"norms":{n:float(v.double().norm()) for n,v in vectors.items()},"pairs":pairs,
                        "recombined_vs_direct":{"relative_l2":float((direct-recombined).norm()/direct.norm()),
                            "max_abs":float((direct-recombined).abs().max()),**gradient_pair_statistics(direct,recombined)}},
                    "target_start_matches":True, "latent_roundtrip_max_abs":float((zp-sample["z_target"]).abs().max()),"delta_unchanged":True,"param_grad_none":True})
                del terms, prediction, zp, container, sample, vectors, arrays, g
            require(dict(guard)=={"optimizer_constructions":0,"backward_calls":0,"autograd_grad_calls":9}, "Unexpected preflight operations")
            require(frozen=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()}
                    and not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()), "H3 changed during preflight")
            check(p)
            result={"status":"completed_target_start_latent_preflight","protocol":bound(output/"protocol.json"),"cases":rows,
                "autograd_grad_calls":9,"forward_graphs":3,"optimizer_updates":0,"guard":dict(guard),
                "h3_frozen_unchanged":True,"delta_unchanged":True,"scale":POLICY["init_scale"],
                "elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "budget":budget.snapshot(),"device":torch.cuda.get_device_name(),"trained_base_accepted":False}
            atomic_write(output/"preflight.json",canonical_json(result)); check_preflight(output,p)
        finally: restore_rng_state(rng)
    print(canonical_json({"event":"target_start_latent_preflight_completed","autograd_grad_calls":9,"optimizer_updates":0}).decode(),flush=True)


def latest_case_checkpoint(manager):
    """Find receipts only in this case; historical auto lookup scans one run level."""
    candidates = []
    for receipt_file in manager.directory.glob("checkpoint-*.json"):
        path, receipt = manager._receipt(receipt_file.with_suffix(".pt"))
        require(receipt["contract_id"] == manager.contract_id, "Another case's receipt entered this directory")
        candidates.append((receipt["created_ns"], path))
    require(candidates, "No committed case checkpoint")
    path = max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    state = manager.read(path)
    require(state["stage"] == "pixel" and type(state["step"]) is int and 0 <= state["step"] <= STEPS,
            "Checkpoint exceeds the declared diagnostic stage or step cap")
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
            if step == 0: require(torch.equal(container.delta_latent, sample["z_target"]-sample["z_input"]), "Initial delta must match target-start contract")
            x0,y0,x1,y1 = case["roi"]
            def arr(t): return t[0,:,0,y0:y1,x0:x1].detach().float().cpu().permute(1,2,0).numpy().copy()
            arrays = {"input": arr(sample["x"]), "target": arr(sample["y"]), "prediction": arr(pred)}
            result = {"status": "completed_target_assisted_latent_evaluation", "step": step, "view_id": sample["view_id"],
                "checkpoint": bound(checkpoint), "current": measure(sample,pred,zp,loss),
                "input": measure(sample,sample["x"],sample["z_input"],loss),
                "latent_delta_l2": float(container.delta_latent.float().norm()), "latent_delta_absmax": float(container.delta_latent.abs().max())}
            terms = objective_terms(pred,zp,sample,loss)
            result["detail_objective"] = {name:float(terms[name]) for name in ("detail","application_total","total")}
            result["detail_objective"]["detail_weight"] = DETAIL_WEIGHT
            if step == 0:
                oracle, _ = restore_pixels(bridge, sample, sample["z_target"], grad=False, strength=1.)
                arrays["oracle"] = arr(oracle)
                result["oracle"] = measure(sample,oracle,sample["z_target"],loss)
                result["oracle_frequency"] = frequency_metrics(arrays["input"],arrays["target"],arrays["oracle"])
                result["oracle_edge_mse"] = edge_mse(arrays["oracle"],arrays["target"])
                result["start_latent_roundtrip_max_abs"] = float((zp-sample["z_target"]).abs().max())
                result["start_output_oracle_max_abs"] = float((pred-oracle).abs().max())
            full = {"prediction":pred.detach().float().cpu().numpy(),"z_prediction":zp.detach().float().cpu().numpy()}
            if step == 0: full["oracle"] = oracle.detach().float().cpu().numpy()
            full_path=folder/"full.npz"; np.savez_compressed(full_path,**full)
            result["full_arrays"]=bound(full_path)
            path = folder / "roi.npz"; np.savez_compressed(path,**arrays)
            result.update(float_arrays=bound(path), frequency=frequency_metrics(arrays["input"],arrays["target"],arrays["prediction"]),
                edge_mse={k:edge_mse(arrays[k],arrays["target"]) for k in ("input","prediction")},
                guard=dict(guard), new_optimizer_updates=0, budget=budget.snapshot(), deployable=False)
            atomic_write(folder / "metrics.json", canonical_json(result))
    finally: restore_rng_state(rng)
    return result


def train_case(output, case_name, resume=False):
    p = read(output / "protocol.json"); check(p)
    check_preflight(output,p)
    case = next(c for c in p["cases"] if c["name"] == case_name)
    run = Path(case["run"]); assert_no_links(run)
    require_new_or_resume(run,resume)
    config = load_config(p["config"])
    if not resume: run.mkdir(parents=True); write_resolved(config,run/"resolved.yaml")
    step, manager, checkpoint = 0, None, None
    transaction = {"in_optimizer":False}; started=time.monotonic()
    with TrainingBudget(ROOT / "runs",config.project.budget_seconds,phase="target_start_latent_"+case_name) as budget:
        try:
            budget.validate_resume_snapshot(read(output/"preflight.json")["budget"])
            dataset=open_dataset(config,ROOT,Path(p["manifest"]));dataset.audit(budget_check=budget.check)
            seed_model(42);bridge=load_bridge(config,ROOT,dataset)
            sample=move_sample(dataset[case["index"]],"cuda")
            validate_sample(sample,case)
            require(all(dataset.views[case["index"]][key] == case[key] for key in ("z_input_key","z_target_key")),
                    "Original latent cache binding changed")
            container=SampleLatent(sample["z_target"]-sample["z_input"])
            require(state_hash(container.delta_latent.detach().cpu()) == p["initial_delta_sha256"][case_name], "Training initializer hash mismatch")
            optimizer=torch.optim.AdamW(container.parameters(),lr=LR,weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
            scaler=make_scaler("pixel");sampler=StatefulSampler(1,seed=42,shuffle=False);loss=application_loss(config,ROOT)
            contract=make_contract(config,ROOT,dataset,"pixel",STEPS)
            contract["compatibility"]["architecture"]={"kind":"free_per_sample_latent_tensor","shape":case["latent_shape"]}
            contract["experiment"]={"kind":KIND,"protocol_sha256":file_sha256(output/"protocol.json"),"case":case,
                "code_sha256":p["code_sha256"],"refiner_loaded":False,"deployable":False,
                "detail_definition":p["detail_definition"],"detail_weight":DETAIL_WEIGHT}
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
                archive_interruption(run)
            def save():
                optimizer.zero_grad(set_to_none=True)
                return manager.save(model=container,optimizer=optimizer,scheduler=scheduler,scaler=scaler,sampler=sampler,
                    stage="pixel",step=step,budget=budget.snapshot(),resolved_config=config.model_dump(mode="json"),extra=extra)
            if checkpoint is None:
                checkpoint=save();atomic_write(run/"initial_checkpoint.json",canonical_json(bound(checkpoint)))
                control_case=next(c for c in p["control_cases"] if c["name"]==case_name)
                control_run=Path(control_case["run"]);control_contract=read(control_run/"training_contract.json")
                require(control_contract["experiment"]["case"]==control_case and
                        control_contract["experiment"]["protocol_sha256"]==p["control_protocol"]["sha256"],"Wrong control initializer")
                binding=read(control_run/"initial_checkpoint.json");control_path=Path(binding["path"])
                require(file_sha256(control_path)==binding["sha256"],"Control initialization changed")
                control_state=CheckpointManager(ROOT/"runs",control_run,contract=control_contract).read(control_path)
                budget.validate_resume_snapshot(control_state["budget"])
                assert_same_initialization(control_state,manager.read(checkpoint))
                atomic_write(run/"same_initialization.json",canonical_json({"control_checkpoint":bound(control_path),
                    "current_checkpoint":bound(checkpoint),"equal_fields":list(INIT_FIELDS),"all_equal":True, "changed_field":"model",
                    "initial_delta_sha256":state_hash(container.delta_latent.detach().cpu())}))
                del control_state
            else:
                control_case=next(c for c in p["control_cases"] if c["name"]==case_name)
                init_record=check_initialization_record(run,Path(control_case["run"]),budget)
                require(init_record["initial_delta_sha256"] == p["initial_delta_sha256"][case_name], "Resume initializer binding mismatch")
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
                terms=objective_terms(pred,zp,sample,loss)
                return terms,{"view_id":sample["view_id"],"asset_id":sample["asset_id"],"clean_pair":False,
                    "supervision_group":"original_degraded","optimization_variable":"sample_delta_latent"}
            while step<STEPS:
                record=update_window(container,optimizer,scheduler,scaler,sampler,get_terms,1,1.,transaction)
                if not record["optimizer_updated"]:
                    extra["overflow_attempts"]+=1;extra["consecutive_overflows"]+=1;checkpoint=save()
                    append(run/"skipped_updates.jsonl",{"completed_step":step,**record})
                    require(not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 acquired gradients")
                    require(extra["consecutive_overflows"]<POLICY["max_consecutive_overflows"],"Repeated overflow")
                    continue
                # Count the already committed optimizer update before any fallible post-update check.
                step+=1;extra["successful_case_visits"]+=1;extra["consecutive_overflows"]=0
                # The shared updater groups non-FFN/non-scene parameters as output. Here there is only one free tensor.
                norms=record.pop("gradient_norms")
                record["gradient_norms"]={"sample_latent":norms["output"]}
                record.update(step=step,phase="pixel",diagnostic_kind=KIND,case=case_name,learning_rate=LR,used_seconds=budget.used)
                append(run/"metrics.jsonl",record)
                require(norms["spatial"]==norms["scene"]==0.,"Unexpected network parameter group")
                require(not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()),"H3 acquired gradients")
                if step%8==0:checkpoint=save()
                print(canonical_json({"event":"target_start_latent_update","case":case_name,"step":step,
                    "total":record["losses"]["total"],"application_total":record["losses"]["application_total"],
                    "weighted_detail":DETAIL_WEIGHT*record["losses"]["detail"]}).decode(),flush=True)
                if step==1 or step%8==0:
                    rng=capture_rng_state()
                    try:plot_run(run,run/"training_losses.png",window=4)
                    finally:restore_rng_state(rng)
                if step in EVALUATIONS:evaluate_current()
            require(extra["successful_case_visits"]==STEPS,"Incomplete per-case optimizer count")
            require(frozen=={n:(id(v),v._version) for n,v in bridge.backend.model.named_parameters()},"H3 changed")
            check(p);check_preflight(output,p);extra["phase_complete"]=True;checkpoint=save()
            report={"status":"completed_target_start_sample_latent_probe","kind":KIND,"phase":"pixel",
                "optimizer_steps":step,"checkpoint":str(checkpoint),"case":case,"extra":extra,"refiner_loaded":False,
                "h3_frozen_unchanged":True,"elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "budget":budget.snapshot(),"trained_base_accepted":False,"deployable":False,"independent_validation":False}
            report["preflight"]=bound(output/"preflight.json")
            atomic_write(run/"training_report.json",canonical_json(report))
        except BaseException as exc:
            if manager is not None and 'save' in locals() and not transaction["in_optimizer"]:checkpoint=save()
            atomic_write(run/"training_interruption.json",canonical_json({"status":"interrupted","last_completed_step":step,
                "checkpoint":str(checkpoint) if checkpoint else None,"optimizer_may_have_partially_updated":transaction["in_optimizer"],
                "error":type(exc).__name__,"message":str(exc),"automatic_restart":False,"budget":budget.snapshot()}))
            if (run/"metrics.jsonl").exists():plot_run(run,run/"training_losses.png",window=4)
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',required=True,choices=('declare','preflight','run'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--case')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=Path(args.output).resolve()
    if args.action=='declare':
        require(not args.case and not args.resume,'Declaration cannot resume');declare(output)
    elif args.action=='preflight':
        require(not args.case and not args.resume,'Preflight checks all three cases before training');preflight(output)
    else:
        p=read(output/'protocol.json');names=[c['name'] for c in p['cases']]
        require(not args.resume or args.case in names,'Resume requires one explicit case')
        require(args.case is None or args.case in names,'Unknown case')
        for name in ([args.case] if args.case else names):train_case(output,name,args.resume)
