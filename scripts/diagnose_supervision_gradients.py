"""Bounded real-H3 supervision audit. No optimizers or parameter updates."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from h3ce.cache.keys import canonical_json, file_sha256, digest
from h3ce.cache.store import atomic_write, assert_no_links
from h3ce.config import load_config
from h3ce.doctor import diagnose
from h3ce.model import SpatialRefinerV2
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget, capture_rng_state, restore_rng_state
from h3ce.train.engine import implementation_hashes
from h3ce.train.guard import no_training_guard
from h3ce.train.perceptual import application_loss
from h3ce.train.pipeline import move_sample, refine_latent, restore_pixels
from h3ce.train.precision import POLICY
from h3ce.train.preflight import open_dataset, load_bridge, require, seed_model, execution_contract
from scripts.gpt_clarity_probe_common import read, bound, check_protocol, make_slots
from scripts.detail_supervision_math import detail_loss, gradient_pair_statistics
from scripts.diagnose_detail_frequency import dctn, band_masks, frequency_metrics
from scripts.verify_loss_balance_initialization import state_hash

OLD = ROOT / "logs/gpt-clarity-probe-20260909"
LAMBDA = .05
STATES = ("initial", "control_76", "mixed_76")


@contextmanager
def gradient_only_guard():
    counts = {"optimizer_constructions": 0, "backward_calls": 0, "autograd_grad_calls": 0}
    original = torch.autograd.grad
    def forbid(key):
        def blocked(*args, **kwargs):
            counts[key] += 1
            raise RuntimeError("Supervision diagnostic forbids " + key)
        return blocked
    def counted(*args, **kwargs):
        counts["autograd_grad_calls"] += 1
        return original(*args, **kwargs)
    with patch.object(torch.optim.Optimizer, "__init__", forbid("optimizer_constructions")), \
         patch.object(torch.autograd, "backward", forbid("backward_calls")), \
         patch.object(torch.autograd, "grad", counted):
        yield counts


def load_saved(name, old):
    arm = "mixed" if name == "mixed_76" else "control"
    run = Path(old["arms"][arm])
    contract = read(run / "training_contract.json")
    report = read(run / "training_report.json")
    require(report["status"] == "completed_bounded_mixed_probe" and report["optimizer_steps"] == 76
            and report["h3_frozen_unchanged"] and not report["trained_base_accepted"], "Completed frozen H3 run required")
    require(contract["experiment"]["protocol_sha256"] == file_sha256(OLD / "protocol.json")
            and contract["experiment"]["arm"] == arm, "Wrong checkpoint experiment")
    cp = Path(read(run / "initial_checkpoint.json")["path"] if name == "initial" else report["checkpoint"])
    state = CheckpointManager(ROOT / "runs", run, contract=contract).read(cp)
    require(state["stage"] == "mixed_probe" and state["step"] == (0 if name == "initial" else 76), "Wrong checkpoint boundary")
    require(digest(state["resolved_config"]) == contract["resolved_sha256"], "Saved config changed")
    require(state["contract"]["compatibility"]["implementation_sha256"] == implementation_hashes(ROOT), "Old core implementation changed")
    return state, cp


def cases_for(dataset, slots):
    result = []
    for slot in slots:
        if slot["slot_group"] != "ai_paired":
            continue
        parent = dataset.views[slot["control"]]
        originals = [i for i, v in enumerate(dataset.views)
            if v["asset_id"] == parent["asset_id"] and v["target_id"] == parent["target_id"]
            and dataset.variants[v["variant_id"]]["clean_pair"] and v["crop_xyxy"] == parent["crop_xyxy"]
            and v["bucket_hw"] == parent["bucket_hw"] and v["mode"] == parent["mode"]]
        require(len(originals) == 1, "AI parent needs exactly one matching original clean view")
        result.append({"kind": "paired", "original": slot["control"], "ai": slot["mixed"], "clean": originals[0]})
    require(len(result) == 3, "Keep three previously accepted geometric pairs")
    return result


def declare(output):
    assert_no_links(output)
    require(output.is_relative_to(ROOT / "logs") and not output.exists(), "Use a new logs directory")
    old = read(OLD / "protocol.json"); check_protocol(old)
    config = load_config(old["config"])
    with no_training_guard(), TrainingBudget(ROOT / "runs", config.project.budget_seconds, phase="supervision_gradient_declare") as budget:
        doctor = diagnose(config, ROOT)
        require(all(doctor["milestones"][m]["acceptance"].startswith("passed") for m in ("M0", "M1")), "M0/M1 acceptance required")
        dataset = open_dataset(config, ROOT, Path(old["manifest"]))
        audit = dataset.audit(budget_check=budget.check)
        slots = make_slots(dataset); require(slots == old["slots"], "Input exposure changed")
        checkpoints = {}
        for name in STATES:
            state, cp = load_saved(name, old)
            checkpoints[name] = {**bound(cp), "model_sha256": state_hash(state["model"])}
        code = [Path(__file__), ROOT / "scripts/detail_supervision_math.py"]
        protocol = {"status": "declared_before_gradient_execution", "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": "User requested setting the future development direction and proceeding in order after the proposed gradient audit",
            "parent_protocol": bound(OLD / "protocol.json"), "parent_summary": bound(OLD / "comparison/summary.json"),
            "config": bound(Path(old["config"])), "manifest": bound(Path(old["manifest"])),
            "checkpoints": checkpoints, "cases": cases_for(dataset, slots), "rois": old["rois"], "audit": audit,
            "code": [bound(p) for p in code], "scale": POLICY["init_scale"], "candidate_lambda": LAMBDA,
            "detail": {"domain": "sRGB working pixels", "sigma_reference": 2., "reference_short_edge": 512,
                "truncate": 3., "mask": "valid crop interior eroded by Gaussian radius; original/face/person normalization",
                "epsilon": .001},
            "candidate": "original_app + 0.05 * ai_detail; replaces full-AI RGB/light supervision only in AI slots",
            "states": list(STATES), "repeat": "control_76 first paired case, new forward and new gradients",
            "expected_forward_graphs": 19, "expected_autograd_grad_calls": 146, "maximum_optimizer_updates": 0,
            "gradients": "Unclipped, unscaled, per single sample, not divided by accumulation; not an AdamW update direction",
            "limits": ["Three paired sources from one episode; not independent validation", "No loss plot can imply training",
                "Gradient alignment is local, not an outcome or a convergence proof", "RGB ROI frequency is not latent spatial frequency"],
            "decision": "Use measured gradient directions to select bounded follow-up; no automatic training from this script"}
        output.mkdir(); atomic_write(output / "protocol.json", canonical_json(protocol))
        atomic_write(output / "doctor.json", canonical_json(doctor))
    print(canonical_json({"event": "gradient_audit_declared", "output": str(output)}).decode(), flush=True)


def check(p):
    require(p["maximum_optimizer_updates"] == 0 and p["candidate_lambda"] == LAMBDA and p["states"] == list(STATES), "Audit scope changed")
    for item in [p["parent_protocol"], p["parent_summary"], p["config"], p["manifest"], *p["code"], *p["checkpoints"].values()]:
        require(file_sha256(item["path"]) == item["sha256"], "Bound file changed: " + item["path"])
    check_protocol(read(p["parent_protocol"]["path"]))


def group_name(name):
    return "scene" if name.startswith("scene_context.") else "output" if name.startswith("output_projection.") else "spatial"


def comparisons(vectors):
    pairs = [("original_rgb", "original_light"), ("original_app", "original_detail"),
        ("original_rgb", "ai_rgb"), ("original_light", "ai_light"), ("original_app", "ai_app"),
        ("original_app", "ai_detail"), ("original_detail", "ai_detail"), ("ai_app", "ai_detail"),
        ("original_app", "candidate"), ("ai_detail", "candidate"), ("ai_detail", "old_same_input_sum"),
        ("ai_detail", "candidate_same_input_sum"), ("original_detail", "candidate_same_input_sum")]
    pairs += [(n, n + "_component_sum") for n in ("original_app", "ai_app", "candidate", "old_same_input_sum", "candidate_same_input_sum")]
    return {a + "__" + b: gradient_pair_statistics(vectors[a], vectors[b])
            for a, b in pairs if a in vectors and b in vectors}


def add_derived(v):
    components = {"original_app": v["original_rgb"] + v["original_light"]}
    if "ai_rgb" in v:
        components["ai_app"] = v["ai_rgb"] + v["ai_light"]
        components["candidate"] = components["original_app"] + LAMBDA * v["ai_detail"]
        components["old_same_input_sum"] = components["original_app"] + components["ai_app"]
        components["candidate_same_input_sum"] = 2 * components["original_app"] + LAMBDA * v["ai_detail"]
    for name, value in components.items():
        v[name + "_component_sum"] = value
        v.setdefault(name, value)
    return v


def repeat_difference(first, second):
    result = gradient_pair_statistics(first, second)
    difference = (second.double() - first.double()).reshape(-1)
    return {**result, "max_absolute_difference": float(difference.abs().max()),
        "relative_l2_difference": float(difference.norm()) / result["first_norm"] if result["first_norm"] else None}


def audit_case(model, bridge, dataset, loss, case, p, output, state_name, budget, *, clean=False, repeat=False, repeat_reference=None):
    index = case["clean"] if clean else case["original"]
    sample = move_sample(dataset[index], "cuda")
    ai = None if clean else move_sample(dataset[case["ai"]], "cuda")
    if ai is not None:
        for key in ("x", "scene", "valid", "geometry", "original_hw", "z_input", "person_mask", "face_mask"):
            require(torch.equal(sample[key], ai[key]), "Pair changed common input or geometry: " + key)
    budget.check()
    zp, _ = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
    prediction, _ = restore_pixels(bridge, sample, zp, grad=True, strength=1.)
    original = loss(prediction, sample["y"], zp, sample["z_target"], sample["valid"], sample["person_mask"], sample["face_mask"])
    objectives = {"original_rgb": original["rgb"], "original_light": .2 * original["lighting_target"],
        "original_detail": detail_loss(prediction, sample["y"], sample["valid"], sample["person_mask"], sample["face_mask"]),
        "original_app": original["total"]}
    if ai is not None:
        terms = loss(prediction, ai["y"], zp, ai["z_target"], sample["valid"], sample["person_mask"], sample["face_mask"])
        objectives.update(ai_rgb=terms["rgb"], ai_light=.2 * terms["lighting_target"],
            ai_detail=detail_loss(prediction, ai["y"], sample["valid"], sample["person_mask"], sample["face_mask"]))
        objectives.update(ai_app=terms["total"], candidate=original["total"] + LAMBDA * objectives["ai_detail"],
            old_same_input_sum=original["total"] + terms["total"],
            candidate_same_input_sum=2 * original["total"] + LAMBDA * objectives["ai_detail"])
    names, parameters = zip(*model.named_parameters())
    vectors = {key: {} for key in ("pixels", "latent", "all_parameters", "output", "spatial", "scene")}
    x0, y0, x1, y1 = p["rois"][sample["view_id"]]
    def roi(t):
        return t[0, :, 0, y0:y1, x0:x1].detach().float().cpu().permute(1, 2, 0).numpy().copy()
    arrays = {"input": roi(sample["x"]), "original": roi(sample["y"]), "prediction": roi(prediction),
              "z_prediction": zp.detach().float().cpu().numpy().copy()}
    if ai is not None: arrays["ai"] = roi(ai["y"])
    scalar_losses = {name: float(value.detach()) for name, value in objectives.items()}
    # Each VJP starts from the same prediction. Retain the graph until the final objective.
    for j, (name, objective) in enumerate(objectives.items()):
        budget.check()
        gradients = torch.autograd.grad(objective * p["scale"], (prediction, zp, *parameters),
            retain_graph=j + 1 < len(objectives), allow_unused=False)
        require(all(torch.isfinite(g).all() for g in gradients), "Nonfinite scaled H3 gradient")
        unscaled = [g.detach().float().cpu() / p["scale"] for g in gradients]
        vectors["pixels"][name] = unscaled[0].reshape(-1)
        vectors["latent"][name] = unscaled[1].reshape(-1)
        vectors["all_parameters"][name] = torch.cat([g.reshape(-1) for g in unscaled[2:]])
        for group in ("output", "spatial", "scene"):
            vectors[group][name] = torch.cat([g.reshape(-1) for n, g in zip(names, unscaled[2:]) if group_name(n) == group])
        arrays[name + "_pixel_gradient"] = roi(unscaled[0])
        arrays[name + "_latent_gradient"] = unscaled[1].numpy().copy()
        del gradients, unscaled
    for group in vectors: add_derived(vectors[group])
    kind = "clean" if clean else "paired"
    label = f"{state_name}_{kind}_{case['original']:02d}" + ("_repeat" if repeat else "")
    path = output / (label + ".npz")
    np.savez_compressed(path, **arrays)
    pixel_frequency = {}
    for name in objectives:
        coeff = dctn(arrays[name + "_pixel_gradient"].astype(np.float64))
        energy = {b: float(np.sum(coeff[mask] ** 2)) for b, mask in band_masks(128).items()}
        total = sum(energy.values())
        pixel_frequency[name] = {"energy": energy, "fraction": {b: e / total if total else None for b, e in energy.items()}}
    record = {"label": label, "state": state_name, "kind": kind, "repeat": repeat,
        "original_index": index, "ai_index": None if clean else case["ai"], "view_id": sample["view_id"],
        "source": dataset.sources[sample["asset_id"]]["path"], "roi_xyxy": [x0,y0,x1,y1],
        "objectives": scalar_losses, "identity_output": bool(torch.equal(prediction.detach(), sample["x"])),
        "gradients": {group: {"norms": {n: float(v.double().norm()) for n, v in values.items()},
            "pairs": comparisons(values), "tensor_sha256": {n: state_hash(v) for n, v in values.items()}}
            for group, values in vectors.items()},
        "rgb_roi_gradient_frequency": pixel_frequency, "float_arrays": bound(path),
        "fixed_frequency": {target: frequency_metrics(arrays["input"], arrays[target], arrays["prediction"])
                            for target in ("original", "ai") if target in arrays}}
    if repeat_reference is not None:
        if repeat:
            record["repeat_difference"] = {"reference_label": repeat_reference["label"],
                "prediction": repeat_difference(repeat_reference["prediction"], prediction.detach().float().cpu()),
                "z_prediction": repeat_difference(repeat_reference["zp"], zp.detach().float().cpu()),
                "gradient_spaces": {group: {name: repeat_difference(repeat_reference["vectors"][group][name], value)
                    for name, value in items.items()} for group, items in vectors.items()}}
        else:
            repeat_reference.update(label=label, prediction=prediction.detach().float().cpu(),
                zp=zp.detach().float().cpu(), vectors=vectors)
    require(not any(v.grad is not None for v in parameters), "Audit accumulated parameter .grad")
    atomic_write(output / (label + ".json"), canonical_json(record))
    print(canonical_json({"event": "supervision_case_audited", "label": label,
        "original_ai_latent_cosine": None if clean else record["gradients"]["latent"]["pairs"]["original_app__ai_app"]["cosine"]}).decode(), flush=True)
    return record


def run(output):
    p = read(output / "protocol.json"); check(p)
    require(not (output / "results.json").exists(), "Do not overwrite or implicitly resume an audit")
    config = load_config(p["config"]["path"])
    started = time.monotonic()
    report = {"status": "incomplete", "protocol": bound(output / "protocol.json"), "cases": [],
        "optimizer_updates": 0, "trained_base_accepted": False, "independent_validation": False,
        "guard": {"optimizer_constructions": 0, "backward_calls": 0, "autograd_grad_calls": 0}}
    atomic_write(output / "results.json", canonical_json(report))
    try:
        with TrainingBudget(ROOT / "runs", config.project.budget_seconds, phase="supervision_gradient_audit") as budget:
            dataset = open_dataset(config, ROOT, Path(p["manifest"]["path"]))
            dataset.audit(budget_check=budget.check)
            require(cases_for(dataset, make_slots(dataset)) == p["cases"], "Paired and clean mapping changed")
            with no_training_guard():
                seed_model(42)
                bridge = load_bridge(config, ROOT, dataset)
                model = SpatialRefinerV2.from_config(config.model).to("cuda").eval()
                loss = application_loss(config, ROOT)
            frozen = {n: (id(v), v._version) for n, v in bridge.backend.model.named_parameters()}
            require(not any(v.requires_grad for v in bridge.backend.model.parameters()), "H3 must be frozen")
            torch.cuda.reset_peak_memory_stats()
            old = read(p["parent_protocol"]["path"])
            with gradient_only_guard() as guard:
                report["guard"] = guard
                for state_name in STATES:
                    state, cp = load_saved(state_name, old)
                    budget.validate_resume_snapshot(state["budget"])
                    require(file_sha256(cp) == p["checkpoints"][state_name]["sha256"], "Checkpoint changed")
                    model.load_state_dict(state["model"], strict=True)
                    versions = {n: (id(v), v._version) for n, v in model.named_parameters()}
                    del state
                    rng = capture_rng_state()
                    repeated = {}
                    for case in p["cases"]:
                        for clean in (False, True):
                            keep = repeated if state_name == "control_76" and case == p["cases"][0] and not clean else None
                            record = audit_case(model, bridge, dataset, loss, case, p, output, state_name, budget, clean=clean, repeat_reference=keep)
                            if state_name == "initial": require(record["identity_output"], "Initial output must equal X")
                            report["cases"].append(record)
                            atomic_write(output / "results.json", canonical_json(report))
                    if state_name == "control_76":
                        restore_rng_state(rng)
                        report["cases"].append(audit_case(model, bridge, dataset, loss, p["cases"][0], p, output, state_name, budget, repeat=True, repeat_reference=repeated))
                    del repeated
                    restore_rng_state(rng)
                    require(versions == {n: (id(v), v._version) for n, v in model.named_parameters()}, "Audit changed R parameters")
                    actual = {n: v.detach().cpu() for n, v in model.state_dict().items()}
                    require(state_hash(actual) == p["checkpoints"][state_name]["model_sha256"], "R state changed")
                    del actual
                require(frozen == {n: (id(v), v._version) for n, v in bridge.backend.model.named_parameters()}, "Audit changed H3")
                require(not any(v.requires_grad or v.grad is not None for v in bridge.backend.model.parameters()), "H3 acquired gradients")
            require(len(report["cases"]) == p["expected_forward_graphs"] and guard["autograd_grad_calls"] == p["expected_autograd_grad_calls"], "Incomplete fixed gradient audit")
            torch.cuda.synchronize(); check(p)
            report.update(status="completed_real_h3_supervision_gradient_audit", guard=dict(guard),
                elapsed_seconds=time.monotonic()-started, h3_frozen_unchanged=True, model_states_unchanged=True,
                runtime=execution_contract(), peak_allocated_bytes=torch.cuda.max_memory_allocated(), budget=budget.snapshot(),
                parameter_groups={group: [n for n, _ in model.named_parameters() if group_name(n) == group]
                                  for group in ("output", "spatial", "scene")})
            atomic_write(output / "results.json", canonical_json(report))
    except BaseException as exc:
        report.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic()-started)
        atomic_write(output / "results.json", canonical_json(report)); raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("declare", "run"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    declare(output) if args.action == "declare" else run(output)
