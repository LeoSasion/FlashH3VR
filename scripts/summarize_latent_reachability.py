"""Verify completed per-sample latent optimization using CPU evidence only.

ROI errors are independently reconstructed from unclamped float32 NPZs. Full
image metrics are the saved native-H3 measurements: the full images were not
saved and cannot be reconstructed here. This never evaluates a model or trains.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.config import load_config
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget
from h3ce.train.engine import implementation_hashes
from h3ce.train.guard import no_training_guard
from h3ce.train.precision import POLICY
from h3ce.train.preflight import require
from h3ce.train.sampler import StatefulSampler
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.evaluate_gpt_clarity_probe import edge_mse
from scripts.gpt_clarity_probe_common import bound
from scripts.plot_training_losses import contributions, plot as legacy_plot, read_records
from scripts.probe_latent_reachability import EVALUATIONS, KIND, LR, STEPS, check
from scripts.summarize_supervision_gradients import finite_tree, near, source_id
from scripts.verify_loss_balance_initialization import assert_equal, check_optimizer, read_json, state_hash

ORIGINALS = (0, 5, 14)
GUARD_ZERO = {"optimizer_constructions": 0, "backward_calls": 0, "autograd_grad_calls": 0}
TITLE = "H3CE | target-assisted per-sample latent optimization; no R/base training"


def checked_path(value, parent=None):
    path = Path(value).resolve()
    assert_no_links(path)
    require(path.is_relative_to(ROOT) and path != ROOT, "Evidence must remain inside the project")
    if parent is not None:
        require(path.is_relative_to(parent.resolve()), "Evidence belongs to a different run")
    return path


def binding(item, expected=None):
    require(set(item) == {"path", "sha256"}, "Invalid evidence binding")
    path = checked_path(item["path"])
    require(expected is None or path == expected.resolve(), "Evidence path differs from its declared role")
    require(file_sha256(path) == item["sha256"], "Evidence SHA256 changed: " + str(path))
    return path


def validate_scope(p, reports):
    require(p["kind"] == KIND and p["status"] == "declared_before_optimizer_updates"
            and p["steps_per_case"] == 32 and p["maximum_optimizer_updates"] == 96
            and p["learning_rate"] == LR and p["accumulation"] == 1
            and p["evaluation_steps"] == list(EVALUATIONS), "Unexpected diagnostic scope")
    require([c["index"] for c in p["cases"]] == list(ORIGINALS)
            and len({c["view_id"] for c in p["cases"]}) == 3
            and len({c["source"] for c in p["cases"]}) == 3
            and len({c["run"] for c in p["cases"]}) == 3, "Expected three distinct original cases")
    for flag in ("refiner_loaded", "h3_trainable", "independent_validation", "trained_base_accepted",
                 "deployable", "automatic_extension"):
        require(p[flag] is False, "Diagnostic scope flag changed: " + flag)
    criterion = p["pilot_criterion"]
    require(criterion["high_mse_relative_to_input_max"] == -.005
            and criterion["edge_mse_relative_to_input_max"] == -.005
            and criterion["global_mae_relative_to_input_max"] == 0.
            and criterion["case_count_min"] == 2, "Prespecified numeric criterion changed")
    require(len(reports) == 3, "All three completed reports are required before summary")
    for case, report in zip(p["cases"], reports):
        finite_tree(report, case["name"])
        require(case["name"] == f"original_{case['index']:02d}" and report["case"] == case
                and report["status"] == "completed_target_assisted_sample_latent_diagnostic"
                and report["kind"] == KIND and report["phase"] == "pixel"
                and report["optimizer_steps"] == 32 and report["h3_frozen_unchanged"] is True,
                "Incomplete or mismatched diagnostic report")
        for flag in ("refiner_loaded", "deployable", "trained_base_accepted", "independent_validation"):
            require(report[flag] is False, "Completed report scope changed: " + flag)
        require(report["extra"]["phase_complete"] is True
                and report["extra"]["successful_case_visits"] == 32, "Missing committed completion")


def validate_delta(model, shape, *, initial=False):
    require(isinstance(model, dict) and set(model) == {"delta_latent"}, "Only one delta parameter is allowed")
    delta = model["delta_latent"]
    require(isinstance(delta, torch.Tensor) and delta.device.type == "cpu" and delta.dtype == torch.float32
            and list(delta.shape) == list(shape) and delta.ndim == 5 and tuple(delta.shape[:3]) == (1, 24, 1)
            and min(delta.shape) > 0 and torch.isfinite(delta).all().item(), "Invalid saved latent parameter")
    if initial:
        require(torch.count_nonzero(delta).item() == 0, "Initial delta must be exactly zero")
    return delta


def replay_sampler(saved, steps):
    sampler = StatefulSampler(1, seed=42, shuffle=False)
    for _ in range(steps):
        sampler.begin_window()
        require(sampler.next_index() == 0, "Unexpected single-view sampler index")
        sampler.commit_window()
    assert_equal(sampler.state_dict(), saved, "unshuffled singleton sampler endpoint")
    return {"shuffle": False, "view_count": 1, "committed_visits": steps,
            "sample_indices_sha256": digest([0] * steps), "state_sha256": state_hash(saved)}


def validate_records(rows, config, case, asset_id):
    require([r["step"] for r in rows] == list(range(1, STEPS + 1)), "Expected all 32 successful update records")
    weights, values = contributions(rows, config)
    require(weights == {"rgb": 1., "latent": 0., "perceptual": 0., "lighting_target": .2},
            "Application objective changed")
    for row in rows:
        finite_tree(row, "training record")
        require(row["phase"] == "pixel" and row["diagnostic_kind"] == KIND and row["case"] == case["name"]
                and row["optimizer_updated"] is True and row["learning_rate"] == LR
                and len(row["samples"]) == 1, "Unexpected optimizer record")
        sample = row["samples"][0]
        require(sample["slot"] == 0 and sample["view_id"] == case["view_id"] and sample["asset_id"] == asset_id
                and sample["clean_pair"] is False and sample["supervision_group"] == "original_degraded"
                and sample["optimization_variable"] == "sample_delta_latent", "Wrong sample was optimized")
        near(row["losses"], sample["losses"], "single-sample loss equals update loss")
        require(set(row["gradient_norms"]) == {"sample_latent"}
                and row["gradient_norms"]["sample_latent"] > 0
                and row["gradient_norm_before_clip"] > 0
                and row["loss_scale_before"] > 0 and row["loss_scale_after"] > 0,
                "Missing scaled single-parameter gradient evidence")
        near(row["gradient_norms"]["sample_latent"], row["gradient_norm_before_clip"],
             "single-parameter global L2", rtol=2e-6, atol=1e-12)
    return weights, values


def verify_measure(value, config, label):
    require(set(value) == {"metrics", "element_weights", "counts"}, "Invalid recorded measure: " + label)
    finite_tree(value, label)
    m = value["metrics"]
    require(m["rgb_global_mae"] >= 0 and value["counts"]["valid_rgb_elements"] > 0
            and value["counts"]["valid_latent_elements"] > 0, "Invalid global metric denominator")
    contributions([{"phase": "pixel", "losses": m}], config)


def load_roi(item, path, *, oracle):
    binding(item, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    require(set(arrays) == ({"input", "target", "prediction", "oracle"} if oracle
                            else {"input", "target", "prediction"}), "Unexpected ROI array coverage")
    for name, array in arrays.items():
        require(array.dtype == np.float32 and array.shape == (128, 128, 3) and np.isfinite(array).all(),
                "Invalid float32 ROI: " + name)
    return arrays


def relative_change(value, baseline):
    require(np.isfinite(value) and np.isfinite(baseline) and value >= 0 and baseline >= 0,
            "Invalid nonnegative error metric")
    return float(value / baseline - 1) if baseline > 0 else None


def pilot_criterion(initial, final, preset):
    changes = {
        "high_mse_relative_to_input": relative_change(final["high_mse"], initial["high_mse"]),
        "edge_mse_relative_to_input": relative_change(final["edge_mse"], initial["edge_mse"]),
        "global_mae_relative_to_input": relative_change(final["global_mae"], initial["global_mae"]),
    }
    checks = {name: value is not None and value <= preset[name + "_max"] for name, value in changes.items()}
    return {"changes": changes, "checks": checks, "numerical_pass": all(checks.values()),
            "visual_pass": None, "acceptance": "pending_visual_review_not_a_trained_base"}


def verify_case(folder, p, case, report, config, view, budget):
    run = checked_path(case["run"], ROOT / "runs" / folder.name)
    require(run == ROOT / "runs" / folder.name / case["name"], "Unexpected case run directory")
    bindings = [bound(run / name) for name in ("training_report.json", "training_contract.json",
                                             "training_request.json", "initial_checkpoint.json", "resolved.yaml", "metrics.jsonl")]
    require(load_config(run / "resolved.yaml").model_dump(mode="json") == config, "Case configuration changed")
    require(read_json(run / "training_request.json") == {"protocol": str(folder / "protocol.json"), "case": case["name"]},
            "Unexpected case request")
    contract = read_json(run / "training_contract.json")
    require(contract["phase"] == "pixel" and contract["max_steps"] == STEPS
            and contract["gradient_precision_policy"] == POLICY
            and contract["manifest_sha256"] == file_sha256(p["manifest"])
            and contract["resolved_sha256"] == digest(config), "Case contract changed")
    require(contract["experiment"] == {"kind": KIND, "protocol_sha256": file_sha256(folder / "protocol.json"),
                "case": case, "code_sha256": p["code_sha256"], "refiner_loaded": False, "deployable": False},
            "Experiment binding changed")
    compatible = contract["compatibility"]
    require(compatible["architecture"] == {"kind": "free_per_sample_latent_tensor", "shape": case["latent_shape"]}
            and compatible["implementation_sha256"] == implementation_hashes(ROOT)
            and compatible["encoder_contract_id"] == view["encoder_contract_id"]
            and compatible["components_sha256"] == file_sha256(ROOT / config["paths"]["components_lock"])
            and compatible["data"] == config["data"], "Latent, encoder, or implementation contract changed")
    manager = CheckpointManager(ROOT / "runs", run, contract=contract)
    initial_path = binding(read_json(run / "initial_checkpoint.json"))
    final_path = checked_path(report["checkpoint"], run / "checkpoints")

    def saved(path, step, initial_optimizer=None):
        require(path.parent == run / "checkpoints", "Checkpoint belongs to another case")
        state = manager.read(path)
        require(state["step"] == step and state["stage"] == "pixel" and state["resolved_config"] == config,
                "Checkpoint boundary/configuration changed")
        validate_delta(state["model"], case["latent_shape"], initial=step == 0)
        replay_sampler(state["sampler"], step)
        require(state["extra"]["successful_case_visits"] == step and state["extra"]["kind"] == KIND
                and state["extra"]["deployable"] is False and state["scheduler"]["last_epoch"] == step,
                "Checkpoint is missing diagnostic state")
        require(isinstance(state["rng"], dict) and state["rng"]["cuda_initialized"] is True
                and len(state["rng"]["torch_cuda"]) > 0, "Missing native GPU RNG checkpoint")
        require(state["scaler"] and state["scaler"]["scale"] > 0, "Missing FP16 scaler checkpoint")
        budget.validate_resume_snapshot(state["budget"])
        if initial_optimizer is not None and step:
            check_optimizer(initial_optimizer, state["optimizer"], state["model"], step)
        bindings.extend((bound(path), bound(path.with_suffix(".json"))))
        return state

    initial = saved(initial_path, 0)
    require(initial["optimizer"]["state"] == {} and initial["scaler"] == {
        "scale": 65536., "growth_factor": 2., "backoff_factor": .5, "growth_interval": 2000, "_growth_tracker": 0},
        "Initial optimizer/scaler history must be fresh")
    groups = initial["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == [0] and groups[0]["lr"] == LR
            and groups[0]["weight_decay"] == 0., "Use one independent AdamW parameter without weight decay")
    final = saved(final_path, STEPS, initial["optimizer"])
    require(final["extra"] == report["extra"] and final["extra"]["phase_complete"] is True,
            "Report disagrees with the completed checkpoint")
    optimizer_check = check_optimizer(initial["optimizer"], final["optimizer"], final["model"], STEPS)
    rows, _ = read_records(run / "metrics.jsonl")
    weights, _ = validate_records(rows, config, case, view["asset_id"])
    require(rows[-1]["loss_scale_after"] == final["scaler"]["scale"], "Final recorded scaler differs")
    skipped_path = run / "skipped_updates.jsonl"
    skipped = [json.loads(line) for line in skipped_path.read_text().splitlines() if line.strip()] if skipped_path.exists() else []
    require(len(skipped) == final["extra"]["overflow_attempts"] and final["extra"]["consecutive_overflows"] == 0,
            "Overflow accounting disagrees with actual update count")
    for row in skipped:
        require(row["optimizer_updated"] is False and 0 <= row["completed_step"] < STEPS
                and row["loss_scale_after"] < row["loss_scale_before"], "Invalid skipped update")
    if skipped_path.exists():
        bindings.append(bound(skipped_path))
    fixed, stored_arrays = [], {}
    for step in EVALUATIONS:
        budget.check()
        directory = run / f"fixed_step_{step:04d}"
        result = read_json(directory / "metrics.json")
        bindings.append(bound(directory / "metrics.json"))
        finite_tree(result, "fixed evaluation")
        require(result["status"] == "completed_target_assisted_latent_evaluation" and result["step"] == step
                and result["view_id"] == case["view_id"] and result["guard"] == GUARD_ZERO
                and result["new_optimizer_updates"] == 0 and result["deployable"] is False,
                "Invalid fixed evaluation scope")
        state = saved(binding(result["checkpoint"]), step, initial["optimizer"])
        if step in (0, STEPS):
            assert_equal(state["model"], (initial if step == 0 else final)["model"], "Fixed/end-point delta")
        delta = state["model"]["delta_latent"]
        near(result["latent_delta_l2"], float(delta.double().norm()), "Saved delta L2", rtol=2e-5, atol=1e-8)
        near(result["latent_delta_absmax"], float(delta.abs().max()), "Saved delta absolute maximum", rtol=0, atol=0)
        arrays = load_roi(result["float_arrays"], directory / "roi.npz", oracle=step == 0)
        bindings.append(result["float_arrays"])
        if step == 0:
            require(np.array_equal(arrays["prediction"], arrays["input"]), "Step-zero output must be exactly X")
        else:
            require(all(np.array_equal(arrays[name], stored_arrays[0][name]) for name in ("input", "target")),
                    "Fixed source ROI changed between evaluation steps")
        frequency = frequency_metrics(arrays["input"], arrays["target"], arrays["prediction"])
        edges = {name: edge_mse(arrays[name], arrays["target"]) for name in ("input", "prediction")}
        near(result["frequency"], frequency, "Recomputed ROI DCT metrics")
        near(result["edge_mse"], edges, "Recomputed ROI edge errors")
        for label in ("input", "current"):
            verify_measure(result[label], config, label)
        if step == 0:
            near(result["current"], result["input"], "Identity full-frame evaluation", rtol=2e-6, atol=1e-9)
            verify_measure(result["oracle"], config, "oracle")
            near(result["oracle_frequency"], frequency_metrics(arrays["input"], arrays["target"], arrays["oracle"]),
                 "Recomputed oracle DCT metrics")
            near(result["oracle_edge_mse"], edge_mse(arrays["oracle"], arrays["target"]), "Recomputed oracle edge error")
        else:
            near(result["input"], fixed[0]["input"], "Repeated full-frame input baseline", rtol=2e-6, atol=1e-9)
        for label in ("current", "input", "oracle"):
            if label in result:
                near(result[label]["counts"], result["input"]["counts"], "Fixed metric element counts", rtol=0, atol=0)
        budget.validate_resume_snapshot(result["budget"])
        fixed.append(result)
        stored_arrays[step] = arrays
        del state
    baseline = {"high_mse": fixed[0]["frequency"]["high"]["input_error_mse"],
                "edge_mse": fixed[0]["edge_mse"]["input"], "global_mae": fixed[0]["input"]["metrics"]["rgb_global_mae"]}
    metrics_rows = []
    for result in fixed:
        row = {"source": source_id(case["source"]), "case": case["name"], "original_index": case["index"],
            "step": result["step"], "global_mae": result["current"]["metrics"]["rgb_global_mae"],
            "high_mse": result["frequency"]["high"]["output_error_mse"], "edge_mse": result["edge_mse"]["prediction"],
            "application_total": result["current"]["metrics"]["total"], "delta_l2": result["latent_delta_l2"]}
        for name in ("global_mae", "high_mse", "edge_mse"):
            row[name + "_relative_to_input"] = relative_change(row[name], baseline[name])
        metrics_rows.append(row)
    oracle = {"global_mae": fixed[0]["oracle"]["metrics"]["rgb_global_mae"],
              "high_mse": fixed[0]["oracle_frequency"]["high"]["output_error_mse"], "edge_mse": fixed[0]["oracle_edge_mse"]}
    verification = {"case": case, "optimizer": optimizer_check, "source_optimizer_updates": STEPS,
        "initial_delta_all_zero": True, "parameter_count": 1, "initial_delta_sha256": state_hash(initial["model"]),
        "final_delta_sha256": state_hash(final["model"]), "final_delta_l2_cpu_float64": float(final["model"]["delta_latent"].double().norm()),
        "sampler": replay_sampler(final["sampler"], STEPS), "overflow_attempts": len(skipped), "effective_weights": weights,
        "h3_frozen_unchanged_reported": True, "refiner_loaded": False, "baseline": baseline, "oracle": oracle,
        "pilot": pilot_criterion(baseline, metrics_rows[-1], p["pilot_criterion"]), "fixed_metrics": metrics_rows,
        "evidence": bindings}
    return verification, stored_arrays, rows


def plot_loss_display(cases, records, config, output):
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
    for panels, case, rows in zip(axes, cases, records):
        steps = np.array([row["step"] for row in rows])
        weights, values = contributions(rows, config)

        def line(axis, values, label):
            handle, = axis.plot(steps, values, lw=.8, alpha=.45, label=label + " actual")
            axis.plot(steps[3:], np.convolve(values, np.ones(4) / 4, mode="valid"), color=handle.get_color(), lw=1.7,
                      label=label + " mean(4)")

        line(panels[0], np.array([r["losses"]["total"] for r in rows]), "Application total")
        for name, weight in weights.items():
            if weight:
                line(panels[1], values[name], f"{name} x {weight:g}")
        line(panels[2], np.array([r["losses"]["latent"] for r in rows]), "Raw latent; weight 0")
        for axis, label in zip(panels, ("Application objective", "Effective contributions", "Latent diagnostic only")):
            axis.set_title(f"{source_id(case['source'])} | {label}", fontsize=10)
            axis.set(xlabel="Successful optimizer update (loss measured before update)", ylabel="Loss")
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    figure.suptitle(TITLE + "\nOne unchanged original view per row; no shuffle; 32 updates per independent delta", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_rois(cases, arrays, output):
    figure, axes = plt.subplots(3, 4, figsize=(12, 9.5), squeeze=False)
    for panels, case, saved in zip(axes, cases, arrays):
        values = (saved[0]["target"], saved[0]["input"], saved[32]["prediction"], saved[0]["oracle"])
        for axis, value, title in zip(panels, values, ("TARGET (original O)", "INPUT (degraded O)", "OPT32 (sample delta)", "ORACLE (target latent)")):
            axis.imshow(np.clip(value, 0, 1), interpolation="nearest")
            axis.set_title(f"{source_id(case['source'])} | {title}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Fixed 128-pixel ROIs | per-sample target-assisted diagnosis; not R/base outputs\nDisplay clipped to [0,1] only; metrics use original float arrays. Oracle is not an optimum.", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_fixed_metrics(verified, output):
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), squeeze=False)
    for panels, case in zip(axes, verified):
        for axis, key, title in zip(panels, ("global_mae", "high_mse", "edge_mse"),
                                    ("Full-frame MAE (runtime measure)", "ROI high-frequency MSE", "ROI edge MSE")):
            rows = case["fixed_metrics"]
            axis.plot([r["step"] for r in rows], [r[key] for r in rows], "o-", lw=1.3, label="Saved fixed evaluation")
            axis.axhline(case["baseline"][key], color="gray", ls="--", label="Input baseline")
            axis.axhline(case["oracle"][key], color="#b5681c", ls=":", label="Target-latent oracle")
            axis.set(title=f"{source_id(case['case']['source'])} | {title}", xlabel="Completed per-sample updates", ylabel="Error (lower is better)")
            axis.set_xticks(EVALUATIONS)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    figure.suptitle(TITLE + "\nOnly 0/8/16/32 are measured; connecting lines add no observations. No independent validation.", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output, dpi=150)
    plt.close(figure)


def summarize(folder):
    folder = checked_path(folder, ROOT / "logs")
    destination = folder / "summary"
    require(not destination.exists(), "Existing summary directory must not be overwritten")
    p = read_json(folder / "protocol.json")
    reports = [read_json(checked_path(c["run"], ROOT / "runs") / "training_report.json") for c in p["cases"]]
    validate_scope(p, reports)  # Fail before acquiring the shared ledger while a case is still running.
    config_object = load_config(p["config"])
    config = config_object.model_dump(mode="json")
    require(config["training"]["gradient_accumulation"] == 1
            and config["training"]["stages"]["bootstrap_pixel"]["lr"] == LR
            and config["training"]["stages"]["bootstrap_pixel"]["max_steps"] == STEPS, "Optimizer configuration changed")
    with no_training_guard() as guard, TrainingBudget(ROOT / "runs", config_object.project.budget_seconds,
                                                     phase="latent_reachability_summary") as budget:
        check(p)
        manifest = checked_path(p["manifest"])
        raw = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        views = [r for r in raw if r["record_type"] == "training_view"]
        sources = {r["asset_id"]: r for r in raw if r["record_type"] == "source_asset"}
        variants = {r["variant_id"]: r for r in raw if r["record_type"] == "degraded_variant"}
        require(len(views) == 19 and p["data_audit"]["views"] == 19, "The full 19-view dataset must remain bound")
        verified, arrays, records = [], [], []
        inputs = [bound(folder / "protocol.json"), bound(Path(p["config"])), bound(manifest)]
        for case, report in zip(p["cases"], reports):
            budget.check()
            budget.validate_resume_snapshot(report["budget"])
            view = views[case["index"]]
            require(view["view_id"] == case["view_id"] and sources[view["asset_id"]]["path"] == case["source"]
                    and variants[view["variant_id"]]["clean_pair"] is False
                    and view["z_input_key"] == case["z_input_key"] and view["z_target_key"] == case["z_target_key"]
                    and case["roi"][2] - case["roi"][0] == case["roi"][3] - case["roi"][1] == 128,
                    "Source, original-input pairing, latent cache or ROI binding changed")
            result, saved_arrays, saved_records = verify_case(folder, p, case, report, config, view, budget)
            verified.append(result)
            arrays.append(saved_arrays)
            records.append(saved_records)
        check(p)
        all_bindings = inputs + [item for case in verified for item in case["evidence"]]
        require(all(file_sha256(item["path"]) == item["sha256"] for item in all_bindings), "Evidence changed during verification")
        destination.mkdir()
        rows = [row for case in verified for row in case["fixed_metrics"]]
        csv_path = destination / "fixed_metrics.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        engine_path = destination / "training_losses_engine.png"
        original_suptitle = Figure.suptitle

        def diagnostic_title(figure, text, *args, **kwargs):
            return original_suptitle(figure, TITLE + "; fixed single view, no shuffle", *args, **kwargs)

        # Keep the locked plotting engine's numerical curves; replace only its generic shuffled-view title.
        with patch.object(Figure, "suptitle", diagnostic_title):
            engine_metadata = legacy_plot([Path(case["run"]) for case in p["cases"]], engine_path, window=4)
        engine_metadata["generic_engine_notes"] = engine_metadata.pop("notes")
        engine_metadata["notes"] = ["Each run optimizes its own one fixed original view; shuffle=False.",
            "Step-k loss is measured before update k. The total is RGB + 0.2 lighting; latent weight is zero.",
            "This is target-assisted per-sample latent diagnosis, not training or accepting an R/base."]
        atomic_write(engine_path.with_suffix(".json"), canonical_json(engine_metadata))
        loss_path = destination / "latent_diagnostic_losses.png"
        roi_path = destination / "fixed_rois.png"
        metrics_path = destination / "fixed_metrics.png"
        plot_loss_display(p["cases"], records, config, loss_path)
        plot_rois(p["cases"], arrays, roi_path)
        plot_fixed_metrics(verified, metrics_path)
        count = sum(case["pilot"]["numerical_pass"] for case in verified)
        require(dict(guard) == GUARD_ZERO, "CPU summary unexpectedly attempted training")
        result = {"status": "verified_completed_target_assisted_sample_latent_diagnostic", "kind": KIND,
            "inputs": inputs, "summary_script": bound(Path(__file__)), "cases": verified,
            "source_optimizer_updates": 96, "source_optimizer_updates_per_case": 32,
            "source_fixed_evaluations": 12, "source_target_latent_oracles": 3,
            "new_optimizer_updates": 0, "new_forward_graphs": 0, "new_gradient_calls": 0,
            "execution_guard": dict(guard), "refiner_loaded": False, "h3_frozen_unchanged_reported": True,
            "trained_base_accepted": False, "deployable": False, "independent_validation": False,
            "pilot": {"preset": p["pilot_criterion"], "numerically_passing_cases": count,
                "numerical_gate_met": count >= p["pilot_criterion"]["case_count_min"], "visual_pass": None,
                "decision": "Numeric result only; final visual review and any next experiment remain separate"},
            "verification": {"checkpoint": "CPU receipt/SHA, single float32 delta, zero initialization, AdamW moments/steps, scheduler/scaler, sampler and RNG presence",
                "roi": "Unclamped float32 NPZ, SHA256, independently recomputed DCT-II frequency metrics and finite-difference edge errors",
                "global_metrics": "Saved native-H3 measure values bound to verified code, case and checkpoint; full images were not saved, so not independently recomputed",
                "latent_l2": "Full saved delta reduced in CPU float64; relative tolerance 2e-5 against native float32 L2",
                "loss": "Locked contributions() verifies actual step losses and fixed application totals; latent/perceptual coefficients remain zero"},
            "limitations": ["Three sample-specific target-assisted optimizers are not a deployable restoration function or a useful-base acceptance test.",
                "This tests bounded local optimization through the existing output path; it does not compare R parameter efficiency at equal budgets.",
                "A failed 32-update run cannot prove decoder incapacity; a successful one cannot prove an R can learn a shared mapping.",
                "Numerical gates need visual review for noise, halos, geometry and tonal drift; target-latent oracle is not an optimum.",
                "Fixed ROI errors are pixel-space metrics, not latent-frequency measurements; no missing evaluations are synthesized."],
            "artifacts": [bound(path) for path in (csv_path, engine_path, engine_path.with_suffix(".json"), loss_path, roi_path, metrics_path)],
            "budget": budget.snapshot()}
        check(p)
        require(all(file_sha256(item["path"]) == item["sha256"] for item in all_bindings), "Evidence changed during plotting")
        atomic_write(destination / "summary.json", canonical_json(result))
    print(canonical_json({"event": "latent_reachability_summary_verified", "path": str(destination / "summary.json"),
                         "source_optimizer_updates": 96, "new_optimizer_updates": 0}).decode(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed probe directory containing protocol.json")
    summarize(parser.parse_args().output)
