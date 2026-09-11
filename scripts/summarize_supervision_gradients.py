"""CPU-only verification and presentation of the completed supervision audit.

This does not load models, call CUDA, calculate new gradients, or train. Full
latent gradients and RGB ROI gradients are independently checked from saved NPZ
files. Full parameter-vector repeat statistics are runtime evidence, not an
independent reconstruction: those full vectors were intentionally not saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from h3ce.cache.keys import canonical_json, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.config import load_config
from h3ce.train.checkpoint import TrainingBudget
from h3ce.train.guard import no_training_guard
from h3ce.train.preflight import require
from scripts.diagnose_supervision_gradients import (
    LAMBDA, STATES, add_derived, check, comparisons, repeat_difference,
)
from scripts.diagnose_detail_frequency import dctn, band_masks, frequency_metrics
from scripts.gpt_clarity_probe_common import read, bound
from scripts.verify_loss_balance_initialization import state_hash

SPACES = ("pixels", "latent", "all_parameters", "output", "spatial", "scene")
DIRECT = ("original_app", "ai_app", "candidate", "old_same_input_sum", "candidate_same_input_sum")
BASE_OBJECTIVES = {"original_rgb", "original_light", "original_detail", "original_app"}
PAIRED_OBJECTIVES = BASE_OBJECTIVES | {"ai_rgb", "ai_light", "ai_detail", "ai_app", "candidate",
                                      "old_same_input_sum", "candidate_same_input_sum"}


def near(actual, expected, label, *, rtol=1e-8, atol=1e-18):
    """Strict recursive shape/key checks plus double-precision numeric tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": keys differ")
        for key in expected:
            near(actual[key], expected[key], label + "/" + str(key), rtol=rtol, atol=atol)
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected, label + ": value differs")
    else:
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ": numeric mismatch")


def finite_tree(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, label + "/" + str(key))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            finite_tree(child, label + "/" + str(i))
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        require(math.isfinite(value), label + ": nonfinite value")


def validate_scope(protocol, report):
    require(protocol["status"] == "declared_before_gradient_execution"
            and protocol["expected_forward_graphs"] == 19
            and protocol["expected_autograd_grad_calls"] == 146
            and protocol["maximum_optimizer_updates"] == 0
            and protocol["states"] == list(STATES) and len(protocol["cases"]) == 3,
            "Unexpected declared gradient scope")
    require(report["status"] == "completed_real_h3_supervision_gradient_audit"
            and report["optimizer_updates"] == 0 and report["h3_frozen_unchanged"] is True
            and report["model_states_unchanged"] is True and report["trained_base_accepted"] is False
            and report["independent_validation"] is False, "Only a completed unchanged-model audit can be summarized")
    require(report["guard"] == {"optimizer_constructions": 0, "backward_calls": 0,
                                "autograd_grad_calls": 146}, "Execution counts differ from bounded scope")
    records = report["cases"]
    require(len(records) == 19 and len({c["label"] for c in records}) == 19, "Missing or duplicate audit cases")
    expected = {}
    for state in STATES:
        for case in protocol["cases"]:
            for clean in (False, True):
                kind = "clean" if clean else "paired"
                label = f"{state}_{kind}_{case['original']:02d}"
                expected[label] = (state, kind, False, case["clean"] if clean else case["original"],
                                   None if clean else case["ai"])
    first = protocol["cases"][0]
    expected[f"control_76_paired_{first['original']:02d}_repeat"] = (
        "control_76", "paired", True, first["original"], first["ai"])
    require({c["label"] for c in records} == set(expected), "Unexpected source/state/repeat coverage")
    for record in records:
        require(tuple(record[k] for k in ("state", "kind", "repeat", "original_index", "ai_index"))
                == expected[record["label"]], "Case metadata differs from its declared source/state")
        if record["state"] == "initial":
            require(record["identity_output"] is True, "Initial model must be identity")
            for group in ("scene", "spatial"):
                require(all(n == 0 for n in record["gradients"][group]["norms"].values()),
                        "Zero-output initialization should have zero upstream parameter gradients")
    require(sum(len(c["objectives"]) for c in records) == 146, "Saved objectives do not account for every VJP")
    return dict(Counter(c["state"] for c in records))


def pixel_bands(array):
    coefficient = dctn(array.astype(np.float64))
    energy = {b: float(np.sum(coefficient[mask] ** 2)) for b, mask in band_masks(array.shape[0]).items()}
    total = sum(energy.values())
    require(np.isclose(total, np.sum(array.astype(np.float64) ** 2), rtol=1e-10, atol=1e-20),
            "Pixel-gradient DCT violates Parseval")
    return {"energy": energy, "fraction": {b: e / total if total else None for b, e in energy.items()}}


def verify_case(folder, record, protocol):
    label = record["label"]
    path = folder / (label + ".json")
    assert_no_links(path)
    require(read(path) == record, "Case sidecar and completed results disagree: " + label)
    finite_tree(record, label)
    objective_names = BASE_OBJECTIVES if record["kind"] == "clean" else PAIRED_OBJECTIVES
    require(set(record["objectives"]) == objective_names, "Unexpected objective coverage: " + label)
    require(set(record["gradients"]) == set(SPACES), "Unexpected gradient spaces")
    roi = protocol["rois"][record["view_id"]]
    require(roi == record["roi_xyxy"] and roi[2] - roi[0] == roi[3] - roi[1] == 128,
            "Fixed ROI changed")
    float_path = Path(record["float_arrays"]["path"]).resolve()
    assert_no_links(float_path)
    require(float_path == (folder / (label + ".npz")).resolve()
            and file_sha256(float_path) == record["float_arrays"]["sha256"], "Float arrays changed: " + label)
    with np.load(float_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = {"input", "original", "prediction", "z_prediction"}
    if record["kind"] == "paired":
        expected.add("ai")
    expected |= {n + suffix for n in objective_names for suffix in ("_pixel_gradient", "_latent_gradient")}
    require(set(arrays) == expected, "Missing or unexpected NPZ arrays")
    zp = arrays["z_prediction"]
    require(zp.ndim == 5 and zp.shape[:3] == (1, 24, 1) and min(zp.shape) > 0, "Invalid saved latent geometry")
    for name, value in arrays.items():
        require(value.dtype == np.float32 and np.isfinite(value).all(), "Invalid float32 array: " + name)
        shape = zp.shape if name.endswith("_latent_gradient") or name == "z_prediction" else (128, 128, 3)
        require(value.shape == shape, "Invalid saved array geometry: " + name)
    vectors = add_derived({name: torch.from_numpy(arrays[name + "_latent_gradient"]).reshape(-1)
                           for name in objective_names})
    latent = record["gradients"]["latent"]
    near({name: float(v.double().norm()) for name, v in vectors.items()}, latent["norms"], label + "/latent/norms")
    near(comparisons(vectors), latent["pairs"], label + "/latent/pairs")
    require({name: state_hash(v) for name, v in vectors.items()} == latent["tensor_sha256"],
            "Saved latent gradients or component sums changed")
    near({name: pixel_bands(arrays[name + "_pixel_gradient"]) for name in objective_names},
         record["rgb_roi_gradient_frequency"], label + "/pixel_bands")
    near({target: frequency_metrics(arrays["input"], arrays[target], arrays["prediction"])
          for target in ("original", "ai") if target in arrays}, record["fixed_frequency"], label + "/fixed_frequency")
    return arrays, vectors


def difference_from_pair(pair):
    """Only a derived estimate; tiny differences can be below dot-product rounding."""
    squared = max(0., pair["first_norm"] ** 2 + pair["second_norm"] ** 2 - 2 * pair["dot"])
    return {"cosine": pair["cosine"], "relative_l2_from_norms_and_dot":
            math.sqrt(squared) / pair["first_norm"] if pair["first_norm"] else None}


def verify_repeat(first_record, second_record, first_arrays, second_arrays, first_vectors, second_vectors):
    runtime = second_record["repeat_difference"]
    require(runtime["reference_label"] == first_record["label"], "Wrong repeat reference")
    near(repeat_difference(torch.from_numpy(first_arrays["z_prediction"]), torch.from_numpy(second_arrays["z_prediction"])),
         runtime["z_prediction"], "repeat/z_prediction")
    latent = {name: repeat_difference(first_vectors[name], second_vectors[name]) for name in first_vectors}
    near(latent, runtime["gradient_spaces"]["latent"], "repeat/latent_gradients")
    roi = repeat_difference(torch.from_numpy(first_arrays["prediction"]), torch.from_numpy(second_arrays["prediction"]))
    require(roi["max_absolute_difference"] <= runtime["prediction"]["max_absolute_difference"] + 1e-12,
            "ROI repeat difference exceeds recorded full prediction difference")
    pixel_roi = {name: repeat_difference(torch.from_numpy(first_arrays[name + "_pixel_gradient"]),
                                        torch.from_numpy(second_arrays[name + "_pixel_gradient"]))
                 for name in first_record["objectives"]}
    for space, items in runtime["gradient_spaces"].items():
        require(set(items) == set(first_record["gradients"][space]["norms"]), "Repeat objective coverage changed")
        for name, pair in items.items():
            near(pair["first_norm"], first_record["gradients"][space]["norms"][name], "repeat/first_norm")
            near(pair["second_norm"], second_record["gradients"][space]["norms"][name], "repeat/second_norm")
            require(pair["max_absolute_difference"] >= 0 and (pair["cosine"] is None or -1 <= pair["cosine"] <= 1),
                    "Invalid runtime repeat statistics")
    return {"reference_label": first_record["label"], "repeat_label": second_record["label"],
            "recomputed_prediction_roi": roi, "recomputed_z_prediction": runtime["z_prediction"],
            "recomputed_latent_gradients": latent, "recomputed_pixel_roi_gradients": pixel_roi,
            "full_prediction_and_parameter_statistics_from_runtime": runtime,
            "full_parameter_vectors_saved": False,
            "limitation": "One repeated grad-enabled case is a local numerical check, not a global statistical noise bound. Full parameter-vector and full-pixel repeat differences cannot be independently recomputed from ROI/latent NPZ files."}


def source_id(source):
    match = re.search(r"_([0-9]{4})_", Path(source).stem)
    return match.group(1) if match else Path(source).stem


def csv_rows(records):
    rows = []
    for record in records:
        if record["repeat"]:
            continue
        for space in SPACES:
            stats = record["gradients"][space]
            norm = stats["norms"]["original_app"]
            paired = record["kind"] == "paired"
            def cosine(pair):
                return stats["pairs"][pair]["cosine"] if paired else None
            rows.append({"label": record["label"], "source": source_id(record["source"]),
                "state": record["state"], "kind": record["kind"], "space": space,
                "original_app_norm": norm, "ai_app_norm": stats["norms"].get("ai_app"),
                "ai_detail_norm": stats["norms"].get("ai_detail"),
                "original_ai_app_cosine": cosine("original_app__ai_app"),
                "ai_detail_old_sum_cosine": cosine("ai_detail__old_same_input_sum"),
                "ai_detail_candidate_sum_cosine": cosine("ai_detail__candidate_same_input_sum"),
                "weighted_ai_detail_to_original_ratio": LAMBDA * stats["norms"]["ai_detail"] / norm if paired and norm else None,
                "zero_upstream_initialization": record["state"] == "initial" and space in ("spatial", "scene"),
                "source_path": record["source"]})
    return rows


def plot(rows, destination):
    selected = [r for r in rows if r["kind"] == "paired" and r["space"] == "latent"]
    require(len(selected) == 9, "Plot requires all three sources at all three model states")
    labels = [r["source"] + "\n" + {"initial": "initial", "control_76": "control 76", "mixed_76": "mixed 76"}[r["state"]] for r in selected]
    positions = np.arange(9)
    colors = [{"initial": "#87949f", "control_76": "#3676ae", "mixed_76": "#d66b34"}[r["state"]] for r in selected]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4), constrained_layout=True)
    fig.suptitle("Supervision gradients at saved model states — no new training", fontsize=16)
    axes[0].bar(positions, [r["original_ai_app_cosine"] for r in selected], color=colors)
    axes[0].set_title("Original app vs AI app\ncosine in full latent-gradient space")
    axes[1].plot(positions, [r["ai_detail_old_sum_cosine"] for r in selected], "o-", color="#9b5550", label="Original app + AI app")
    axes[1].plot(positions, [r["ai_detail_candidate_sum_cosine"] for r in selected], "s-", color="#287e70", label="2 × original app + 0.05 × AI detail")
    axes[1].set_title("AI detail vs same-input summed supervision\ncosine in full latent-gradient space")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[2].bar(positions, [r["weighted_ai_detail_to_original_ratio"] for r in selected], color=colors)
    axes[2].set_title("0.05 × AI-detail norm / original-app norm\nfull latent gradients; original loss units")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for i, ax in enumerate(axes):
        ax.set_xticks(positions, labels, fontsize=8)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        if i < 2:
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="#404040", lw=.7)
        for boundary in (2.5, 5.5):
            ax.axvline(boundary, color="#cccccc", lw=.7)
    fig.supxlabel("Initial R has zero spatial/scene parameter gradients: their cosines are undefined. Latent gradients above remain measurable.\nPositive alignment is local evidence only; these panels do not show recovery quality or convergence.", fontsize=10)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def summarize(folder):
    folder = Path(folder).resolve()
    assert_no_links(folder)
    require(folder.is_relative_to(ROOT / "logs"), "Audit must be inside project logs")
    destination = folder / "summary"
    require(not destination.exists(), "Existing summary must not be overwritten")
    protocol = read(folder / "protocol.json")
    report = read(folder / "results.json")
    counts = validate_scope(protocol, report)  # Refuse an active GPU audit before taking its shared budget lock.
    config = load_config(protocol["config"]["path"])
    with no_training_guard() as guard, TrainingBudget(ROOT / "runs", config.project.budget_seconds,
                                                     phase="supervision_gradient_summary") as budget:
        check(protocol)
        require(report["protocol"] == bound(folder / "protocol.json"), "Results/protocol binding changed")
        budget.validate_resume_snapshot(report["budget"])
        input_bindings = [bound(folder / "protocol.json"), bound(folder / "results.json")]
        repeat_data = {}
        differences = []
        verified = []
        for record in report["cases"]:
            budget.check()
            arrays, latent = verify_case(folder, record, protocol)
            verified.append({"case": bound(folder / (record["label"] + ".json")), "arrays": record["float_arrays"]})
            for space in SPACES:
                for name in DIRECT:
                    key = name + "__" + name + "_component_sum"
                    if key not in record["gradients"][space]["pairs"]:
                        continue
                    value = difference_from_pair(record["gradients"][space]["pairs"][key])
                    if space == "latent":
                        value["recomputed_from_full_saved_vectors"] = repeat_difference(latent[name], latent[name + "_component_sum"])
                    differences.append({"case": record["label"], "state": record["state"], "space": space,
                                        "objective": name, **value})
            if record["state"] == "control_76" and record["kind"] == "paired" and record["original_index"] == protocol["cases"][0]["original"]:
                repeat_data[bool(record["repeat"])] = (record, arrays, latent)
        require(set(repeat_data) == {False, True}, "Missing repeat baseline")
        first, second = repeat_data[False], repeat_data[True]
        repeat = verify_repeat(first[0], second[0], first[1], second[1], first[2], second[2])
        rows = csv_rows(report["cases"])
        check(protocol)
        require(all(file_sha256(item["path"]) == item["sha256"] for item in input_bindings), "Audit changed during summary")
        destination.mkdir()
        csv_path = destination / "gradient_metrics.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        figure = destination / "gradient_directions.png"
        plot(rows, figure)
        result = {"status": "verified_completed_supervision_gradient_audit", "inputs": input_bindings,
            "script": bound(Path(__file__)), "verified_case_files": verified, "state_counts": counts,
            "forward_graphs_in_source_audit": 19, "autograd_grad_calls_in_source_audit": 146,
            "new_optimizer_updates": 0, "new_forward_graphs": 0, "new_gradient_calls": 0,
            "execution_guard": dict(guard), "h3_frozen_unchanged_in_source_audit": True,
            "model_states_unchanged_in_source_audit": True, "trained_base_accepted": False,
            "independent_validation": False, "rows": rows, "direct_vs_component_sums": differences,
            "repeat_verification": repeat, "artifacts": [bound(csv_path), bound(figure)], "budget": budget.snapshot(),
            "verification": {"latent_norms_pairs_and_hashes": "recomputed from full float32 NPZ gradients; float64 reductions",
                "rgb_roi_gradient_frequency": "recomputed orthonormal spatial DCT and Parseval check",
                "full_parameter_statistics": "recorded by guarded GPU audit, not independently reconstructible from saved NPZ",
                "relative_l2_from_norms_and_dot": "derived estimate; cancellation limits very small values",
                "undefined_cosines": "null when either gradient is zero; initial spatial and scene zero is expected"},
            "limitations": ["No training, image quality improvement, or useful base acceptance is established by gradient alignment",
                "Three sources share one episode; the nine selected views do not represent all 19-slot gradient contributions",
                "0.05 is the declared diagnostic candidate coefficient, not a fitted or validated training weight",
                "RGB ROI frequency is not latent feature frequency; one repeated case is not a universal numerical error bound"]}
        atomic_write(destination / "summary.json", canonical_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.audit))
    print(canonical_json({"status": result["status"], "summary": str(Path(args.audit) / "summary/summary.json"),
                          "new_optimizer_updates": 0}).decode())
