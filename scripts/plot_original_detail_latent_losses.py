"""Observe C2 original-detail sample-latent records, without starting any work.

The historical plot implementation stays unchanged. Scoped callbacks add the
original application subtotal and correct the fixed-sample diagnostic captions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
from h3ce.cache.keys import digest
from h3ce.config import load_config
from scripts import plot_training_losses as engine

KIND = "original_detail_sample_latent_probe"
DETAIL_WEIGHT = .5
STEPS_PER_CASE = 32
BASELINE_CONFIG = ROOT / "logs/latent-reachability-20260909-v1/resolved.yaml"
BASE_WEIGHTS = {"rgb": 1., "latent": 0., "perceptual": 0., "lighting_target": .2}
DETAIL_DEFINITION = {
    "domain": "sRGB working pixels", "sigma_reference": 2., "reference_short_edge": 512,
    "truncate": 3.,
    "mask": "valid crop interior eroded by Gaussian radius; original/face/person normalization",
    "epsilon": .001,
}
_ALLOWED_LOSSES = {*BASE_WEIGHTS, "rgb_person", "rgb_face", "application_total", "detail", "total"}
_PLOT_LOCK = threading.RLock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and np.isfinite(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    result = json.loads(Path(path).read_bytes())
    _require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def _absolute(value, description):
    _require(isinstance(value, str) and Path(value).is_absolute(), f"{description} must be an absolute path")
    return Path(value).resolve()


def contributions(rows, config):
    """Separate old application subtotal from the new objective; no double count."""
    _require(rows and len(rows) <= STEPS_PER_CASE and all(isinstance(row, dict) for row in rows),
             "Expected one bounded sample-latent run")
    _require(all(row.get("phase") == "pixel" and type(row.get("step")) is int for row in rows)
             and [row["step"] for row in rows] == list(range(1, len(rows) + 1)),
             "Expected contiguous pixel-stage optimizer records from step 1")
    try:
        configured = config["training"]["losses"]
        weights = {name: configured[name] for name in engine.COMPONENTS}
    except (KeyError, TypeError) as error:
        raise ValueError("Missing resolved application weights") from error
    _require(all(_number(value) and value == BASE_WEIGHTS[name] for name, value in weights.items()),
             "Original application weights must remain RGB=1, light=0.2, latent/perceptual=0")
    weights = {name: float(value) for name, value in weights.items()}
    weights["detail"] = DETAIL_WEIGHT
    for row in rows:
        losses = row.get("losses")
        _require(isinstance(losses, dict) and set(losses).issubset(_ALLOWED_LOSSES),
                 "Unknown or hidden loss field in sample-latent record")
        _require({*weights, "application_total", "total"}.issubset(losses), "Missing application_total/detail loss field")
        _require(all(_number(value) and value >= 0 for value in losses.values()), "Nonfinite or negative recorded loss")
    values = {name: np.asarray([row["losses"][name] for row in rows], dtype=float) * coefficient
              for name, coefficient in weights.items()}
    application = np.asarray([row["losses"]["application_total"] for row in rows], dtype=float)
    total = np.asarray([row["losses"]["total"] for row in rows], dtype=float)
    _require(np.allclose(sum(values[name] for name in engine.COMPONENTS), application, rtol=2e-6, atol=1e-9),
             "application_total does not match original RGB + 0.2 lighting")
    _require(np.allclose(application + values["detail"], total, rtol=2e-6, atol=1e-9),
             "total does not match application_total + 0.5 raw detail")
    return weights, values


def bound_evidence(run):
    run = Path(run).resolve()
    contract = _read(run / "training_contract.json")
    request = _read(run / "training_request.json")
    experiment = contract.get("experiment", {})
    _require(isinstance(experiment, dict) and experiment.get("kind") == KIND
             and contract.get("phase") == "pixel", "Run is not a C2 original-detail latent diagnostic")
    _require(experiment.get("refiner_loaded") is False and experiment.get("deployable") is False,
             "C2 must not load a refiner or claim deployable output")
    protocol_path = _absolute(request.get("protocol"), "Training protocol")
    protocol = _read(protocol_path)
    protocol_sha = _sha(protocol_path)
    _require(experiment.get("protocol_sha256") == protocol_sha, "Training protocol hash changed")
    _require(protocol.get("kind") == KIND and type(protocol.get("steps_per_case")) is int
             and protocol["steps_per_case"] == STEPS_PER_CASE, "Wrong protocol kind or per-case update bound")
    _require(experiment.get("detail_definition") == protocol.get("detail_definition") == DETAIL_DEFINITION,
             "Original detail definition changed")
    _require(all(_number(value) and value == DETAIL_WEIGHT for value in
                 (experiment.get("detail_weight"), protocol.get("detail_weight"))), "Detail coefficient must be 0.5")
    case_name = request.get("case")
    _require(isinstance(case_name, str) and isinstance(protocol.get("cases"), list), "Missing case binding")
    matches = [case for case in protocol["cases"] if isinstance(case, dict) and case.get("name") == case_name]
    _require(len(matches) == 1, "Case must occur exactly once in the protocol")
    case = matches[0]
    _require(experiment.get("case") == case and _absolute(case.get("run"), "Case run") == run,
             "Training run does not match its original source case")
    config_path = _absolute(protocol.get("config"), "Protocol configuration")
    config_items = [item for item in protocol.get("evidence", []) if isinstance(item, dict)
                    and isinstance(item.get("path"), str) and Path(item["path"]).resolve() == config_path]
    _require(len(config_items) == 1 and config_items[0].get("sha256") == _sha(config_path),
             "Declared configuration file binding changed")
    resolved = load_config(run / "resolved.yaml").model_dump(mode="json")
    resolved_digest = digest(resolved)
    _require(contract.get("resolved_sha256") == resolved_digest
             and digest(load_config(config_path).model_dump(mode="json")) == resolved_digest
             and digest(load_config(BASELINE_CONFIG).model_dump(mode="json")) == resolved_digest,
             "Resolved configuration semantics must match the unchanged C baseline")
    code = protocol.get("code_sha256")
    _require(isinstance(code, dict) and experiment.get("code_sha256") == code, "Experiment implementation bindings changed")
    paths = (Path(__file__).resolve(), Path(engine.__file__).resolve(), ROOT / "scripts/detail_supervision_math.py")
    hashes = {}
    for path in paths:
        key = path.relative_to(ROOT).as_posix()
        hashes[key] = _sha(path)
        _require(code.get(key) == hashes[key], f"Bound chart/detail source changed: {key}")
    return {
        "run": str(run), "case": case_name, "view_id": case.get("view_id"),
        "training_protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "training_contract_sha256": _sha(run / "training_contract.json"),
        "resolved_semantic_sha256": resolved_digest,
        "baseline_config": {"path": str(BASELINE_CONFIG.resolve()), "sha256": _sha(BASELINE_CONFIG)},
        "wrapper_source_sha256": hashes["scripts/plot_original_detail_latent_losses.py"],
        "historical_plot_engine_sha256": hashes["scripts/plot_training_losses.py"],
        "detail_math_sha256": hashes["scripts/detail_supervision_math.py"],
        "detail_definition": DETAIL_DEFINITION, "detail_effective_coefficient": DETAIL_WEIGHT,
        "refiner_loaded": False, "deployable": False,
    }


def _series_statistics(values, window):
    window = min(window, len(values))
    return {"all_logged_steps": float(np.mean(values)), "first_window": float(np.mean(values[:window])),
            "last_window": float(np.mean(values[-window:]))}


@contextmanager
def _plot_context(bindings, window):
    """Patch process-local callbacks only, and always restore the old engine."""
    with _PLOT_LOCK:
        previous_contributions = engine.contributions
        previous_savefig = engine.plt.Figure.savefig
        captured = []

        def collect(rows, config):
            weights, values = contributions(rows, config)
            _require(len(captured) < len(bindings), "Unexpected extra plotting run")
            binding = bindings[len(captured)]
            for row in rows:
                _require(row.get("case", binding["case"]) == binding["case"]
                         and row.get("diagnostic_kind", KIND) == KIND, "Recorded case/kind differs from contract")
            captured.append(rows)
            return weights, values

        def savefig(figure, *args, **kwargs):
            _require(len(captured) == len(bindings) and len(figure.axes) == 3 * len(bindings),
                     "Unexpected historical plot layout")
            for index, (rows, binding) in enumerate(zip(captured, bindings)):
                panels = figure.axes[index*3:index*3+3]
                steps = np.asarray([row["step"] for row in rows])
                application = np.asarray([row["losses"]["application_total"] for row in rows])
                count = min(window, len(rows))
                smooth = np.convolve(application, np.ones(count)/count, mode="valid")
                panels[0].plot(steps[count-1:], smooth, color="#555555", linestyle="--", linewidth=1.5,
                               label=f"application_total; trailing {count}-step mean")
                panels[0].plot(steps, application, color="#555555", alpha=.25, linewidth=.6)
                panels[0].set_title(f"{binding['case']} | single fixed-sample latent")
                panels[0].legend(fontsize=8)
                panels[1].set_title("Objective contributions: detail is raw x 0.5")
            figure.suptitle("Only per-sample latent optimization | fixed samples; no shuffle | H3 frozen; R not loaded", fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, .95))
            return previous_savefig(figure, *args, **kwargs)

        engine.contributions = collect
        engine.plt.Figure.savefig = savefig
        try:
            yield captured
        finally:
            engine.contributions = previous_contributions
            engine.plt.Figure.savefig = previous_savefig


def plot_runs(runs, output, window=4, live=False):
    runs, output = [Path(run).resolve() for run in runs], Path(output)
    _require(runs and len(runs) <= 3 and len(set(runs)) == len(runs), "Select one to three distinct C2 case runs")
    _require(type(window) is int and window > 0, "Smoothing window must be a positive integer")
    bindings = [bound_evidence(run) for run in runs]
    with _plot_context(bindings, window) as captured:
        result = engine.plot(runs, output, window, live=live)
    _require([bound_evidence(run) for run in runs] == bindings, "Bound evidence changed while plotting")
    for evidence, rows in zip(result["runs"], captured):
        evidence["application_total"] = _series_statistics([row["losses"]["application_total"] for row in rows], window)
        evidence["raw_detail"] = _series_statistics([row["losses"]["detail"] for row in rows], window)
    result["original_detail_latent_observer"] = {
        "kind": KIND, "run_bindings": bindings, "same_fixed_sample_each_step": True,
        "detail_effective_coefficient": DETAIL_WEIGHT,
        "objective": "total = application_total + 0.5 * detail; application_total = original RGB + 0.2 original lighting",
        "refiner_loaded": False, "deployable": False,
    }
    result["notes"] = [
        "Each recorded loss is measured before its optimizer update, then logged after that update.",
        "Every row uses the same fixed original sample at every step; there is no shuffled sampling.",
        "application_total is a subtotal shown separately, not an additional objective contribution.",
        "detail is the raw sRGB high-pass term; the objective coefficient is exactly 0.5.",
        "Raw latent is diagnostic only; its objective coefficient is zero.",
        "Only temporary per-sample latents are optimized; H3 is frozen and R is not loaded.",
        "No missing optimizer steps or validation measurements are synthesized; this observer starts no work.",
    ]
    metadata = output.with_suffix(".json")
    temporary = metadata.with_suffix(".original-detail.tmp.json")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata)
    return result


def plot_run(run, output, window=4, live=False):
    return plot_runs([run], output, window=window, live=live)


def watch(run, output, window=4, interval=30):
    _require(_number(interval) and 1 <= interval <= 60, "Watch interval must be between 1 and 60 seconds")
    run = Path(run)
    bound_evidence(run)
    previous = None
    while True:
        metrics = run / "metrics.jsonl"
        snapshot = metrics.read_bytes() if metrics.exists() else b""
        done = any((run / name).exists() for name in ("training_report.json", "training_interruption.json"))
        if b"\n" in snapshot and (snapshot != previous or done):
            plot_run(run, output, window=window, live=not done)
            previous = snapshot
        if done:
            return
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--watch", action="store_true", help="Observe an existing run; never starts or resumes it")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.watch:
        watch(args.run, args.output, args.window, args.interval)
    else:
        plot_run(args.run, args.output, args.window)
