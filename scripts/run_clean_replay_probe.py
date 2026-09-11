"""Declare, train, or evaluate the bounded clean-replay comparison explicitly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3ce.cache.keys import file_sha256
from h3ce.config import load_config
from h3ce.train.engine import implementation_hashes
from scripts.run_loss_balance_probe import cli_result, validate_finished


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def declare(experiment, config, selection):
    experiment, config, selection = (Path(p).resolve() for p in (experiment, config, selection))
    if not experiment.is_relative_to(ROOT / "logs") or (experiment / "protocol.json").exists():
        raise ValueError("Use an experiment in logs without an existing protocol")
    cfg = load_config(config)
    if (cfg.native_temporal.mode != "frozen" or cfg.training.losses.latent != 0
            or cfg.training.losses.perceptual != 0 or cfg.project.seed != 42
            or cfg.training.gradient_accumulation != 4 or cfg.training.stages.bootstrap_pixel.lr != 1e-5):
        raise ValueError("This protocol requires the selected frozen-H3 pixel objective and fixed schedule")
    data = read_json(selection)
    experiment.mkdir(parents=True, exist_ok=True)
    arms = []
    for name in ("control", "replay"):
        manifest = data["manifests"][name]
        if file_sha256(Path(manifest["path"])) != manifest["sha256"]:
            raise ValueError("Selected manifest changed")
        arms.append({"name": name, "config": str(config), "config_sha256": file_sha256(config),
                     "manifest": manifest["path"], "manifest_sha256": manifest["sha256"]})
    protocol = {"status": "declared_before_training", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "eight_training_originals_clean_replay_and_same_source_heldout_probe",
        "source_config": str(config), "source_config_sha256": file_sha256(config),
        "selection": str(selection), "selection_sha256": file_sha256(selection),
        "full_manifest": data["full_manifest"], "full_manifest_sha256": data["full_manifest_sha256"],
        "arms": arms, "seed": 42, "optimizer_steps_per_arm": 64, "gradient_accumulation": 4,
        "resume": "none", "initializer": "fresh_zero_output_refiner_same_seed_no_prior_model",
        "changed_factor": "clean_replay_in_optimizer_manifest_at_equal_64_step_budget",
        "pair_budget": {"control": {"degraded": 8, "clean": 0}, "replay": {"degraded": 8, "clean": 8}},
        "expected_sample_visits_per_original": {"control": {"degraded": 32, "clean": 0},
                                                 "replay": {"degraded": 16, "clean": 16}},
        "evaluation_suites": ["train_degraded", "train_clean", "heldout_degraded", "heldout_clean"],
        "evaluation_policy": "Report every case and both weightings, with clean MAE absolute; no best-step selection",
        "independent_validation": False, "trained_base_accepted": False, "automatic_long_training": False,
        "implementation_sha256": implementation_hashes(ROOT), "runner_sha256": file_sha256(Path(__file__))}
    write_json(experiment / "protocol.json", protocol)
    print(json.dumps({"status": "declared_before_training", "protocol": str(experiment / "protocol.json")}), flush=True)


def checked_protocol(experiment):
    experiment = Path(experiment).resolve()
    if not experiment.is_relative_to(ROOT / "logs"):
        raise ValueError("Experiment must be inside project logs")
    protocol = read_json(experiment / "protocol.json")
    if file_sha256(Path(__file__)) != protocol["runner_sha256"]:
        raise ValueError("Experiment runner changed after declaration")
    if implementation_hashes(ROOT) != protocol["implementation_sha256"]:
        raise ValueError("Training implementation changed after declaration")
    for field in ("source_config", "selection", "full_manifest"):
        if file_sha256(Path(protocol[field])) != protocol[field + "_sha256"]:
            raise ValueError(f"Declared {field} changed")
    for arm in protocol["arms"]:
        for field in ("config", "manifest"):
            if file_sha256(Path(arm[field])) != arm[field + "_sha256"]:
                raise ValueError("Declared arm input changed")
    return experiment, protocol


def execute(command, log_path, label):
    """No shell interpolation and no overwrite of a previous execution log."""
    print(json.dumps({"event": "started", "label": label}), flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}) + "\n")
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict) and value.get("event") == "optimizer_step" and value["step"] % 8 == 0:
                    print(json.dumps({"label": label, **value}), flush=True)
            code = process.wait()
            log.write(json.dumps({"exit_code": code}) + "\n")
    return {"command": command, "log": str(log_path), "log_sha256": file_sha256(log_path), "exit_code": code}


def train(experiment):
    experiment, protocol = checked_protocol(experiment)
    if any((experiment / (arm["name"] + ".log")).exists() for arm in protocol["arms"]):
        raise ValueError("An arm already has a log; completed or partial training is never automatically repeated")
    executions = []
    for arm in protocol["arms"]:
        checked_protocol(experiment)
        command = [sys.executable, "-u", "-m", "h3ce", "train", "--config", arm["config"],
                   "--stage", "bootstrap", "--phase", "overfit", "--manifest", arm["manifest"],
                   "--max-steps", "64", "--resume", "none"]
        result = {"arm": arm["name"], **execute(command, experiment / (arm["name"] + ".log"), arm["name"])}
        result.update(cli_result(Path(result["log"]).read_text(encoding="utf-8")))
        validate_finished(result)
        executions.append(result)
        write_json(experiment / "executions.json", executions)
        print(json.dumps({"event": "training_completed", **result}), flush=True)


def evaluate(experiment):
    experiment, protocol = checked_protocol(experiment)
    executions = read_json(experiment / "executions.json")
    if [a["arm"] for a in executions] != [a["name"] for a in protocol["arms"]]:
        raise ValueError("Both declared arms must finish before evaluation")
    selection = read_json(protocol["selection"])
    results = []
    for arm in executions:
        validate_finished(arm)
        for suite in protocol["evaluation_suites"]:
            manifest = selection["manifests"][suite]
            if file_sha256(Path(manifest["path"])) != manifest["sha256"]:
                raise ValueError("Selected evaluation manifest changed")
            output = Path(arm["run"]) / (suite + "_evaluation")
            if output.exists():
                raise ValueError("Evaluation output already exists")
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_bootstrap_checkpoint.py"),
                       "--run", arm["run"], "--manifest", manifest["path"], "--output", str(output)]
            if suite.startswith("heldout_"):
                command += ["--heldout-source-manifest", protocol["full_manifest"]]
            elif suite == "train_clean":
                command += ["--companion-source-manifest", protocol["full_manifest"]]
            result = {"arm": arm["arm"], "suite": suite,
                      **execute(command, experiment / (arm["arm"] + "_" + suite + ".log"), arm["arm"] + " " + suite)}
            results.append(result)
            write_json(experiment / "evaluation_executions.json", results)
            if result["exit_code"]:
                raise RuntimeError("Evaluation failed; preserve its log and investigate before retry")
            path = output / "metrics.json"
            metrics = read_json(path)
            if metrics["additional_optimizer_steps"] != 0 or not metrics["h3_parameters_frozen_and_unchanged"]:
                raise RuntimeError("Forward evaluation mutation guard did not pass")
            result.update(report=str(path), report_sha256=file_sha256(path))
            write_json(experiment / "evaluation_executions.json", results)
            print(json.dumps({"event": "evaluation_completed", "arm": arm["arm"], "suite": suite,
                "means": metrics["summaries"]["current"]["source_equal_mean"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "train", "evaluate"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config")
    parser.add_argument("--selection")
    args = parser.parse_args()
    if args.action == "declare":
        if not args.config or not args.selection:
            parser.error("declare requires --config and --selection")
        declare(args.experiment, args.config, args.selection)
    elif args.config or args.selection:
        parser.error("Training/evaluation read the declared configuration; do not provide overrides")
    else:
        {"train": train, "evaluate": evaluate}[args.action](args.experiment)
