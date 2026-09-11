"""Bounded frozen-H3 learning-rate comparison, with observed loss charts and saved-step evaluations."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
import yaml
from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import atomic_write
from h3ce.config import load_config
from h3ce.train.engine import implementation_hashes
from scripts.plot_training_losses import plot, read_records
from scripts.evaluate_bootstrap_checkpoint import read_committed
from scripts.run_clean_replay_probe import execute
from scripts.run_loss_balance_probe import cli_result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_configs(source):
    require(source["native_temporal"]["mode"] == "frozen" and not source["vae"]["encoder_trainable"]
            and source["training"]["losses"]["latent"] == source["training"]["losses"]["perceptual"] == 0
            and source["project"]["seed"] == 42 and source["training"]["gradient_accumulation"] == 4
            and source["data"]["sampling"]["profile"] == "balanced"
            and source["data"]["degradation"]["resolution"]["ratio_range"] == [.18, .18],
            "Expected the existing medium-resolution frozen image experiment")
    result = []
    for name, lr in (("lr_1e5", 1e-5), ("lr_3e5", 3e-5)):
        cfg = copy.deepcopy(source)
        cfg["training"]["stages"]["bootstrap_pixel"]["lr"] = lr
        cfg["training"]["checkpoint_every_steps"] = 64
        cfg["training"]["preview_every_steps"] = 64
        result.append((name, cfg))
    return result


def declare(output, recovery_run=None):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "logs") and not output.exists(), "Use a new experiment in logs")
    source = ROOT / "configs/quality-calibration-20260908/medium.yaml"
    cfg = load_config(source).model_dump(mode="json")
    base = ROOT / "runs/quality-calibration-20260908/encoded/medium"
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "declared_before_training",
        "authorization": "User requested continued tuning and testing after the loss diagnosis and controlled comparison proposal",
        "scope": "eight_same_source_originals_learning_rate_and_training_length_probe",
        "source_config": str(source), "source_config_sha256": file_sha256(source),
        "optimizer_steps_per_arm": 256, "evaluation_steps": [0, 64, 128, 256],
        "resume": "none", "initializer": "fresh_same_seed_zero_output_refiner",
        "changed_factor_between_arms": "training.stages.bootstrap_pixel.lr",
        "common_logging_changes_from_previous_run": {"checkpoint_every_steps": 64, "preview_every_steps": 64},
        "implementation_sha256": implementation_hashes(ROOT), "runner_sha256": file_sha256(Path(__file__)),
        "independent_validation": False, "trained_base_accepted": False, "automatic_training_beyond_arms": False,
        "arms": [], "inputs": {}}
    for key, filename in (("training", "overfit_manifest.jsonl"), ("degraded", "degraded_manifest.jsonl"),
                          ("clean", "clean_manifest.jsonl"), ("full", "training_manifest.jsonl")):
        path = base / filename
        protocol["inputs"][key] = {"path": str(path), "sha256": file_sha256(path)}
    configs = ROOT / "configs" / output.name
    require(not configs.exists(), "Config directory already exists")
    configs.mkdir(parents=True)
    for name, data in arm_configs(cfg):
        path = configs / (name + ".yaml")
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        require(load_config(path).model_dump(mode="json") == data, "Resolved configuration differs")
        protocol["arms"].append({"name": name, "lr": data["training"]["stages"]["bootstrap_pixel"]["lr"],
                                 "config": str(path), "config_sha256": file_sha256(path)})
    if recovery_run:
        prior_run = Path(recovery_run).resolve()
        interruption = read(prior_run / "training_interruption.json")
        require(interruption["optimizer_may_have_partially_updated"] is False, "Recovery requires a complete optimizer boundary")
        prior_config, state, checkpoint = read_committed(ROOT, prior_run, interruption["resume_checkpoint"])
        require(state["step"] == interruption["last_completed_step"] == 2 and state["contract"]["max_steps"] == 256
                and state["resolved_config"] == load_config(protocol["arms"][0]["config"]).model_dump(mode="json"),
                "Recovery must retain exactly the interrupted arm configuration and schedule")
        protocol["arms"][0].update(resume_checkpoint=str(checkpoint), resume_checkpoint_sha256=file_sha256(checkpoint),
            prefix_run=str(prior_run), prefix_metrics_sha256=file_sha256(prior_run / "metrics.jsonl"), resume_step=2)
        protocol["recovery"] = {"scope": "explicit continuation from verified step 2, no replay of completed optimizer steps",
            "source_run": str(prior_run), "interruption": interruption,
            "cause": "Observer compared raw YAML bytes; platform newline encoding differed despite equal resolved config",
            "repair": "Compare semantic resolved-config digests; preserve observer failures without closing child stdout"}
    output.mkdir(parents=True)
    atomic_write(output / "protocol.json", canonical_json(protocol))
    return protocol


def check(protocol):
    require(file_sha256(Path(__file__)) == protocol["runner_sha256"], "Runner changed after declaration")
    require(implementation_hashes(ROOT) == protocol["implementation_sha256"], "Training implementation changed")
    require(file_sha256(Path(protocol["source_config"])) == protocol["source_config_sha256"], "Source config changed")
    for item in protocol["inputs"].values():
        require(file_sha256(Path(item["path"])) == item["sha256"], "Declared manifest changed")
    for arm in protocol["arms"]:
        require(file_sha256(Path(arm["config"])) == arm["config_sha256"], "Declared arm changed")
        if arm.get("prefix_run"):
            require(file_sha256(Path(arm["resume_checkpoint"])) == arm["resume_checkpoint_sha256"]
                    and file_sha256(Path(arm["prefix_run"]) / "metrics.jsonl") == arm["prefix_metrics_sha256"],
                    "Recovery evidence changed")


def find_new_run(previous, arm):
    config_digest = digest(load_config(arm["config"]).model_dump(mode="json"))
    candidates = [p for p in (ROOT / "runs").iterdir() if p.is_dir() and p not in previous
                  and (p / "resolved.yaml").is_file()
                  and digest(load_config(p / "resolved.yaml").model_dump(mode="json")) == config_digest
                  and (p / "training_contract.json").is_file()]
    require(len(candidates) == 1, "Cannot uniquely bind this child training process to a new run")
    return candidates[0]


def chart_for_run(arm, current_run, output):
    observed = current_run
    if arm.get("prefix_run"):
        prefix = Path(arm["prefix_run"])
        first, first_raw = read_records(prefix / "metrics.jsonl")
        second, second_raw = read_records(current_run / "metrics.jsonl")
        require(first[0]["step"] == 1 and first[-1]["step"] == arm["resume_step"]
                and second[0]["step"] == first[-1]["step"] + 1, "Recovery metrics have overlap or a missing step")
        observed = output / (arm["name"] + "_chart_records")
        observed.mkdir(exist_ok=True)
        atomic_write(observed / "resolved.yaml", (current_run / "resolved.yaml").read_bytes())
        atomic_write(observed / "metrics.jsonl", first_raw + second_raw)
        atomic_write(observed / "sources.json", canonical_json({"scope": "concatenated actual logs across explicit resume",
            "runs": [str(prefix), str(current_run)], "sha256": [file_sha256(prefix / "metrics.jsonl"), file_sha256(current_run / "metrics.jsonl")]}))
    return plot([observed], output / (arm["name"] + "_loss.png"), window=8)


def validate_result(value, code, steps):
    status = value.get("status")
    require(value.get("optimizer_steps") == steps and status in {"passed_overfit_probe", "failed_overfit_probe"}
            and code == (0 if status == "passed_overfit_probe" else 2), "Training did not finish its declared limit")


def train_arm(protocol, arm, output):
    check(protocol)
    log_path = output / (arm["name"] + ".log")
    require(not log_path.exists(), "Completed or partial training cannot be automatically repeated")
    previous = set((ROOT / "runs").iterdir())
    command = [sys.executable, "-u", "-m", "h3ce", "train", "--config", arm["config"], "--stage", "bootstrap",
               "--phase", "overfit", "--manifest", protocol["inputs"]["training"]["path"],
               "--max-steps", str(protocol["optimizer_steps_per_arm"]), "--resume", arm.get("resume_checkpoint", "none")]
    current_run, chart_errors = None, []
    print(json.dumps({"event": "arm_started", **arm}), flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}) + "\n")
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as child:
            for line in child.stdout:
                log.write(line); log.flush()
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") == "optimizer_step":
                    if current_run is None:
                        try:
                            current_run = find_new_run(previous, arm)
                        except Exception as exc:
                            chart_errors.append({"step": row["step"], "error": str(exc), "scope": "run_binding"})
                            atomic_write(output / (arm["name"] + "_chart_errors.json"), canonical_json(chart_errors))
                            continue  # Preserve child stdout; never interrupt optimization because an observer failed.
                    atomic_write(output / "progress.json", canonical_json({"arm": arm["name"], "run": str(current_run),
                        "step": row["step"], "limit": protocol["optimizer_steps_per_arm"], "losses": row["losses"]}))
                    if row["step"] == 1 or row["step"] % 8 == 0:
                        try:
                            chart_for_run(arm, current_run, output)
                        except Exception as exc:
                            chart_errors.append({"step": row["step"], "error": str(exc)})
                            atomic_write(output / (arm["name"] + "_chart_errors.json"), canonical_json(chart_errors))
                        print(json.dumps({"arm": arm["name"], **row}), flush=True)
            code = child.wait()
            log.write(json.dumps({"exit_code": code}) + "\n")
    value = cli_result(log_path.read_text(encoding="utf-8"))
    result = {"arm": arm["name"], "lr": arm["lr"], "command": command, "exit_code": code, **value,
              "log": str(log_path), "log_sha256": file_sha256(log_path), "chart_errors": chart_errors}
    atomic_write(output / (arm["name"] + "_execution.json"), canonical_json(result))
    validate_result(value, code, protocol["optimizer_steps_per_arm"])
    require(Path(value["run"]) == current_run, "CLI final run differs from observed run")
    chart_for_run(arm, current_run, output)
    check(protocol)
    return result


def checkpoint_at(run, step):
    report = read(run / "training_report.json")
    if step == report["optimizer_steps"]:
        return Path(report["checkpoint"])
    receipts = [p for p in (run / "checkpoints").glob(f"checkpoint-overfit-{step:012d}-*.json")]
    require(receipts, f"No saved checkpoint for step {step}")
    # Step zero has before- and after-baseline copies; use the later committed one.
    if step != 0:
        require(len(receipts) == 1, "Ambiguous intermediate checkpoint")
    return max(receipts, key=lambda p: read(p)["created_ns"]).with_suffix(".pt")


def evaluate_arm(protocol, result, output):
    run = Path(result["run"])
    arm = next(a for a in protocol["arms"] if a["name"] == result["arm"])
    evaluations = []
    for step in protocol["evaluation_steps"]:
        evaluation_run = Path(arm["prefix_run"]) if step == 0 and arm.get("prefix_run") else run
        if step == 0 and arm.get("prefix_run"):
            receipts = list((evaluation_run / "checkpoints").glob("checkpoint-overfit-000000000000-*.json"))
            require(receipts, "Recovery source has no initial checkpoint")
            checkpoint = max(receipts, key=lambda p: read(p)["created_ns"]).with_suffix(".pt")
        else:
            checkpoint = checkpoint_at(run, step)
        for kind in ("degraded", "clean"):
            check(protocol)
            destination = run / f"step_{step:04d}_{kind}_evaluation"
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_bootstrap_checkpoint.py"),
                       "--run", str(evaluation_run), "--checkpoint", str(checkpoint), "--manifest", protocol["inputs"][kind]["path"],
                       "--output", str(destination)]
            if kind == "clean":
                command += ["--companion-source-manifest", protocol["inputs"]["full"]["path"]]
            label = f"{result['arm']}_{step:04d}_{kind}"
            record = {"arm": result["arm"], "step": step, "kind": kind,
                      **execute(command, output / (label + ".log"), label)}
            require(record["exit_code"] == 0, "Fixed-step evaluation failed; preserve its log")
            path = destination / "metrics.json"
            data = read(path)
            require(data["additional_optimizer_steps"] == 0 and all(v == 0 for v in data["execution_guard"].values())
                    and data["h3_parameters_frozen_and_unchanged"]
                    and data["models"]["current"]["optimizer_steps_in_checkpoint"] == step,
                    "Evaluation guard or checkpoint step disagrees")
            record.update(report=str(path), report_sha256=file_sha256(path))
            evaluations.append(record)
            atomic_write(output / (result["arm"] + "_evaluations.json"), canonical_json(evaluations))
            print(json.dumps({"event": "fixed_step_completed", "arm": result["arm"], "step": step, "kind": kind,
                              "metrics": data["summaries"]["current"]["source_equal_mean"]}), flush=True)
    return evaluations


def main(output, action, recovery_run=None):
    output = Path(output).resolve()
    if action == "declare":
        declare(output, recovery_run)
        print(json.dumps({"status": "declared_before_training", "protocol": str(output / "protocol.json")}), flush=True)
        return
    protocol = read(output / "protocol.json")
    require(recovery_run is None, "Recovery is declared once in the protocol, not supplied at execution")
    check(protocol)
    require(not any((output / (arm["name"] + ".log")).exists() for arm in protocol["arms"]),
            "Existing training requires inspection; this runner never repeats or auto-resumes")
    results = []
    for arm in protocol["arms"]:
        result = train_arm(protocol, arm, output)
        results.append(result)
        atomic_write(output / "executions.json", canonical_json(results))
        evaluate_arm(protocol, result, output)
    atomic_write(output / "completed.json", canonical_json({"status": "both_arms_and_fixed_steps_completed",
        "new_optimizer_steps": sum(r["optimizer_steps"] - a.get("resume_step", 0) for r, a in zip(results, protocol["arms"])),
        "total_optimizer_steps_across_arms": sum(r["optimizer_steps"] for r in results), "trained_base_accepted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action", choices=["declare", "run"], required=True)
    parser.add_argument("--recovery-run", help="Declare explicit step-2 recovery for the identified observer interruption")
    main(**vars(parser.parse_args()))
