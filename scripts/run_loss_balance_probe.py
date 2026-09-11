"""Run a declared, bounded three-arm real-H3 overfit comparison via the existing CLI."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3ce.config import load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cli_result(contents):
    """The CLI prints its final object with indentation, after JSONL progress."""
    matches = []
    for start in re.finditer(r"(?m)^\{", contents):
        try:
            value, _ = json.JSONDecoder().raw_decode(contents[start.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and {"status", "run", "report", "optimizer_steps"} <= value.keys():
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("Expected exactly one completed CLI training result")
    return matches[0]


def run(config, manifest, output, continue_experiment=False):
    config, manifest, output = (Path(p).resolve() for p in (config, manifest, output))
    if not output.is_relative_to(ROOT / "logs") or (output.exists() and not continue_experiment):
        raise ValueError("Use a new comparison directory inside project logs")
    if continue_experiment:
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if (str(config), sha(config), str(manifest), sha(manifest)) != (
                protocol["source_config"], protocol["source_config_sha256"],
                protocol["manifest"], protocol["manifest_sha256"]):
            raise ValueError("Declared experiment configuration or manifest changed")
        for arm in protocol["arms"]:
            if sha(arm["config"]) != arm["config_sha256"]:
                raise ValueError("Declared arm configuration changed")
        return execute(protocol, output, continue_experiment=True)
    original = load_config(config).model_dump(mode="json")
    if original["native_temporal"]["mode"] != "frozen":
        raise ValueError("This experiment requires a frozen H3 VAE")
    output.mkdir(parents=True)
    config_dir = ROOT / "configs" / output.name
    config_dir.mkdir(exist_ok=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "declared_before_training",
        "scope": "same_16_training_originals_loss_balance_probe_not_base_acceptance",
        "source_config": str(config), "source_config_sha256": sha(config),
        "manifest": str(manifest), "manifest_sha256": sha(manifest),
        "optimizer_steps_per_arm": 64, "gradient_accumulation": original["training"]["gradient_accumulation"],
        "seed": original["project"]["seed"], "resume": "none",
        "initialization": "identical_seed_new_spatial_refiner_zero_output_frozen_H3",
        "only_changed_config_field": "training.losses.latent",
        "arms": [], "trained_base_accepted": False,
        "evaluation": {"primary": ["rgb", "rgb_global_mae"],
            "weighting": ["source_equal_mean", "valid_element_pooled"],
            "clean": "same_16_originals_clean_companions_never_sampled_by_optimizer",
            "cross_arm_total_loss_comparison": False,
            "selection": "Report all arms and cases; favor RGB improvement with the least clean drift; no long training automatically"},
        "runner_sha256": sha(__file__),
    }
    for name, weight in (("latent_010", .1), ("latent_001", .01), ("latent_000", 0.)):
        candidate = copy.deepcopy(original)
        candidate["training"]["losses"]["latent"] = weight
        path = config_dir / (name + ".yaml")
        path.write_text(yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False), encoding="utf-8")
        assert load_config(path).model_dump(mode="json") == candidate
        protocol["arms"].append({"name": name, "latent_weight": weight, "config": str(path), "config_sha256": sha(path)})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return execute(protocol, output)


def execute(protocol, output, continue_experiment=False):
    executions = []
    for arm in protocol["arms"]:
        command = [sys.executable, "-u", "-m", "h3ce", "train", "--config", arm["config"],
                   "--stage", "bootstrap", "--phase", "overfit", "--manifest", protocol["manifest"],
                   "--max-steps", "64", "--resume", "none"]
        result = {"arm": arm["name"], "command": command, "log": str(output / (arm["name"] + ".log"))}
        if Path(result["log"]).exists():
            if not continue_experiment:
                raise ValueError("Arm log already exists")
            contents = Path(result["log"]).read_text(encoding="utf-8")
            result.update(cli_result(contents))
            result["exit_code"] = json.loads(contents.rstrip().splitlines()[-1])["exit_code"]
            validate_finished(result)
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            if report["status"] != result["status"] or report["optimizer_steps"] != 64 or report["phase"] != "overfit":
                raise ValueError("Existing log disagrees with completed training report")
            result["log_sha256"] = sha(result["log"])
            executions.append(result)
            print(json.dumps({"event": "completed_arm_recovered_without_training", **result}), flush=True)
            continue
        print(json.dumps({"event": "arm_started", **arm}), flush=True)
        with Path(result["log"]).open("w", encoding="utf-8") as log:
            log.write(json.dumps({"command": command}) + "\n")
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("event") == "optimizer_step" and record["step"] % 8 == 0:
                        print(json.dumps({"arm": arm["name"], **record}), flush=True)
                result["exit_code"] = process.wait()
                log.write(json.dumps({"exit_code": result["exit_code"]}) + "\n")
        result.update(cli_result(Path(result["log"]).read_text(encoding="utf-8")))
        result["log_sha256"] = sha(result["log"])
        executions.append(result)
        (output / "executions.json").write_text(json.dumps(executions, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "arm_finished", **result}), flush=True)
        validate_finished(result)
    (output / "executions.json").write_text(json.dumps(executions, indent=2) + "\n", encoding="utf-8")
    return executions


def validate_finished(result):
    if result.get("optimizer_steps") != 64 or result.get("status") not in {"passed_overfit_probe", "failed_overfit_probe"}:
        raise RuntimeError("An arm did not complete; preserve its logs and stop the comparison")
    if result["exit_code"] != (0 if result["status"] == "passed_overfit_probe" else 2):
        raise RuntimeError("Unexpected CLI exit code")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-experiment", action="store_true", help="Read completed arm logs; start only arms without logs. Never rerun or auto-resume a partial arm.")
    run(**vars(parser.parse_args()))
