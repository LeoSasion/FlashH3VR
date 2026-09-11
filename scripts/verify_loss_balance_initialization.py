"""Verify a declared loss comparison from committed state, without constructing models.

This is a CPU inspection of saved real-training evidence, not another GPU test or
an interrupted-versus-uninterrupted training experiment. Only --report is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.config import load_config
from h3ce.train.checkpoint import CheckpointManager
from h3ce.train.engine import implementation_hashes
from h3ce.train.guard import no_training_guard
from h3ce.train.preflight import require
from h3ce.train.sampler import StatefulSampler


ARMS = {"latent_010": .1, "latent_001": .01, "latent_000": 0.}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    assert_no_links(path)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def project_path(root, value):
    require(isinstance(value, (str, Path)) and str(value), "Missing project evidence path")
    path = (root / value).resolve()
    require(path.is_relative_to(root) and path != root, "Evidence must remain inside this project")
    assert_no_links(path)
    return path


def tensor_bytes(value):
    require(value.device.type == "cpu" and value.layout == torch.strided,
            "Saved tensor inspection must use dense CPU tensors")
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_hash(value):
    """Hash typed structure and raw tensor bytes, including signed zero bits."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            header = ["tensor", str(item.dtype), list(item.shape)]
            data = tensor_bytes(item)
        elif isinstance(item, dict):
            result.update(canonical_json(["dict", len(item)]))
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
            return
        elif isinstance(item, (tuple, list)):
            result.update(canonical_json([type(item).__name__, len(item)]))
            for child in item:
                visit(child)
            return
        else:
            header = [type(item).__name__]
            data = canonical_json(item)
        result.update(canonical_json([header, len(data)]))
        result.update(data)

    visit(value)
    return result.hexdigest()


def assert_equal(expected, actual, label):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and expected.dtype == actual.dtype
                and expected.shape == actual.shape and torch.equal(expected, actual)
                and tensor_bytes(expected) == tensor_bytes(actual), f"{label}: tensor bytes differ")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), f"{label}: dictionary keys differ")
        for key in expected:
            assert_equal(expected[key], actual[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(expected) is type(actual) and len(expected) == len(actual), f"{label}: sequence differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equal(left, right, f"{label}[{index}]")
    else:
        require(type(expected) is type(actual) and expected == actual, f"{label}: scalar differs")


def without_latent(config):
    result = copy.deepcopy(config)
    result["training"]["losses"].pop("latent")
    return result


def manifest_views(path):
    fields = {"source_asset": "asset_id", "prepared_target": "target_id",
              "degraded_variant": "variant_id", "training_view": "view_id"}
    records = {kind: {} for kind in fields}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_object)
        require(isinstance(row, dict) and row.get("record_type") in fields, "Unknown manifest record")
        kind = row["record_type"]
        identifier = row.get(fields[kind])
        require(isinstance(identifier, str) and identifier and identifier not in records[kind],
                "Missing or duplicate manifest identifier")
        records[kind][identifier] = row
    views = list(records["training_view"].values())
    require(len(views) == 16 and len({view["asset_id"] for view in views}) == 16,
            "The declared experiment needs exactly 16 distinct training originals")
    require(Counter(view["mode"] for view in views) == {"fullbody": 8, "face": 8},
            "The fixed manifest must contain 8 fullbody and 8 face views")
    for view in views:
        source = records["source_asset"].get(view["asset_id"])
        target = records["prepared_target"].get(view["target_id"])
        variant = records["degraded_variant"].get(view["variant_id"])
        require(source is not None and target is not None and variant is not None,
                "Training view has an incomplete manifest parent chain")
        require(target["asset_id"] == view["asset_id"] and variant["target_id"] == view["target_id"]
                and variant.get("clean_pair") is False, "Only the declared degraded views may be sampled")
    require(len({view["encoder_contract_id"] for view in views}) == 1, "Encoder contracts differ within the manifest")
    return views


def replay_sampler(initial, final, *, seed, steps, accumulation, view_ids):
    """Reconstruct indices from the private CPU generator and check both endpoints."""
    sampler = StatefulSampler(len(view_ids), seed=seed)
    assert_equal(sampler.state_dict(), initial, "initial sampler must be fresh")
    indices = []
    for _ in range(steps):
        sampler.begin_window()
        indices.extend(sampler.next_index() for _ in range(accumulation))
        sampler.commit_window()
    assert_equal(sampler.state_dict(), final, "replayed final sampler")
    counts = Counter(view_ids[index] for index in indices)
    require(len(indices) == 256 and set(counts.values()) == {16} and len(counts) == 16,
            "Expected 256 sample visits, with exactly 16 visits to every fixed view")
    return {"scope": "deterministic_reconstruction_with_committed_endpoint_agreement_not_recorded_per_microbatch_trace",
            "indices": indices, "indices_sha256": digest(indices), "per_view_visits": dict(counts),
            "microbatch_visits": len(indices), "completed_full_passes": 16,
            "final_epoch": sampler.epoch, "final_cursor": sampler.cursor,
            "initial_state_sha256": state_hash(initial), "final_state_sha256": state_hash(final)}


def check_optimizer(initial, final, model, steps):
    require(isinstance(initial, dict) and initial.get("state") == {}, "Initial optimizer must have empty history")
    require(isinstance(final, dict) and isinstance(final.get("state"), dict), "Missing completed optimizer state")
    assert_equal(initial["param_groups"], final["param_groups"], "optimizer parameter groups")
    groups = initial["param_groups"]
    ids = [identifier for group in groups for identifier in group["params"]]
    require(len(ids) == len(set(ids)) == len(model) and set(final["state"]) == set(ids),
            "Every refiner parameter must have exactly one completed optimizer state")
    require(all(type(identifier) is int for identifier in ids), "Optimizer parameter identifiers must be integers")
    for identifier, parameter in zip(ids, model.values()):
        history = final["state"][identifier]
        require(set(history) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected AdamW state layout")
        value = history["step"]
        require(isinstance(value, torch.Tensor) and value.numel() == 1
                and torch.isfinite(value).all().item() and float(value) == steps,
                "Every AdamW parameter step must equal the declared final step")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = history[key]
            require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                    and moment.dtype == parameter.dtype and torch.isfinite(moment).all().item(),
                    "Invalid AdamW moment tensor")
    return {"initial_state_entries": 0, "final_state_entries": len(ids), "all_final_parameter_steps": steps}


def verify_arm(root, arm, execution, protocol, source_config, views):
    config_path = project_path(root, arm["config"])
    require(file_sha256(config_path) == arm["config_sha256"], "Arm config checksum changed")
    config = load_config(config_path)
    config_data = config.model_dump(mode="json")
    require(without_latent(config_data) == without_latent(source_config)
            and config.training.losses.latent == arm["latent_weight"],
            "Only training.losses.latent may differ from the source configuration")
    runs = (root / config.paths.runs).resolve()
    run = project_path(root, execution["run"])
    require(run.is_relative_to(runs) and run != runs, "Arm run must be inside project runs")
    require(load_config(run / "resolved.yaml").model_dump(mode="json") == config_data,
            "Actual resolved configuration differs from the declared arm")
    log = project_path(root, execution["log"])
    require(file_sha256(log) == execution["log_sha256"], "Execution log checksum changed")
    command = execution["command"]
    expected_tail = ["-u", "-m", "h3ce", "train", "--config", str(config_path),
                     "--stage", "bootstrap", "--phase", "overfit", "--manifest", str(project_path(root, protocol["manifest"])),
                     "--max-steps", "64", "--resume", "none"]
    require(isinstance(command, list) and command[1:] == expected_tail, "Execution command differs from the declared fresh experiment")
    request = read_json(run / "training_request.json")
    manifest = project_path(root, protocol["manifest"])
    expected_request = {"stage": "bootstrap", "phase": "overfit", "check_only": False,
                        "manifest": str(manifest), "resume": "none", "max_steps": 64,
                        "overfit_run": None, "init_run": None}
    require(request == expected_request, "Actual training request must use a fresh overfit run without an initializer")
    contract = read_json(run / "training_contract.json")
    require(contract["phase"] == "overfit" and type(contract["max_steps"]) is int and contract["max_steps"] == 64
            and contract["manifest_sha256"] == protocol["manifest_sha256"]
            and contract["resolved_sha256"] == digest(config_data), "Training contract differs from the declared experiment")
    compatibility = contract["compatibility"]
    require(compatibility["implementation_sha256"] == implementation_hashes(root)
            and compatibility["components_sha256"] == file_sha256(root / config.paths.components_lock)
            and compatibility["architecture"] == config.model.model_dump(mode="json")
            and compatibility["data"] == config.data.model_dump(mode="json")
            and compatibility["encoder_contract_id"] == views[0]["encoder_contract_id"],
            "Current implementation, components, architecture, data or encoder differs")
    require(contract.get("lineage") == {"overfit_checkpoint_sha256": None, "initializer_checkpoint_sha256": None},
            "The declared experiment must start without prior trained weights")
    require(isinstance(compatibility.get("runtime"), dict) and compatibility["runtime"], "Missing recorded training runtime")
    manager = CheckpointManager(runs, run, contract=contract)
    initial_candidates = []
    for receipt in sorted((run / "checkpoints").glob("checkpoint-overfit-000000000000-*.json")):
        initial_path = receipt.with_suffix(".pt")
        initial = manager.read(initial_path)
        if "probe_before" in initial["extra"]:
            initial_candidates.append((initial_path, initial))
    require(len(initial_candidates) == 1, "Expected one committed step-zero boundary containing probe_before")
    initial_path, initial = initial_candidates[0]
    report_path = run / "training_report.json"
    require(project_path(root, execution["report"]) == report_path, "Execution report points to another run")
    report = read_json(report_path)
    final_path = project_path(root, report["checkpoint"])
    require(final_path.parent == run / "checkpoints", "Final checkpoint must belong to its arm run")
    final = manager.read(final_path)
    for label, state in (("initial", initial), ("final", final)):
        require(state["stage"] == "overfit" and state["resolved_config"] == config_data,
                f"{label} checkpoint has incompatible resolved state")
        require(isinstance(state["model"], dict) and state["model"]
                and all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item()
                        for value in state["model"].values()), f"{label} model must contain finite tensors")
    require(type(initial["step"]) is int and initial["step"] == 0
            and initial["extra"].get("phase_complete", False) is False,
            "Initial checkpoint is not the fresh pre-training boundary")
    for key in ("output_projection.weight", "output_projection.bias"):
        require(key in initial["model"] and torch.count_nonzero(initial["model"][key]).item() == 0,
                "Initial output projection must be exactly zero")
    require(initial["model"].keys() == final["model"].keys(), "Final model parameter keys changed")
    for key in initial["model"]:
        require(initial["model"][key].shape == final["model"][key].shape
                and initial["model"][key].dtype == final["model"][key].dtype, "Final parameter layout changed")
    require(type(final["step"]) is int and final["step"] == report["optimizer_steps"] == execution["optimizer_steps"] == 64
            and final["extra"].get("phase_complete") is True and final["extra"] == report["extra"],
            "Final checkpoint and report must agree on all 64 completed steps")
    require(report["phase"] == "overfit" and report["status"] == execution["status"]
            and report["status"] in {"passed_overfit_probe", "failed_overfit_probe"}
            and report["trained_base_accepted"] is False and report["training_started"] is True
            and report["vae_parameters_unchanged"] is True and report["manifest_sha256"] == protocol["manifest_sha256"],
            "Completed report lacks frozen-VAE evidence or changes the quality scope")
    require(execution["exit_code"] == (0 if report["status"] == "passed_overfit_probe" else 2), "Unexpected training CLI exit code")
    assert_equal(initial["extra"]["probe_before"], final["extra"]["probe_before"], "preserved initial probe")
    require(final["extra"]["overfit_evidence"]["view_ids"] == [view["view_id"] for view in views],
            "Final overfit evidence names different training views")
    optimizer = check_optimizer(initial["optimizer"], final["optimizer"], final["model"], 64)
    for label, state, step in (("initial", initial, 0), ("final", final, 64)):
        require(state["scheduler"]["last_epoch"] == step and state["scheduler"]["_step_count"] == step + 1,
                f"{label} scheduler does not match its optimizer boundary")
        require(state["scaler"] == {}, "This frozen-H3/BF16 experiment must keep its scaler disabled")
    replay = replay_sampler(initial["sampler"], final["sampler"], seed=protocol["seed"], steps=64,
                            accumulation=4, view_ids=[view["view_id"] for view in views])
    metrics_path = run / "metrics.jsonl"
    metrics = [json.loads(line, object_pairs_hook=unique_object) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require([row["step"] for row in metrics] == list(range(1, 65))
            and all(row["phase"] == "overfit" and row["learning_rate"] == 1e-5 for row in metrics),
            "Training metrics do not record exactly 64 steps at the declared learning rate")
    details = {"name": arm["name"], "latent_weight": arm["latent_weight"], "run": str(run),
               "config_sha256": arm["config_sha256"], "contract_id": initial["contract_id"],
               "initial_checkpoint": str(initial_path), "initial_checkpoint_sha256": file_sha256(initial_path),
               "initial_model_content_sha256": state_hash(initial["model"]),
               "initial_tensor_count": len(initial["model"]),
               "initial_parameter_elements": sum(value.numel() for value in initial["model"].values()),
               "initial_state_sha256": {key: state_hash(initial[key]) for key in ("optimizer", "scheduler", "scaler", "sampler", "rng")},
               "initial_output_projection_zero": True, "initial_cuda_rng_states_saved": len(initial["rng"]["torch_cuda"]),
               "final_checkpoint": str(final_path), "final_checkpoint_sha256": file_sha256(final_path),
               "final_model_content_sha256": state_hash(final["model"]), "training_status": report["status"],
               "completed_optimizer_steps": 64, "optimizer": optimizer, "sampler_replay": replay,
               "metrics_sha256": file_sha256(metrics_path), "log_sha256": execution["log_sha256"]}
    return details, initial, final, contract


def inspect(root, protocol_path, executions_path):
    protocol, executions = read_json(protocol_path), read_json(executions_path)
    require(protocol["status"] == "declared_before_training" and protocol["only_changed_config_field"] == "training.losses.latent"
            and protocol["optimizer_steps_per_arm"] == 64 and protocol["gradient_accumulation"] == 4
            and protocol["seed"] == 42 and protocol["resume"] == "none" and protocol["trained_base_accepted"] is False,
            "Protocol does not declare the expected bounded three-arm experiment")
    arms = protocol["arms"]
    require(isinstance(arms, list) and len(arms) == 3
            and {arm["name"]: arm["latent_weight"] for arm in arms} == ARMS,
            "Protocol must declare exactly the .1, .01 and zero latent-weight arms")
    require(isinstance(executions, list) and len(executions) == 3
            and {entry["arm"] for entry in executions} == set(ARMS), "All three arm executions must be present exactly once")
    by_name = {entry["arm"]: entry for entry in executions}
    require(len({project_path(root, entry["run"]) for entry in executions}) == 3,
            "Each arm must have its own fresh training run")
    source_path = project_path(root, protocol["source_config"])
    require(file_sha256(source_path) == protocol["source_config_sha256"], "Source configuration checksum changed")
    source = load_config(source_path).model_dump(mode="json")
    require(source["native_temporal"]["mode"] == "frozen" and source["native_temporal"]["decoder_adapter"] is None
            and source["training"]["microbatch"] == 1 and source["training"]["gradient_accumulation"] == 4
            and source["training"]["stages"]["bootstrap_pixel"]["lr"] == 1e-5
            and source["training"]["losses"]["perceptual"] == 0 and source["project"]["seed"] == 42,
            "Source config does not match the frozen-H3 loss comparison")
    declared_runner = protocol_path.parent / "runner_declared.py"
    if not declared_runner.exists():
        declared_runner = root / "scripts/run_loss_balance_probe.py"
    assert_no_links(declared_runner)
    require(file_sha256(declared_runner) == protocol["runner_sha256"], "Declared runner source checksum changed")
    manifest = project_path(root, protocol["manifest"])
    require(file_sha256(manifest) == protocol["manifest_sha256"], "Declared training manifest checksum changed")
    views = manifest_views(manifest)
    reports, anchor_initial, anchor_final, anchor_contract = [], None, None, None
    for arm in arms:
        details, initial, final, contract = verify_arm(root, arm, by_name[arm["name"]], protocol, source, views)
        if anchor_initial is not None:
            for key in ("model", "optimizer", "scheduler", "scaler", "sampler", "rng"):
                assert_equal(anchor_initial[key], initial[key], f"cross-arm initial {key}")
            # The weighted initial total changes by design when latent weight
            # changes. Only the common raw RGB probe can be compared directly.
            for key in ("mean_rgb", "per_view_rgb", "scope"):
                assert_equal(anchor_initial["extra"]["probe_before"][key], initial["extra"]["probe_before"][key],
                             f"cross-arm initial probe {key}")
            for key in ("scheduler", "scaler", "sampler"):
                assert_equal(anchor_final[key], final[key], f"cross-arm final {key}")
            for key in ("compatibility", "manifest_sha256", "lineage"):
                require(anchor_contract[key] == contract[key], f"Cross-arm {key} differs")
        else:
            anchor_initial, anchor_final, anchor_contract = initial, final, contract
        reports.append(details)
    return {"arms": reports, "protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
            "executions": str(executions_path), "executions_sha256": file_sha256(executions_path),
            "declared_runner": str(declared_runner), "declared_runner_sha256": protocol["runner_sha256"],
            "orchestration_correction": str(protocol_path.parent / "orchestration_correction.json")
                if (protocol_path.parent / "orchestration_correction.json").is_file() else None,
            "orchestration_correction_sha256": file_sha256(protocol_path.parent / "orchestration_correction.json")
                if (protocol_path.parent / "orchestration_correction.json").is_file() else None,
            "manifest": str(manifest), "manifest_sha256": protocol["manifest_sha256"],
            "only_changed_config_field": "training.losses.latent", "recorded_training_runtime": anchor_contract["compatibility"]["runtime"],
            "implementation_sha256": anchor_contract["compatibility"]["implementation_sha256"],
            "all_initial_model_tensor_bytes_equal": True, "all_initial_optimizer_scheduler_scaler_sampler_rng_equal": True,
            "all_final_schedulers_scalers_samplers_equal": True, "declared_real_optimizer_steps_inspected": 192,
            "clean_views_sampled": 0, "cuda_rng_states_compared_without_initializing_cuda": True}


def verify(protocol, executions, report, *, root=None):
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    protocol_path, executions_path, report_path = (project_path(root, value) for value in (protocol, executions, report))
    require(report_path.is_relative_to(root / "logs") and not report_path.exists(),
            "Use a new explicit report path under project logs; existing evidence is never overwritten")
    started = time.monotonic()
    guard = {"model_constructions": 0, "model_forward_calls": 0, "optimizer_step_calls": 0}
    result = {"status": "incomplete", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "committed_three_arm_initialization_and_sampler_state_verification_only",
              "new_optimizer_steps": 0, "trained_base_accepted": False, "h3_model_loaded": False,
              "global_rng_restored": False, "resumed_training_equivalence_tested": False,
              "limitations": ["Saved state verification does not measure restoration quality",
                  "Sampler trace is reconstructed from its private generator and checked against saved endpoints",
                  "Matching seeds and states do not establish cross-device bitwise training determinism",
                  "No model or optimizer is constructed, and no forward, backward or optimizer update is executed"]}

    def prohibit(counter):
        def reject(*args, **kwargs):
            guard[counter] += 1
            raise RuntimeError(f"Read-only state verification prohibits {counter}")
        return reject

    try:
        with no_training_guard() as training_guard, ExitStack() as extra:
            for owner, method, counter in ((torch.nn.Module, "__init__", "model_constructions"),
                                            (torch.nn.Module, "_call_impl", "model_forward_calls"),
                                            (torch.optim.Optimizer, "step", "optimizer_step_calls"),
                                            (torch.optim.AdamW, "step", "optimizer_step_calls")):
                extra.enter_context(patch.object(owner, method, prohibit(counter)))
            try:
                result.update(inspect(root, protocol_path, executions_path))
            finally:
                guard.update(training_guard)
            require(all(value == 0 for value in guard.values()), "Forbidden learning or forward execution was attempted")
            result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=type(exc).__name__, message=str(exc))
        raise
    finally:
        result.update(execution_guard=guard, elapsed_seconds=time.monotonic() - started,
                      script_sha256=file_sha256(Path(__file__)))
        atomic_write(report_path, canonical_json(result))
        print(json.dumps({"status": result["status"], "report": str(report_path)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="logs/loss-balance-20260907/protocol.json")
    parser.add_argument("--executions", default="logs/loss-balance-20260907/executions.json")
    parser.add_argument("--report", required=True)
    verify(**vars(parser.parse_args()))
