"""Local attestation of a real image overfit probe, never a useful-base release.

Explicit registration checks the actual checkpoint payload. Historical inspection
checks the registered bytes and current contracts without importing torch or
loading models. Like the M0/M1 registry, this is not a third-party signature.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace

from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.config import load_config
from h3ce.errors import H3CEError
from h3ce.train.overfit_data import overfit_probe_passed, summarize_probe, validate_overfit_dataset


REGISTRY_PATH = "overfit.acceptance.registry.json"
SCOPE = "real_image_overfit_probe_only_not_useful_base_acceptance"
_ROLES = {"report", "contract", "request", "resolved_config", "manifest",
          "checkpoint", "checkpoint_receipt", "restore_report"}
_STATE_NAMES = {"model", "optimizer", "scheduler", "scaler", "sampler", "rng"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, value):
    _require(isinstance(value, (str, Path)), "Evidence path must be text")
    path = root / value
    assert_no_links(path)
    resolved = path.resolve()
    _require(resolved.is_relative_to(root) and resolved != root, "Evidence path escapes project")
    return resolved


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "Evidence document must be an object")
    return value


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _config_contract(config):
    """Learning schedules may differ from an explicitly selected historical probe."""
    document = config.model_dump(mode="json")
    return {**{key: document[key] for key in ("backend", "vae", "native_temporal", "data", "cache", "model")},
            "paths": {key: document["paths"][key] for key in ("raw", "tmp", "models", "runs", "components_lock")},
            "training": {key: value for key, value in document["training"].items() if key != "stages"},
            "seed": document["project"]["seed"]}


def _implementation_paths(root):
    files = [root / "h3ce/cli.py", root / "h3ce/config.py"]
    for directory in ("h3ce/model", "h3ce/train"):
        files.extend((root / directory).glob("*.py"))
    return {path.relative_to(root).as_posix() for path in files}


def _validate_documents(config, root, paths):
    _require(set(paths) == _ROLES, "Evidence fingerprint coverage is incomplete")
    report, contract = _json(paths["report"]), _json(paths["contract"])
    request, receipt = _json(paths["request"]), _json(paths["checkpoint_receipt"])
    restore = _json(paths["restore_report"])
    trained = load_config(paths["resolved_config"])
    trained_json = trained.model_dump(mode="json")
    _require(_config_contract(config) == _config_contract(trained), "Current execution/data configuration differs from selected training configuration")
    _require(trained.native_temporal.mode == "frozen" and trained.native_temporal.decoder_adapter is None,
             "Only image probes using the frozen H3 VAE may be registered")
    _require(not trained.lora.routing.adapters and trained.model.architecture_id == "h3ce_spatial_refiner_v2",
             "Probe must bootstrap the shared spatial refiner without character adapters")
    _require(digest(trained_json) == contract["resolved_sha256"], "Training resolved configuration differs from checkpoint contract")
    compatibility = contract["compatibility"]
    _require(compatibility["architecture"] == trained_json["model"] and compatibility["data"] == trained_json["data"],
             "Training compatibility disagrees with resolved data/model configuration")
    _require(file_sha256(_path(root, trained.paths.components_lock)) == compatibility["components_sha256"],
             "Training component lock changed")
    implementation = compatibility["implementation_sha256"]
    _require(set(implementation) == _implementation_paths(root), "Training implementation coverage changed or is incomplete")
    for relative, expected in implementation.items():
        _require(file_sha256(_path(root, relative)) == expected, f"Training implementation changed: {relative}")
    _require(compatibility.get("runtime", {}).get("gpu") and compatibility["runtime"].get("cuda_runtime"),
             "Probe lacks actual CUDA execution provenance")
    step = report.get("optimizer_steps")
    _require(type(step) is int and step >= 2 and contract["phase"] == report["phase"] == "overfit"
             and contract["max_steps"] == step, "Probe must complete at least two optimizer steps")
    _require(report.get("status") == "passed_overfit_probe" and report.get("training_started") is True
             and report.get("vae_parameters_unchanged") is True and report.get("trained_base_accepted") is False,
             "Report is not a passed real frozen-VAE image probe")
    _require(request.get("stage") == "bootstrap" and request.get("phase") == "overfit" and request.get("check_only") is False,
             "Forward-only checks cannot be registered as training")
    _require(_path(root, request["manifest"]) == paths["manifest"], "Request references a different manifest")
    manifest_hash = file_sha256(paths["manifest"])
    _require(manifest_hash == report["manifest_sha256"] == contract["manifest_sha256"], "Training manifest changed")
    rows = [json.loads(line) for line in paths["manifest"].read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), "Malformed image manifest")
    indexed = {}
    for kind, field in (("source_asset", "asset_id"), ("prepared_target", "target_id"),
                        ("degraded_variant", "variant_id"), ("training_view", "view_id")):
        selected = [row for row in rows if row.get("record_type") == kind]
        indexed[kind] = {row[field]: row for row in selected}
        _require(len(indexed[kind]) == len(selected), f"Duplicate {kind} identifiers")
    _require(sum(len(group) for group in indexed.values()) == len(rows), "Unexpected manifest record type")
    sources, targets, variants = (indexed[kind] for kind in ("source_asset", "prepared_target", "degraded_variant"))
    views = [row for row in rows if row.get("record_type") == "training_view"]
    counts = validate_overfit_dataset(SimpleNamespace(views=views, sources=sources, targets=targets, variants=variants))
    for view in views:
        source = sources[view["asset_id"]]
        _require(source["kind"] == view["media_kind"] == "image" and source["pts"] == []
                 and source["split"] == view["split"] == "train"
                 and view["encoder_contract_id"] == compatibility["encoder_contract_id"],
                 "Probe is not a consistent still-image training subset")
    evidence = report["extra"]["overfit_evidence"]
    _require(report["extra"].get("phase_complete") is True and evidence.get("passed") is True,
             "Probe phase/evidence is incomplete")
    _require(set(evidence["view_ids"]) == {view["view_id"] for view in views}
             and len(evidence["view_ids"]) == len(views), "Probe evidence identifies different views")
    _require(all(_positive(evidence["gradients"].get(key)) for key in ("scene", "spatial"))
             and _positive(evidence.get("pixel_gradient_max")), "Probe lacks nonzero scene/spatial/RGB gradient evidence")
    before, after = evidence["before"], evidence["after"]
    for probe in (before, after):
        if "by_pair_kind" in probe:
            expected = summarize_probe(probe["per_view_total"], probe["per_view_rgb"],
                ["clean" if variants[view["variant_id"]]["clean_pair"] else "degraded" for view in views],
                [view["asset_id"] for view in views], [view["view_id"] for view in views])
            _require(all(probe.get(key) == value for key, value in expected.items()),
                     "Pair-kind probe metrics/counts do not reproduce their declared manifest and per-view losses")
        else:
            _require(counts["clean"] == 0 and "counts" not in probe,
                     "Clean replay requires explicit per-kind probe evidence")
    _require(all(_positive(group.get(key)) for group in (before, after) for key in ("mean_total", "mean_rgb"))
             and overfit_probe_passed(step=step, gradients=evidence["gradients"],
                 pixel_gradient_max=evidence["pixel_gradient_max"], before=before, after=after),
             "Probe did not satisfy the degraded-pair numeric loss gate")
    checkpoint = paths["checkpoint"]
    run = paths["report"].parent
    _require(run.is_relative_to(_path(root, trained.paths.runs)) and checkpoint.parent == run / "checkpoints"
             and all(paths[key].parent == run for key in ("contract", "request", "resolved_config")),
             "Training evidence or checkpoint belongs to a different run")
    _require(_path(root, report["checkpoint"]) == checkpoint and paths["checkpoint_receipt"] == checkpoint.with_suffix(".json"),
             "Report/receipt references a different checkpoint")
    checkpoint_hash = file_sha256(checkpoint)
    _require(receipt["schema_version"] == 1 and receipt["filename"] == checkpoint.name
             and receipt["sha256"] == checkpoint_hash and receipt["bytes"] == checkpoint.stat().st_size
             and receipt["contract_id"] == digest(contract) and receipt["stage"] == "overfit" and receipt["step"] == step,
             "Checkpoint receipt does not match the committed bytes and contract")
    _require(restore.get("status") == "passed" and restore.get("scope") == "real_checkpoint_load_state_equality_only"
             and restore.get("source_overfit_status") == "passed_overfit_probe"
             and restore.get("original_real_optimizer_steps") == step and restore.get("new_optimizer_steps") == 0
             and restore.get("checkpoint_sha256") == checkpoint_hash and restore.get("checkpoint_contract_id") == digest(contract)
             and _path(root, restore["source_checkpoint"]) == checkpoint,
             "Independent restore report does not verify this trained checkpoint")
    comparisons = restore["exact_state_comparisons"]
    _require(set(comparisons) == _STATE_NAMES and all(comparisons[key].get("exact_equal") is True for key in _STATE_NAMES)
             and _positive(restore.get("optimizer_state_entries")) and restore.get("saved_optimizer_step_values") == [float(step)]
             and restore.get("execution_guard") == {"backward_attempts": 0, "autograd_grad_attempts": 0, "optimizer_step_attempts": 0},
             "Independent state restoration evidence is incomplete")
    return report, contract, trained, len(views)


def write_overfit_registry(config, root, run_path, restore_report_path):
    """Explicit registration reads the actual checkpoint; never executes a model."""
    root = Path(root).resolve()
    try:
        run = _path(root, run_path)
        report = _json(run / "training_report.json")
        request = _json(run / "training_request.json")
        paths = {"report": run / "training_report.json", "contract": run / "training_contract.json",
                 "request": run / "training_request.json", "resolved_config": run / "resolved.yaml",
                 "manifest": _path(root, request["manifest"]), "checkpoint": _path(root, report["checkpoint"]),
                 "restore_report": _path(root, restore_report_path)}
        paths["checkpoint_receipt"] = paths["checkpoint"].with_suffix(".json")
        paths = {key: _path(root, value) for key, value in paths.items()}
        report, contract, trained, view_count = _validate_documents(config, root, paths)
        # Keep this import inside the explicit writer. doctor calls only inspect.
        from h3ce.train.checkpoint import CheckpointManager
        manager = CheckpointManager(root / trained.paths.runs, run, contract=contract)
        payload = manager.read(paths["checkpoint"])
        _require(payload["stage"] == "overfit" and payload["step"] == report["optimizer_steps"]
                 and payload["extra"] == report["extra"] and payload["resolved_config"] == trained.model_dump(mode="json"),
                 "Checkpoint payload differs from reported training evidence")
        _require(isinstance(payload["model"], dict) and payload["model"] and payload["optimizer"]
                 and payload["optimizer"].get("state"), "Checkpoint lacks actual model/optimizer state")
        steps = {float(item["step"]) for item in payload["optimizer"]["state"].values()}
        _require(steps == {float(payload["step"])}, "Optimizer state does not reflect all recorded steps")
        _require(payload["sampler"]["length"] == view_count, "Checkpoint sampler uses a different image subset")
        registry = {"schema_version": 1, "kind": "real_image_overfit_probe", "scope": SCOPE,
                    "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                    "optimizer_steps": payload["step"], "useful_base_accepted": False,
                    "character_export_unlocked": False, "payload_checked_at_registration": True,
                    "evidence": {key: {"path": value.relative_to(root).as_posix(), "sha256": file_sha256(value)}
                                 for key, value in paths.items()}}
        atomic_write(_path(root, REGISTRY_PATH), canonical_json(registry))
        return registry
    except (KeyError, TypeError, ValueError, OSError, AttributeError, H3CEError) as exc:
        raise H3CEError("E_OVERFIT_ACCEPTANCE", "Cannot register inconsistent image overfit evidence", {"reason": str(exc)}) from exc


def inspect_overfit(config, root):
    """Inspect file hashes and JSON/YAML only; never import torch or load models."""
    root = Path(root).resolve()
    result = {"status": "not_registered", "reason": "missing_overfit_acceptance_registry",
              "report_path": None, "optimizer_steps": 0, "scope": SCOPE,
              "historical": True, "gpu_test_run_by_doctor": False, "useful_base_accepted": False,
              "character_export_unlocked": False, "requires_current_m0_m1_acceptance": True}
    try:
        registry_path = _path(root, REGISTRY_PATH)
        if not registry_path.is_file():
            return result
        registry = _json(registry_path)
        _require(registry.get("schema_version") == 1 and registry.get("kind") == "real_image_overfit_probe"
                 and registry.get("scope") == SCOPE and registry.get("useful_base_accepted") is False
                 and registry.get("character_export_unlocked") is False and registry.get("payload_checked_at_registration") is True,
                 "Invalid image overfit attestation schema or scope")
        _require(set(registry["evidence"]) == _ROLES, "Evidence fingerprint coverage is incomplete")
        paths = {}
        for key, entry in registry["evidence"].items():
            _require(not Path(entry["path"]).is_absolute(), "Registered paths must be project-relative")
            paths[key] = _path(root, entry["path"])
            _require(file_sha256(paths[key]) == entry["sha256"], f"Registered evidence changed: {key}")
        report, _, trained, count = _validate_documents(config, root, paths)
        _require(registry["optimizer_steps"] == report["optimizer_steps"], "Registry step count changed")
        result.update(status="passed", reason=None, report_path=str(paths["report"]),
                      report_sha256=registry["evidence"]["report"]["sha256"], optimizer_steps=report["optimizer_steps"],
                      selected_training_config=str(paths["resolved_config"]),
                      recorded_pixel_learning_rate=trained.training.stages.bootstrap_pixel.lr,
                      pair_count=count, checkpoint=str(paths["checkpoint"]),
                      state_recovery_scope="load_state_equality_not_resumed_training_equivalence")
    except (KeyError, TypeError, ValueError, OSError, AttributeError, H3CEError) as exc:
        result.update(status="blocked", reason="stale_or_invalid_overfit_evidence", detail=str(exc))
    return result
