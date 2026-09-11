"""Read-only component/environment diagnostics, never a simulated H3 acceptance."""
from __future__ import annotations

import importlib.metadata
from copy import deepcopy
import platform
import shutil
import subprocess
from pathlib import Path

from h3ce.components import component_inventory
from h3ce.acceptance import inspect_acceptance
from h3ce.overfit_acceptance import inspect_overfit
from h3ce.errors import H3CEError


def inventory_or_error(config, root):
    try:
        return component_inventory(config, root)
    except H3CEError as exc:
        return {"ready": False, "components": [], "failure": exc.as_dict()}


def resolve_optional_losses(config, inventory, root=None):
    config = config.model_copy(deep=True)
    lpips = next((item for item in inventory.get("components", []) if "lpips" in item["id"].lower()), None)
    notes = []
    available = False
    if lpips and lpips['status'] == 'ready' and config.training.losses.perceptual:
        from h3ce.train.perceptual import verify_perceptual_entry
        # A present calibration alone must not enable an unverified backbone/runtime.
        verify_perceptual_entry(lpips, Path(root).resolve() if root is not None else Path(inventory['lock_path']).parent)
        available = True
    if config.training.losses.perceptual and not available:
        config.training.losses.perceptual = 0.0
        notes.append({"field": "training.losses.perceptual", "effective_value": 0.0,
            "reason": "Optional locked LPIPS-Alex is unavailable; metric disabled"})
    return config, notes


def diagnose(config, root: Path, inventory=None):
    inventory = inventory if inventory is not None else inventory_or_error(config, root)
    dependencies = {}
    for package in ("h3ce", "pydantic", "PyYAML", "numpy", "Pillow", "torch", "torchvision", "ultralytics", "av", "safetensors"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    gpu = {"inventory": None, "h3_gpu_test": "not_run", "performance": None}
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            result = subprocess.run([binary, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=False)
            gpu["inventory"] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    evidence = inspect_acceptance(config, root, inventory)
    detector_passed = evidence["detectors"]["status"] == "passed"
    h3_passed = evidence["h3"]["status"] == "passed"
    overfit = inspect_overfit(config, root)
    if overfit["status"] == "passed" and not (detector_passed and h3_passed):
        overfit = {**overfit, "status": "blocked", "reason": "current_M0_M1_acceptance_required"}
    overfit_passed = overfit["status"] == "passed"
    inventory = deepcopy(inventory)
    for item in inventory.get("components", []):
        if item["id"] in {"person", "face"} and detector_passed:
            item["model_acceptance"] = "passed_historical_real_yolo11_gpu"
        elif item["id"] in {"h3_visual_vae", "aitoolkit_h3_backend"} and h3_passed:
            item["model_acceptance"] = "passed_historical_real_h3_gpu"
    h3 = {"native_module_tree": None, "dtype": None, "normalization_source": None,
          "legal_shapes_verified": False, "roundtrip": "not_run", "pixel_gradients": "not_run"}
    if h3_passed:
        report = evidence["h3"]["report"]
        native = report["contract"]["native"]
        h3.update({"native_module_tree": report["module_tree"], "dtype": native["storage_dtypes"],
                   "precision": native["precision"], "normalization_source": native["normalization_source"],
                   "legal_shapes_verified": True, "frame_mappings": report["contract"]["frame_mappings"],
                   "roundtrip": "passed_historical_verified", "pixel_gradients": "passed_historical_verified",
                   "historical_report": evidence["h3"]["report_path"], "gpu_test_run_by_doctor": False,
                   "scope": report["scope"], "weights_sha256": native["weights_sha256"]})
        gpu["h3_gpu_test"] = "passed_historical_verified"
        gpu["historical_test_gpu"] = report["gpu"]["name"]
        gpu["historical_peak_allocated_bytes"] = report.get("vram_peak_allocated_bytes")
    return {"status": "blocked", "blockers": (["M0_real_detector_acceptance_required"] if not detector_passed else [])
            + (["M1_real_h3_acceptance_required"] if not h3_passed else [])
            + ["M2_useful_base_acceptance_required" if overfit_passed else "M2_overfit_and_useful_base_acceptance_required"],
        "project_root": str(root), "python": platform.python_version(),
        "platform": platform.platform(), "dependencies": dependencies, "components": inventory,
        "gpu": gpu, "milestones": {
            "M0": {"implementation": "implemented", "capabilities": ["strict_config", "explicit_weight_acquisition", "yolo11_person_face", "image_video_pts_preparation", "shot_geometry_tracking", "deterministic_degradation", "layered_cache"],
                   "acceptance": "passed_real_yolo11_gpu" if detector_passed else "blocked_real_yolo11_evidence_required",
                   "acceptance_scope": "Historical locked detector architecture, classes and real-image bbox smoke tests; data coverage remains dataset-specific"},
            "M1": {"implementation": "implemented", "acceptance": "passed_real_h3_gpu" if h3_passed else "blocked_real_h3_evidence_required",
                   "acceptance_scope": "Historical real pretrained H3 image/video roundtrip and gradient tests; no restoration training"},
            "M2": {"implementation": "image_bootstrap_implemented",
                   "acceptance": "overfit_passed_useful_base_pending" if overfit_passed else "not_registered",
                   "overfit_probe": overfit, "useful_base_accepted": False,
                   "capabilities": ["spatial_refiner", "scene_cross_attention", "cached_image_loader", "restoration_losses",
                                    "forward_only_check", "overfit_latent_pixel_phases", "atomic_resume", "shared_time_budget"],
                   "limitations": ["requires_materialized_image_training_manifest", "video_training_loader_not_implemented",
                                   "useful_trained_base_not_registered"]}},
        "h3": h3,
        "acceptance_evidence": {kind: {key: value for key, value in item.items() if key != "report"} for kind, item in evidence.items()},
        "unsupported": ["aligned pairs", "h264 compression", "lazy variant materialization", "automatic cache quota enforcement",
                        "automatic_training_cache_materialization", "video_restoration_training", "character_training",
                        "restoration_inference/evaluation/export", "V1 configuration migration"],
        "training_allowed": False}
