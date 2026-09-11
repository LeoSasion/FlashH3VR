"""Read-only validation of reviewed historical model acceptance evidence.

The registry is a local attestation, not a cryptographic third-party signature.
Registration never runs a model or changes an original acceptance report. It
records the reviewed implementation state so later code/runtime/weight changes
invalidate the attestation instead of inheriting stale GPU claims.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from h3ce.acquire import _atomic_json, verify_detector_runtime_sources
from h3ce.components import canonical_hash, component_inventory, config_dict, sha256_file
from h3ce.errors import H3CEError


REGISTRY_PATH = "acceptance.registry.json"
IMPLEMENTATION_PATHS = {
    "detectors": ("h3ce/data/detect_yolo11.py", "h3ce/acquire.py", "h3ce/components.py", "scripts/verify_detectors.py"),
    "h3": ("h3ce/vae/_vendor.py", "h3ce/vae/aitoolkit_h3_backend.py", "h3ce/vae/bridge.py", "h3ce/vae/decoder_adapter.py", "scripts/verify_h3.py"),
}
COMPONENT_IDS = {"detectors": ("person", "face"), "h3": ("h3_visual_vae", "aitoolkit_h3_backend")}
RUNTIME_PACKAGES = {
    "detectors": ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "opencv-python"),
    "h3": ("torch", "safetensors", "numpy", "Pillow", "av"),
}
_COMPONENT_FIELDS = ("id", "provider", "source_url", "revision", "local_path", "sha256", "architecture", "code_revision", "runtime_files_sha256", "upstream_source_sha256")
_H3_CASES = {
    "real_image_256_roundtrip_and_upstream_parity",
    "real_image_frozen_decoder_pixel_gradient", "real_image_384_native_spatial_tiling",
    "real_video_5_frames_roundtrip", "real_video_22_frames_roundtrip", "real_video_39_frames_roundtrip",
    "real_video_3_frames_roundtrip", "real_video_5_frozen_decoder_pixel_gradient",
    "actual_decoder_last_four_projection_gradients_and_cancellation",
}


def _local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Evidence paths must be relative and remain inside the project")
    return path


def _runtime_versions(kind):
    return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES[kind]}


def _component_fingerprint(entry):
    return canonical_hash({field: entry.get(field) for field in _COMPONENT_FIELDS})


def _read_report(kind, report_path, entries, runtime):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or not report.get("cases"):
        raise ValueError("Report is not a nonempty passed actual-model acceptance")
    if kind == "detectors":
        if report.get("stage") != "real_yolo11_detector_acceptance" or report.get("device") == "cpu" or report.get("cuda", {}).get("available") is not True:
            raise ValueError("Report does not establish actual GPU YOLO11 acceptance")
        if any(case.get("passed") is not True or not case.get("person_boxes") or not case.get("face_boxes") for case in report["cases"]):
            raise ValueError("Detector report contains failed or incomplete cases")
        for role in COMPONENT_IDS[kind]:
            if _component_fingerprint(report["components"][role]) != _component_fingerprint(entries[role]):
                raise ValueError(f"Detector report uses different {role} component provenance")
            classes = report.get("validated_classes", {}).get(role, {})
            if classes.get("names", {}).get(str(classes.get("class_id"))) != role:
                raise ValueError("Detector report lacks verified role-specific classes")
            verify_detector_runtime_sources(entries[role])
        for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow"):
            if report.get("runtime", {}).get(name) != runtime[name]:
                raise ValueError(f"Detector report runtime differs: {name}")
    else:
        cases = {case.get("case"): case for case in report["cases"]}
        if not _H3_CASES <= cases.keys() or any(case.get("status") != "passed" for case in report["cases"]) or report.get("missing"):
            raise ValueError("H3 report lacks required successful real image/video/gradient cases")
        native = report["contract"]["native"]
        if native["weights_sha256"] != entries["h3_visual_vae"]["sha256"] or native["portable_source_sha256"] != entries["aitoolkit_h3_backend"]["sha256"]:
            raise ValueError("H3 report weight/backend hashes differ from current components")
        if native["upstream_revision"] != entries["aitoolkit_h3_backend"]["revision"] or native["upstream_sha256"] != entries["aitoolkit_h3_backend"].get("upstream_source_sha256"):
            raise ValueError("H3 upstream source provenance differs from accepted implementation")
        if report.get("torch") != runtime["torch"] or report.get("python") != platform.python_version() or not report.get("cuda_runtime") or not report.get("gpu", {}).get("name"):
            raise ValueError("H3 report runtime differs or GPU evidence is missing")
        if report["contract"].get("frame_mappings") != {"1": 1, "5": 2, "22": 7, "39": 12} or not report.get("module_tree"):
            raise ValueError("H3 report lacks native module/frame geometry evidence")
        if not isinstance(native.get("storage_dtypes"), dict) or not native["storage_dtypes"] or any(not isinstance(native.get(name), str) or not native[name] for name in ("normalization_source", "precision")) or not isinstance(report.get("scope"), str):
            raise ValueError("H3 report lacks precision/normalization/scope evidence")
    return report


def _configured_paths_match(kind, config, root, entries):
    config = config_dict(config)
    for component_id in COMPONENT_IDS[kind]:
        if component_id == "aitoolkit_h3_backend":
            continue
        configured = config["vae"]["weights"] if kind == "h3" else config["data"]["detection"][component_id]["weights"]
        if (root / configured).resolve() != (root / entries[component_id]["local_path"]).resolve():
            raise ValueError(f"Configured {component_id} path differs from accepted component")


def build_acceptance_registry(config, root: Path, report_paths: dict[str, str]) -> dict:
    """Explicitly register reviewed successful reports against current local state."""
    root = root.resolve()
    inventory = component_inventory(config, root)
    entries = {item["id"]: item for item in inventory["components"]}
    records = {}
    for kind, relative in report_paths.items():
        if kind not in COMPONENT_IDS:
            raise H3CEError("E_ACCEPTANCE_EVIDENCE", "Unknown acceptance family")
        if any(entries.get(name, {}).get("status") != "ready" for name in COMPONENT_IDS[kind]):
            raise H3CEError("E_ACCEPTANCE_EVIDENCE", "Cannot register acceptance for missing or unverified components")
        try:
            _configured_paths_match(kind, config, root, entries)
            report_path = _local_path(root, relative)
            runtime = _runtime_versions(kind)
            _read_report(kind, report_path, entries, runtime)
            records[kind] = {
                "report_path": relative, "report_sha256": sha256_file(report_path),
                "implementation_hashes": {path: sha256_file(_local_path(root, path)) for path in IMPLEMENTATION_PATHS[kind]},
                "component_fingerprints": {name: _component_fingerprint(entries[name]) for name in COMPONENT_IDS[kind]},
                "runtime_versions": runtime, "python": platform.python_version(),
                "evidence_type": "reviewed_post_run_registration", "gpu_execution_performed_by_registration": False,
            }
        except (KeyError, TypeError, ValueError, OSError, AttributeError, importlib.metadata.PackageNotFoundError, H3CEError) as exc:
            raise H3CEError("E_ACCEPTANCE_EVIDENCE", "Cannot register inconsistent acceptance evidence", {"kind": kind, "reason": str(exc)}) from exc
    return {"schema_version": 1, "registered_at_utc": datetime.now(timezone.utc).isoformat(), "reports": records}


def write_acceptance_registry(config, root: Path, report_paths: dict[str, str]) -> dict:
    registry = build_acceptance_registry(config, root, report_paths)
    _atomic_json(root / REGISTRY_PATH, registry)
    return registry


def inspect_acceptance(config, root: Path, inventory: dict) -> dict:
    """Verify historical evidence using only files/metadata; never import a model."""
    root = root.resolve()
    result = {kind: {"status": "blocked", "reason": "missing_acceptance_registry", "historical": True, "gpu_test_run_by_doctor": False} for kind in COMPONENT_IDS}
    registry_path = root / REGISTRY_PATH
    if not registry_path.is_file():
        return result
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1 or not isinstance(registry.get("reports"), dict):
            raise ValueError("Unsupported acceptance registry schema")
    except (OSError, ValueError, AttributeError) as exc:
        return {kind: {**item, "reason": "invalid_acceptance_registry", "detail": str(exc)} for kind, item in result.items()}
    entries = {item["id"]: item for item in inventory.get("components", [])}
    for kind, record in registry["reports"].items():
        if kind not in COMPONENT_IDS:
            continue
        try:
            if any(entries.get(name, {}).get("status") != "ready" for name in COMPONENT_IDS[kind]):
                raise ValueError("Current component files or locks are missing/unverified")
            _configured_paths_match(kind, config, root, entries)
            path = _local_path(root, record["report_path"])
            if sha256_file(path) != record["report_sha256"]:
                raise ValueError("Acceptance report content changed")
            hashes = record["implementation_hashes"]
            if set(hashes) != set(IMPLEMENTATION_PATHS[kind]):
                raise ValueError("Implementation fingerprint coverage is incomplete")
            if any(sha256_file(_local_path(root, file)) != expected for file, expected in hashes.items()):
                raise ValueError("Accepted implementation source changed")
            if record["component_fingerprints"] != {name: _component_fingerprint(entries[name]) for name in COMPONENT_IDS[kind]}:
                raise ValueError("Accepted component source/hash/code lock changed")
            runtime = _runtime_versions(kind)
            if runtime != record["runtime_versions"] or platform.python_version() != record["python"]:
                raise ValueError("Accepted execution runtime changed")
            report = _read_report(kind, path, entries, runtime)
            result[kind].update({"status": "passed", "reason": None, "report_path": str(path), "report_sha256": record["report_sha256"], "evidence_type": record["evidence_type"], "report": report})
        except (KeyError, TypeError, ValueError, OSError, AttributeError, importlib.metadata.PackageNotFoundError, H3CEError) as exc:
            result[kind].update({"reason": "stale_or_invalid_acceptance_evidence", "detail": str(exc)})
    return result
