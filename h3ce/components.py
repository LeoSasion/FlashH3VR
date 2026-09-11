"""Offline component inventory and validation before external weight loading.

A manifest entry describes an approved artifact; it does not prove that artifact
has been obtained. Null hashes and revisions intentionally keep it non-loadable.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3ce.errors import H3CEError


REQUIRED_FIELDS = {
    "id", "provider", "source_url", "revision", "local_path", "sha256",
    "architecture", "code_revision", "license_source",
}


def config_dict(config: Any) -> dict:
    return config.model_dump(mode="json") if hasattr(config, "model_dump") else config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_component_lock(path: str | Path) -> dict[str, dict]:
    """Read a versioned JSON lock without loading any component code or weights."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise H3CEError("E_COMPONENT_LOCK", "Cannot read the component lock.", {"path": str(path), "reason": str(exc)}) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("components"), list):
        raise H3CEError("E_COMPONENT_LOCK", "Expected schema_version 1 and a components list.")
    result = {}
    for entry in document["components"]:
        if not isinstance(entry, dict) or not REQUIRED_FIELDS <= entry.keys():
            raise H3CEError("E_COMPONENT_LOCK", "A component entry lacks required lock fields.")
        component_id = entry["id"]
        if not isinstance(component_id, str) or not component_id or component_id in result:
            raise H3CEError("E_COMPONENT_LOCK", "Component IDs must be unique nonempty strings.")
        for field in REQUIRED_FIELDS - {"license_source"}:
            if entry[field] is not None and not isinstance(entry[field], str):
                raise H3CEError("E_COMPONENT_LOCK", f"{component_id}.{field} must be a string or null.")
        if not isinstance(entry["license_source"], (dict, str)):
            raise H3CEError("E_COMPONENT_LOCK", f"{component_id}.license_source must retain license sources.")
        if entry["sha256"] is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", entry["sha256"]):
            raise H3CEError("E_COMPONENT_LOCK", f"{component_id}.sha256 is not a SHA256 digest.")
        result[component_id] = dict(entry)
    return result


def component_path(entry: dict, project_root: str | Path) -> Path | None:
    path = entry.get("local_path")
    if not path:
        return None
    local = Path(path)
    return (local if local.is_absolute() else Path(project_root) / local).resolve()


def _approved_detector_source(entry: dict, role: str) -> bool:
    parsed = urlparse(entry.get("source_url") or "")
    if parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.query or parsed.fragment:
        return False
    revision = entry.get("revision")
    if not revision or "/" in revision or revision in {"latest", "main", "master", "dev"}:
        return False
    if role == "person":
        return entry.get("provider") == "ultralytics" and bool(re.fullmatch(
            rf"/ultralytics/assets/releases/download/{re.escape(revision)}/yolo11[nsmlx]\.pt", parsed.path
        ))
    if entry.get("provider") == "akanametov/yolo-face":
        return bool(re.fullmatch(
            rf"/akanametov/yolo-face/releases/download/{re.escape(revision)}/yolov11[nsmlx]-face\.pt", parsed.path
        ))
    # The alternative in S5 is approved as a source, but still must pass the same
    # live architecture/API contract after loading. No family fallback occurs.
    return entry.get("provider") == "zjykzj/YOLO11Face" and bool(re.fullmatch(
        rf"/zjykzj/YOLO11Face/releases/download/{re.escape(revision)}/[^/]+\.pt", parsed.path
    ))


def inspect_component(entry: dict, project_root: str | Path) -> dict:
    """Compute honest readiness; checking a hash is not a real-model acceptance test."""
    path = component_path(entry, project_root)
    blockers = []
    for field in ("revision", "sha256", "architecture", "code_revision", "source_url", "local_path"):
        if not entry.get(field):
            blockers.append(f"unresolved_{field}")
    if path is None or not path.is_file():
        blockers.append("missing_local_file")
    actual_hash = None
    if path is not None and path.is_file():
        actual_hash = sha256_file(path)
        if entry.get("sha256") and actual_hash != entry["sha256"].lower():
            blockers.append("sha256_mismatch")
    role = entry["id"] if entry["id"] in {"person", "face"} else None
    if role and not _approved_detector_source(entry, role):
        blockers.append("unapproved_detector_source")
    return {
        **entry, "resolved_local_path": str(path) if path else None,
        "actual_sha256": actual_hash, "status": "ready" if not blockers else "blocked",
        "blockers": blockers, "model_acceptance": "not_run",
    }


def require_locked_component(entry: dict, project_root: str | Path, *, role: str, configured_weights: str | Path) -> Path:
    """Verify provenance, pinning and bytes *before* any unsafe .pt deserialization."""
    code = "E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS"
    inspected = inspect_component(entry, project_root)
    if inspected["blockers"]:
        raise H3CEError(code, f"{role} weights are not locked and locally verified; downloads are disabled.", {"component": role, "blockers": inspected["blockers"]})
    path = component_path(entry, project_root)
    expected_path = Path(configured_weights)
    expected_path = (expected_path if expected_path.is_absolute() else Path(project_root) / expected_path).resolve()
    if path != expected_path or path.suffix.lower() != ".pt":
        raise H3CEError(code, "Configured weights must match the locked local .pt file.", {"configured": str(expected_path), "locked": str(path)})
    if not _approved_detector_source(entry, role):
        raise H3CEError(code, "Detector artifact source is not approved by the project contract.")
    expected_suffix = "face" if role == "face" else "coco"
    if not re.fullmatch(rf"yolo11[nsmlx]-detect-{expected_suffix}", entry["architecture"]):
        raise H3CEError(code, f"The {role} lock must name the appropriate YOLO11 detection architecture.")
    if not re.fullmatch(r"ultralytics==[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?", entry["code_revision"]):
        raise H3CEError("E_COMPONENT_CODE", "Detector code_revision must pin an exact ultralytics distribution version.")
    return path


def component_inventory(config: Any, project_root: str | Path) -> dict:
    config = config_dict(config)
    lock_path = Path(config["paths"]["components_lock"])
    lock_path = (lock_path if lock_path.is_absolute() else Path(project_root) / lock_path).resolve()
    entries = read_component_lock(lock_path)
    inspected = [inspect_component(entry, project_root) for entry in entries.values()]
    required = {"person", "face", "h3_visual_vae", "aitoolkit_h3_backend"}
    missing = sorted(required - entries.keys())
    return {
        "lock_path": str(lock_path), "lock_sha256": sha256_file(lock_path),
        "ready": not missing and all(item["status"] == "ready" for item in inspected if item["id"] in required),
        "missing_required_components": missing, "components": inspected,
    }
