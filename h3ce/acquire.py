"""Explicit acquisition of approved YOLO11 release artifacts.

Only this module performs detector networking. Normal model loading stays offline.
The first acquisition records actual bytes; subsequent runs never bless changed
bytes or replace an existing file whose provenance cannot be verified.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from h3ce.components import (
    _approved_detector_source, component_path, config_dict, read_component_lock,
    sha256_file,
)
from h3ce.errors import H3CEError


_RUNTIME_FILES = (
    "ultralytics/nn/tasks.py", "ultralytics/nn/modules/head.py",
    "ultralytics/nn/modules/block.py", "ultralytics/cfg/models/11/yolo11.yaml",
    "ultralytics/engine/predictor.py", "ultralytics/engine/results.py",
    "ultralytics/utils/__init__.py", "ultralytics/utils/ops.py",
)


def detector_runtime_metadata() -> dict:
    """Pin the installed distribution and relevant sources without importing it."""
    try:
        distribution = importlib.metadata.distribution("ultralytics")
    except importlib.metadata.PackageNotFoundError as exc:
        raise H3CEError("E_DEPENDENCY", "Install an exact Ultralytics version before acquiring detector weights.") from exc
    files = {}
    for relative in _RUNTIME_FILES:
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            raise H3CEError("E_COMPONENT_CODE", "Installed Ultralytics source layout differs from the supported runtime.", {"missing": relative})
        files[relative] = sha256_file(path)
    return {"code_revision": f"ultralytics=={distribution.version}", "runtime_files_sha256": files}


def verify_detector_runtime_sources(entry: dict) -> None:
    expected = entry.get("runtime_files_sha256")
    if expected is None:
        return  # Legacy locks retain exact-version validation in the loader.
    actual = detector_runtime_metadata()
    if actual["code_revision"] != entry["code_revision"] or actual["runtime_files_sha256"] != expected:
        raise H3CEError("E_COMPONENT_CODE", "Installed detector runtime sources differ from the component lock.")


def _approved_transport_url(url: str, entry: dict) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != "https" or parsed.username or parsed.password or port not in (None, 443):
        return False
    if url == entry["source_url"]:
        return True
    # GitHub release downloads redirect to its signed release-asset CDN. Never
    # permit arbitrary redirects, HTTP, userinfo, or different GitHub repositories.
    return parsed.hostname == "release-assets.githubusercontent.com" and parsed.path.startswith("/github-production-release-asset/")


class _ApprovedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, entry: dict):
        self.entry = entry

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _approved_transport_url(newurl, self.entry):
            raise H3CEError("E_COMPONENT_SOURCE", "Release asset redirected outside approved HTTPS hosts.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_artifact(entry: dict, destination: Path) -> dict:
    expected_size = entry.get("source_asset_size_bytes")
    if expected_size is not None and (not isinstance(expected_size, int) or expected_size <= 0):
        raise H3CEError("E_COMPONENT_LOCK", "Release asset size must be a positive integer.")
    opener = urllib.request.build_opener(_ApprovedRedirectHandler(entry))
    request = urllib.request.Request(entry["source_url"], headers={"User-Agent": "H3CE-explicit-component-acquisition/1"})
    # A failed attempt has no committed artifact. Retries always restart the
    # partial file and remain restricted to the same approved source.
    for attempt in range(3):
        try:
            with opener.open(request, timeout=60) as response, destination.open("wb") as output:
                if not _approved_transport_url(response.geturl(), entry):
                    raise H3CEError("E_COMPONENT_SOURCE", "Unexpected final release asset URL.")
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > (expected_size if expected_size else 250 * 1024 * 1024):
                        raise H3CEError("E_COMPONENT_SIZE", "Downloaded detector exceeds its approved release size.")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
                final_host = urlparse(response.geturl()).hostname
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == 2:
                raise H3CEError("E_COMPONENT_DOWNLOAD", "Approved detector download failed; no artifact was committed.", {"reason": str(exc)}) from exc
            time.sleep(attempt + 1)
    if expected_size is not None and count != expected_size:
        raise H3CEError("E_COMPONENT_SIZE", "Release asset byte count differs from approved metadata.", {"expected": expected_size, "actual": count})
    # This checks a container signature without executing the checkpoint pickle.
    if not zipfile.is_zipfile(destination):
        raise H3CEError("E_COMPONENT_FORMAT", "Approved YOLO11 artifact is not a PyTorch ZIP checkpoint.")
    actual_hash = sha256_file(destination)
    for expected in (entry.get("sha256"), (entry.get("source_asset_digest") or "").removeprefix("sha256:")):
        if expected and actual_hash != expected.lower():
            raise H3CEError("E_COMPONENT_HASH", "Release artifact SHA256 differs from the locked digest.")
    return {"sha256": actual_hash, "size_bytes": count, "final_host": final_host}


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(document, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def acquire_detectors(config, project_root: str | Path, *, allow_download: bool = False) -> dict:
    """Acquire and lock approved artifacts only with explicit network consent."""
    if allow_download is not True:
        raise H3CEError("E_DOWNLOAD_DISABLED", "Detector acquisition requires explicit --allow-download.")
    config = config_dict(config)
    root = Path(project_root).resolve()
    lock_path = Path(config["paths"]["components_lock"])
    lock_path = lock_path if lock_path.is_absolute() else root / lock_path
    entries = read_component_lock(lock_path)
    runtime = detector_runtime_metadata()
    plans = []
    for role in ("person", "face"):
        entry = entries.get(role)
        if not entry or not _approved_detector_source(entry, role):
            raise H3CEError("E_COMPONENT_SOURCE", f"The {role} lock does not name an approved versioned release artifact.")
        destination = component_path(entry, root)
        configured = Path(config["data"]["detection"][role]["weights"])
        configured = (configured if configured.is_absolute() else root / configured).resolve()
        if destination != configured or destination.suffix.lower() != ".pt":
            raise H3CEError("E_COMPONENT_LOCK", f"Configured {role} weight path differs from its component lock.")
        if entry.get("code_revision") not in (None, runtime["code_revision"]):
            raise H3CEError("E_COMPONENT_CODE", "Acquisition cannot silently update a previously pinned runtime version.")
        if entry.get("runtime_files_sha256") is not None and entry["runtime_files_sha256"] != runtime["runtime_files_sha256"]:
            raise H3CEError("E_COMPONENT_CODE", "Acquisition cannot silently update previously pinned runtime sources.")
        if destination.exists() and (not entry.get("sha256") or sha256_file(destination) != entry["sha256"].lower()):
            raise H3CEError("E_COMPONENT_HASH", "Existing detector bytes are unverified or changed; acquisition will not overwrite them.", {"path": str(destination)})
        plans.append((role, entry, destination))
    results = []
    for role, original, destination in plans:
        entry = dict(original)
        if destination.is_file():
            result = {"sha256": sha256_file(destination), "size_bytes": destination.stat().st_size, "status": "reused_verified"}
            if entry.get("runtime_files_sha256") == runtime["runtime_files_sha256"]:
                results.append({"id": role, "path": str(destination), **result, "code_revision": runtime["code_revision"]})
                continue  # Reuse must not change timestamps or invalidate detection caches.
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
            os.close(fd)
            partial = Path(temporary)
            try:
                result = {**_download_artifact(entry, partial), "status": "acquired"}
                os.replace(partial, destination)
            finally:
                partial.unlink(missing_ok=True)
        entry.update(runtime)
        entry.update({
            "sha256": result["sha256"],
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition": {"explicit_allow_download": True, **result},
            "verification": "Approved source and local SHA256 verified. Live architecture and bbox acceptance are recorded separately.",
        })
        # Preserve other component families from the most recent lock. Callers
        # must serialize lock writes; the replacement itself is atomic.
        document = json.loads(lock_path.read_text(encoding="utf-8"))
        document["components"] = [entry if item["id"] == role else item for item in document["components"]]
        _atomic_json(lock_path, document)
        results.append({"id": role, "path": str(destination), **result, "code_revision": runtime["code_revision"]})
    return {"stage": "acquire_detectors", "status": "complete", "components": results, "model_acceptance": "not_run"}
