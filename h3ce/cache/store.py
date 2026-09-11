"""Atomic, checksummed cache with manifest pins and conservative cleanup."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from h3ce.errors import H3CEError
from .keys import canonical_json, file_sha256
from .locks import FileLock

STAGES = {"working_targets", "detections", "variants", "views", "latents", "decoded", "reports"}
MARKER = ".h3ce-cache-root"
MARKER_CONTENT = "h3ce-cache-v2\n"


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def assert_separate_paths(*paths: Path):
    resolved = [Path(path).resolve() for path in paths]
    for i, path in enumerate(resolved):
        for other in resolved[i + 1:]:
            if within(path, other) or within(other, path):
                raise H3CEError("E_PATH_CONTRACT", f"Protected directories overlap: {path} / {other}")


def assert_no_links(path: Path):
    path = Path(os.path.abspath(path))
    for item in [*path.parents, path]:
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise H3CEError("E_CACHE_PATH", f"Symlink/junction is not allowed in cache paths: {item}")


def atomic_write(path: Path, payload: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CacheStore:
    def __init__(self, root: Path, *, protected=(), create: bool = True):
        self.root = Path(os.path.abspath(root))
        assert_no_links(self.root)
        assert_separate_paths(self.root, *protected)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / MARKER
        if not marker.exists() and create:
            if any(self.root.iterdir()):
                raise H3CEError("E_CACHE_ROOT", "Refusing to mark a nonempty unrecognized cache root.")
            atomic_write(marker, MARKER_CONTENT.encode())
        self._check_root()
        for stage in STAGES | {"locks"}:
            self._safe(self.root / stage)
            if create:
                (self.root / stage).mkdir(exist_ok=True)
        self._safe(self.root / "index.sqlite")
        if not create and not (self.root / "index.sqlite").is_file():
            raise H3CEError("E_CACHE_ROOT", "Cache index does not exist.")
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY, stage TEXT NOT NULL, path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pins (
                    manifest TEXT NOT NULL, key TEXT NOT NULL,
                    PRIMARY KEY (manifest, key));
            """)

    def _check_root(self):
        assert_no_links(self.root)
        marker = self.root / MARKER
        assert_no_links(marker)
        if not marker.is_file() or marker.read_text(encoding="utf-8") != MARKER_CONTENT:
            raise H3CEError("E_CACHE_ROOT", "Missing or invalid H3CE cache root marker.")

    def _safe(self, path: Path) -> Path:
        assert_no_links(path)
        resolved = path.resolve()
        if not within(resolved, self.root.resolve()):
            raise H3CEError("E_CACHE_PATH", "Cache path escapes its marked root.")
        return path

    @contextmanager
    def _db(self):
        self._safe(self.root / "index.sqlite")
        db = sqlite3.connect(self.root / "index.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _lock(self):
        return FileLock(self._safe(self.root / "locks" / "store.lock"))

    def get(self, key: str, *, pin_manifest: str | None = None):
        if pin_manifest is not None:
            with self._lock():
                entry = self.get(key)
                if entry is not None:
                    with self._db() as db:
                        db.execute("INSERT OR IGNORE INTO pins VALUES (?,?)", (pin_manifest, key))
                return entry
        with self._db() as db:
            row = db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        entry = dict(row)
        path = self._safe(self.root / entry["path"])
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise H3CEError("E_CACHE_CHECKSUM", f"Cache entry failed checksum: {key}")
        entry["metadata"] = json.loads(entry["metadata"])
        entry["absolute_path"] = str(path)
        return entry

    def put_bytes(self, stage: str, key: str, payload: bytes, *, suffix=".bin", metadata=None,
                  pin_manifest: str | None = None):
        if stage not in STAGES or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise H3CEError("E_CACHE_KEY", "Invalid stage or SHA256 cache key.")
        if suffix not in {".bin", ".json", ".npy"}:
            raise H3CEError("E_CACHE_PATH", "Unsupported cache file suffix.")
        self._check_root()
        with self._lock():
            previous = self.get(key)
            sha = hashlib.sha256(payload).hexdigest()
            if previous:
                if previous["sha256"] != sha or previous["stage"] != stage:
                    raise H3CEError("E_CACHE_COLLISION", "Different content for an existing deterministic key.")
                if pin_manifest is not None:
                    with self._db() as db:
                        db.execute("INSERT OR IGNORE INTO pins VALUES (?,?)", (pin_manifest, key))
                return previous
            path = self._safe(self.root / stage / key[:2] / (key + suffix))
            atomic_write(path, payload)
            if file_sha256(path) != sha:
                raise H3CEError("E_CACHE_CHECKSUM", "Written content failed verification.")
            with self._db() as db:
                db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?)", (
                    key, stage, str(path.relative_to(self.root)), sha, len(payload),
                    canonical_json(metadata or {}).decode()))
                if pin_manifest is not None:
                    db.execute("INSERT OR IGNORE INTO pins VALUES (?,?)", (pin_manifest, key))
            return self.get(key)

    def put_json(self, stage: str, key: str, value, *, pin_manifest=None):
        return self.put_bytes(stage, key, canonical_json(value), suffix=".json", pin_manifest=pin_manifest)

    def put_array(self, stage: str, key: str, array, *, metadata=None, pin_manifest=None):
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        return self.put_bytes(stage, key, stream.getvalue(), suffix=".npy", metadata=metadata, pin_manifest=pin_manifest)

    def read_array(self, key: str):
        entry = self.get(key)
        if entry is None:
            raise H3CEError("E_CACHE_MISSING", f"Missing entry {key}")
        return np.load(entry["absolute_path"], allow_pickle=False)

    def pin(self, manifest: str, keys):
        with self._lock(), self._db() as db:
            for key in set(keys):
                if db.execute("SELECT 1 FROM entries WHERE key=?", (key,)).fetchone() is None:
                    raise H3CEError("E_CACHE_MISSING", "Cannot pin an uncommitted entry.")
                db.execute("INSERT OR IGNORE INTO pins VALUES (?,?)", (manifest, key))

    def inspect(self):
        self._check_root()
        with self._db() as db:
            stages = [dict(row) for row in db.execute(
                "SELECT stage, COUNT(*) AS entries, SUM(bytes) AS bytes FROM entries GROUP BY stage")]
            pins = db.execute("SELECT COUNT(DISTINCT key) FROM pins").fetchone()[0]
        return {"root": str(self.root), "stages": stages, "pinned_entries": pins}

    def prune(self, *, dry_run=True):
        self._check_root()
        with self._lock(), self._db() as db:
            candidates = [dict(row) for row in db.execute(
                "SELECT * FROM entries WHERE key NOT IN (SELECT key FROM pins)")]
            # Validate every resolved target before deleting the first item. No traversal scan.
            for entry in candidates:
                path = self._safe(self.root / entry["path"])
                if entry["stage"] not in STAGES or not within(path.resolve(), (self.root / entry["stage"]).resolve()):
                    raise H3CEError("E_CACHE_PATH", "Indexed path has the wrong stage.")
                if path.is_dir():
                    raise H3CEError("E_CACHE_PATH", "Refusing recursive cleanup of an indexed directory.")
            if not dry_run:
                for entry in candidates:
                    self._safe(self.root / entry["path"]).unlink(missing_ok=True)
                    db.execute("DELETE FROM entries WHERE key=?", (entry["key"],))
        return {"dry_run": dry_run, "entries": len(candidates),
                "bytes": sum(entry["bytes"] for entry in candidates),
                "keys": [entry["key"] for entry in candidates]}
