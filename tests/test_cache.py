"""CPU/file-system contracts only: T13/T14 keys and T18 transactional safety."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from h3ce.cache.keys import (
    decoded_key,
    effective_decoder_hash,
    latent_key,
    stage_key,
)
from h3ce.cache.locks import FileLock
from h3ce.cache.store import CacheStore, MARKER, atomic_write
from h3ce.errors import H3CEError


def key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture
def cache(tmp_path):
    return CacheStore(tmp_path / "tmp", protected=[tmp_path / "raw", tmp_path / "runs", tmp_path / "exports"])


def assert_code(expected, action):
    with pytest.raises(H3CEError) as error:
        action()
    assert error.value.code == expected


def decoder_contract():
    return dict(
        base_decoder_weights="base-sha", adapter_weights=None, adapter_scale=1.0,
        code="locked-code", precision="upstream-float32", tile_contract={"tile": 256},
    )


@pytest.mark.parametrize("field,new_value", [
    ("base_decoder_weights", "other-base"),
    ("adapter_weights", "trained-adapter-sha"),
    ("adapter_scale", 0.25),
    ("code", "other-code"),
    ("precision", "other-precision"),
    ("tile_contract", {"tile": 512}),
])
def test_decoder_dependencies_invalidate_decode_but_preserve_latent(field, new_value):
    # T13 is a key-dependency contract; this does not run or assert H3 inference.
    view = stage_key("views", variant="same-input", geometry={"crop": [0, 0, 64, 64]})
    frozen_latent = latent_key(view, "encoder-and-normalization-v1")
    contract = decoder_contract()
    original = decoded_key(frozen_latent, effective_decoder_hash(**contract))
    contract[field] = new_value
    changed = decoded_key(frozen_latent, effective_decoder_hash(**contract))
    assert changed != original
    assert latent_key(view, "encoder-and-normalization-v1") == frozen_latent
    assert latent_key(view, "encoder-and-normalization-v2") != frozen_latent


def test_trainable_decoder_outputs_refused_and_teacher_uses_own_lock():
    # T14 refuses caching Dphi during optimization and binds frozen teacher to D0.
    latent = latent_key("view", "encoder")
    decoder = effective_decoder_hash(**decoder_contract())
    assert_code("E_CACHE_TRAINABLE_DECODER", lambda: decoded_key(latent, decoder, trainable=True))
    teacher = decoded_key(latent, decoder, trainable=False)
    altered_teacher = decoder_contract()
    altered_teacher["base_decoder_weights"] = "new-d0-lock"
    assert decoded_key(latent, effective_decoder_hash(**altered_teacher)) != teacher


def test_dependency_order_does_not_change_key():
    assert stage_key("views", a=1, b={"x": 2, "y": 3}) == stage_key("views", b={"y": 3, "x": 2}, a=1)
    assert stage_key("views", a=1) != stage_key("variants", a=1)


def test_atomic_replacement_interruption_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint-like-file.bin"
    atomic_write(destination, b"committed-before-interruption")

    def interrupted_replace(*_):
        raise OSError("injected interruption before atomic replacement")

    with monkeypatch.context() as patch:
        patch.setattr("h3ce.cache.store.os.replace", interrupted_replace)
        with pytest.raises(OSError, match="injected interruption"):
            atomic_write(destination, b"incomplete-new-content")
    assert destination.read_bytes() == b"committed-before-interruption"
    assert not list(tmp_path.glob(".partial-*"))


def test_interrupted_cache_write_is_not_visible_and_can_retry(cache, monkeypatch):
    content_key = key("interruption")

    def interrupted_replace(*_):
        raise OSError("injected interruption")

    with monkeypatch.context() as patch:
        patch.setattr("h3ce.cache.store.os.replace", interrupted_replace)
        with pytest.raises(OSError, match="interruption"):
            cache.put_bytes("variants", content_key, b"complete-payload")
    assert cache.get(content_key) is None
    assert not list(cache.root.rglob(".partial-*"))
    committed = cache.put_bytes("variants", content_key, b"complete-payload")
    assert Path(committed["absolute_path"]).read_bytes() == b"complete-payload"
    assert committed["sha256"] == hashlib.sha256(b"complete-payload").hexdigest()


def test_uncommitted_partial_file_cannot_be_read_or_pinned(cache):
    partial = cache.root / "variants" / ".partial-worker-killed"
    partial.write_bytes(b"half-written")
    content_key = key("not-committed")
    assert cache.get(content_key) is None
    assert_code("E_CACHE_MISSING", lambda: cache.pin("active-run", [content_key]))
    assert cache.prune()["entries"] == 0
    assert partial.read_bytes() == b"half-written"


def test_checksum_corruption_fails_closed_and_is_not_overwritten(cache):
    content_key = key("checksum")
    entry = cache.put_bytes("working_targets", content_key, b"trusted-content")
    path = Path(entry["absolute_path"])
    path.write_bytes(b"corrupted-content")
    assert_code("E_CACHE_CHECKSUM", lambda: cache.get(content_key))
    assert_code("E_CACHE_CHECKSUM", lambda: cache.put_bytes("working_targets", content_key, b"trusted-content"))
    assert path.read_bytes() == b"corrupted-content"


def test_missing_committed_payload_fails_closed(cache):
    content_key = key("missing-payload")
    entry = cache.put_bytes("variants", content_key, b"data")
    Path(entry["absolute_path"]).unlink()
    assert_code("E_CACHE_CHECKSUM", lambda: cache.get(content_key))


def test_identical_put_reuses_entry_and_conflicting_content_refused(cache):
    content_key = key("deduplicated")
    first = cache.put_bytes("variants", content_key, b"same-payload")
    second = cache.put_bytes("variants", content_key, b"same-payload")
    assert first == second
    assert_code("E_CACHE_COLLISION", lambda: cache.put_bytes("variants", content_key, b"different-payload"))
    assert_code("E_CACHE_COLLISION", lambda: cache.put_bytes("views", content_key, b"same-payload"))
    assert sum(stage["entries"] for stage in cache.inspect()["stages"]) == 1


def test_array_roundtrip_has_no_pickle_support(cache):
    content_key = key("float-array")
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    entry = cache.put_array("latents", content_key, array, metadata={"encoder_contract_id": "e1"})
    np.testing.assert_array_equal(cache.read_array(content_key), array)
    assert entry["metadata"] == {"encoder_contract_id": "e1"}
    with pytest.raises(ValueError, match="allow_pickle=False"):
        cache.put_array("latents", key("object-array"), np.array([object()], dtype=object))
    assert cache.get(key("object-array")) is None


def test_prune_defaults_dry_run_and_preserves_all_active_manifest_references(cache, tmp_path):
    pinned_a, pinned_b, unpinned = key("active-a"), key("active-b"), key("unused")
    for content_key in (pinned_a, pinned_b, unpinned):
        cache.put_bytes("variants", content_key, content_key.encode())
    cache.pin("runs/a", [pinned_a])
    cache.pin("runs/b", [pinned_a, pinned_b])
    protected = []
    for name in ("raw", "runs", "exports"):
        path = tmp_path / name / "must-survive.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(name, encoding="utf-8")
        protected.append(path)
    dry = cache.prune()
    assert dry["dry_run"] is True
    assert dry["keys"] == [unpinned]
    assert cache.get(unpinned) is not None
    deleted = cache.prune(dry_run=False)
    assert deleted["keys"] == [unpinned]
    assert cache.get(unpinned) is None
    assert cache.get(pinned_a) is not None
    assert cache.get(pinned_b) is not None
    assert cache.inspect()["pinned_entries"] == 2
    assert all(path.read_text(encoding="utf-8") == path.parent.name for path in protected)


def test_pin_batch_rolls_back_when_any_entry_is_uncommitted(cache):
    existing = key("existing")
    cache.put_bytes("views", existing, b"payload")
    assert_code("E_CACHE_MISSING", lambda: cache.pin("broken-manifest", [existing, key("missing")]))
    assert cache.inspect()["pinned_entries"] == 0


@pytest.mark.parametrize("operation", ["read", "put_existing", "put_new", "put_array", "put_json"])
def test_active_pin_is_committed_before_prune_can_acquire_store_lock(cache, monkeypatch, operation):
    content_key = key("atomic-pin-" + operation)
    if operation in {"read", "put_existing"}:
        cache.put_bytes("variants", content_key, b"active-data")
    original_lock = cache._lock
    prunes = []
    fired = False

    @contextmanager
    def prune_at_first_unlock():
        nonlocal fired
        with original_lock():
            yield
        # Deterministically give a competing prune its earliest possible turn,
        # immediately after the producer releases the shared lock.
        if not fired:
            fired = True
            prunes.append(cache.prune(dry_run=False))

    monkeypatch.setattr(cache, "_lock", prune_at_first_unlock)
    if operation == "read":
        entry = cache.get(content_key, pin_manifest="active-run")
    elif operation == "put_array":
        entry = cache.put_array("variants", content_key, np.zeros((2, 2), np.float32), pin_manifest="active-run")
    elif operation == "put_json":
        entry = cache.put_json("detections", content_key, {"person_boxes": []}, pin_manifest="active-run")
    else:
        entry = cache.put_bytes("variants", content_key, b"active-data", pin_manifest="active-run")
    assert len(prunes) == 1
    assert prunes[0]["keys"] == []
    assert Path(entry["absolute_path"]).is_file()
    assert cache.inspect()["pinned_entries"] == 1


def test_failed_pinned_write_does_not_publish_entry_or_pin(cache, monkeypatch):
    content_key = key("failed-pinned-write")

    def interrupted_replace(*_):
        raise OSError("injected interruption")

    with monkeypatch.context() as patch:
        patch.setattr("h3ce.cache.store.os.replace", interrupted_replace)
        with pytest.raises(OSError, match="interruption"):
            cache.put_bytes("variants", content_key, b"data", pin_manifest="active-run")
    assert cache.get(content_key) is None
    assert cache.inspect()["pinned_entries"] == 0


@pytest.mark.parametrize("change", ["remove", "replace"])
def test_prune_and_writes_require_current_valid_marker(cache, change):
    entry = cache.put_bytes("variants", key("marker-guard"), b"safe")
    marker = cache.root / MARKER
    if change == "remove":
        marker.unlink()
    else:
        marker.write_text("unrecognized cache", encoding="utf-8")
    assert_code("E_CACHE_ROOT", lambda: cache.prune(dry_run=False))
    assert_code("E_CACHE_ROOT", lambda: cache.put_bytes("views", key("new"), b"new"))
    assert Path(entry["absolute_path"]).read_bytes() == b"safe"


def test_nonempty_unrecognized_directory_is_not_claimed(tmp_path):
    raw = tmp_path / "originals"
    raw.mkdir()
    source = raw / "source.jpg"
    source.write_bytes(b"untouched-original")
    assert_code("E_CACHE_ROOT", lambda: CacheStore(raw))
    assert not (raw / MARKER).exists()
    assert source.read_bytes() == b"untouched-original"


@pytest.mark.parametrize("cache_relative,protected_relative", [
    ("raw", "raw"), ("raw/cache", "raw"), ("workspace", "workspace/raw"),
])
def test_cache_root_cannot_equal_contain_or_nest_protected_paths(tmp_path, cache_relative, protected_relative):
    cache_path = tmp_path / cache_relative
    protected = tmp_path / protected_relative
    assert_code("E_PATH_CONTRACT", lambda: CacheStore(cache_path, protected=[protected]))
    assert not (cache_path / MARKER).exists()


def test_indexed_path_traversal_prevents_every_deletion(cache, tmp_path):
    valid_key = key("first-valid")
    malicious_key = key("second-invalid")
    valid = cache.put_bytes("variants", valid_key, b"keep-valid-if-any-target-unsafe")
    cache.put_bytes("variants", malicious_key, b"original-cache-content")
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "original.bin"
    source.write_bytes(b"untouched-raw")
    with sqlite3.connect(cache.root / "index.sqlite") as db:
        db.execute("UPDATE entries SET path=? WHERE key=?", ("../raw/original.bin", malicious_key))
    assert_code("E_CACHE_PATH", lambda: cache.get(malicious_key))
    assert_code("E_CACHE_PATH", lambda: cache.prune(dry_run=False))
    assert Path(valid["absolute_path"]).read_bytes() == b"keep-valid-if-any-target-unsafe"
    assert source.read_bytes() == b"untouched-raw"


def test_indexed_wrong_stage_or_directory_is_not_deleted(cache):
    content_key = key("indexed-directory")
    cache.put_bytes("variants", content_key, b"data")
    with sqlite3.connect(cache.root / "index.sqlite") as db:
        db.execute("UPDATE entries SET path='views' WHERE key=?", (content_key,))
    assert_code("E_CACHE_PATH", lambda: cache.prune(dry_run=False))
    with sqlite3.connect(cache.root / "index.sqlite") as db:
        db.execute("UPDATE entries SET path='variants' WHERE key=?", (content_key,))
    assert_code("E_CACHE_PATH", lambda: cache.prune(dry_run=False))
    assert (cache.root / "variants").is_dir()


def directory_link(link: Path, target: Path):
    """Use a real Windows junction when unprivileged symlink creation is denied."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi
        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError as exc:
            pytest.skip(f"This host cannot create a symlink or junction: {exc}")


def test_cache_root_symlink_or_junction_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-cache"
    directory_link(link, outside)
    assert_code("E_CACHE_PATH", lambda: CacheStore(link))
    assert not (outside / MARKER).exists()


def test_prepare_does_not_resolve_away_linked_cache_root(tmp_path):
    from h3ce.config import ProjectConfig
    from h3ce.prepare import prepare_images

    outside = tmp_path / "outside"
    outside.mkdir()
    directory_link(tmp_path / "tmp", outside)
    config = ProjectConfig(schema_version=2, project={"root": str(tmp_path)})
    assert_code("E_CACHE_PATH", lambda: prepare_images(config, tmp_path, tmp_path / "runs" / "linked-test"))
    assert not (outside / MARKER).exists()


def test_stage_symlink_or_junction_cannot_be_traversed(cache, tmp_path):
    content_key = key("linked-stage")
    outside = tmp_path / "outside-stage"
    outside.mkdir()
    protected = outside / "preserve.txt"
    protected.write_bytes(b"outside-is-untouched")
    stage = cache.root / "variants"
    stage.rmdir()
    directory_link(stage, outside)
    assert_code("E_CACHE_PATH", lambda: cache.put_bytes("variants", content_key, b"must-not-write"))
    assert_code("E_CACHE_PATH", lambda: CacheStore(cache.root, create=False))
    assert list(outside.iterdir()) == [protected]


def test_prune_detects_hash_directory_link_before_deleting(cache, tmp_path):
    content_key = key("linked-indexed-entry")
    entry = cache.put_bytes("variants", content_key, b"cache-data")
    payload = Path(entry["absolute_path"])
    prefix = payload.parent
    payload.unlink()
    prefix.rmdir()
    outside = tmp_path / "outside-payload"
    outside.mkdir()
    external_payload = outside / payload.name
    external_payload.write_bytes(b"raw-data-must-survive")
    directory_link(prefix, outside)
    assert_code("E_CACHE_PATH", lambda: cache.prune(dry_run=False))
    assert external_payload.read_bytes() == b"raw-data-must-survive"


def test_two_processes_commit_same_key_without_collision_or_partial_read(cache):
    content_key = key("concurrent")
    script = """
from pathlib import Path
import sys
from h3ce.cache.store import CacheStore
store = CacheStore(Path(sys.argv[1]), create=False)
for _ in range(8):
    entry = store.put_bytes('variants', sys.argv[2], b'concurrent-payload')
    assert Path(entry['absolute_path']).read_bytes() == b'concurrent-payload'
"""
    processes = [subprocess.Popen(
        [sys.executable, "-c", script, str(cache.root), content_key],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for _ in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert sum(stage["entries"] for stage in cache.inspect()["stages"]) == 1
    assert cache.get(content_key)["bytes"] == len(b"concurrent-payload")
    assert not list(cache.root.rglob(".partial-*"))


def test_lock_is_released_after_exception(tmp_path):
    lock_path = tmp_path / "lock"
    with pytest.raises(RuntimeError, match="simulated worker failure"):
        with FileLock(lock_path, timeout=0.2):
            raise RuntimeError("simulated worker failure")
    with FileLock(lock_path, timeout=0.2):
        assert lock_path.is_file()
