import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.download_public_assets import BUNDLE_MEMBERS, MANIFEST, download_asset, unpack_dense


def bundle(tmp_path, *, extra=None, bad_checksum=False):
    data = {name: name.encode() for name in BUNDLE_MEMBERS - {"SHA256SUMS"}}
    checks = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    if bad_checksum:
        checks["NOTICE"] = "0" * 64
    data["SHA256SUMS"] = "".join(f"{digest}  {name}\n" for name, digest in checks.items()).encode()
    if extra:
        data[extra] = b"unexpected"
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, value in data.items():
            zipped.writestr(name, value)
    entry = {"filename": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
             "size_bytes": archive.stat().st_size, "weight_filename": "flashh3vr-dense-1837.safetensors",
             "weight_sha256": checks["flashh3vr-dense-1837.safetensors"]}
    return archive, entry


def test_existing_wrong_file_is_not_overwritten_or_downloaded(tmp_path, monkeypatch):
    archive, entry = bundle(tmp_path)
    archive.write_bytes(b"user file")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("network must not be used"))
    with pytest.raises(ValueError):
        download_asset(entry, tmp_path)
    assert archive.read_bytes() == b"user file"


@pytest.mark.parametrize("extra,bad_checksum", [("../escape", False), (None, True)])
def test_invalid_bundle_does_not_extract(tmp_path, extra, bad_checksum):
    archive, entry = bundle(tmp_path, extra=extra, bad_checksum=bad_checksum)
    dest = tmp_path / "models"
    dest.mkdir()
    with pytest.raises(ValueError):
        unpack_dense(archive, entry, dest)
    assert list(dest.iterdir()) == []


def test_conflicting_notice_prevents_partial_extraction(tmp_path):
    archive, entry = bundle(tmp_path)
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "NOTICE").write_bytes(b"other package")
    with pytest.raises(ValueError):
        unpack_dense(archive, entry, dest)
    assert [p.name for p in dest.iterdir()] == ["NOTICE"]


def test_bundle_can_be_verified_without_rewriting(tmp_path):
    archive, entry = bundle(tmp_path)
    dest = tmp_path / "models"
    dest.mkdir()
    unpack_dense(archive, entry, dest)
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dest.iterdir()}
    unpack_dense(archive, entry, dest, verify_only=True)
    assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dest.iterdir()}


def test_manifest_matches_runtime_dense_and_h3_identities():
    from flashh3vr import DENSE_WEIGHT_SHA256, H3_WEIGHT_SHA256
    assets = json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]
    assert assets["dense"]["weight_sha256"] == DENSE_WEIGHT_SHA256
    assert assets["h3"]["sha256"] == H3_WEIGHT_SHA256
