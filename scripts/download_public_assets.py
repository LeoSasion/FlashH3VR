"""Explicit downloads for the public inference release; standard library only.

Run from a source checkout. Existing matching files are reused; mismatching files
are never overwritten. This does not install packages or load model checkpoints.
Model and dependency terms are documented in docs/ASSETS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile


MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "public_assets.json"
BUNDLE_MEMBERS = {
    "flashh3vr-dense-1837.safetensors", "LICENSE-MINIMAX-H3", "MODEL_CARD.md",
    "NOTICE", "README.md", "SHA256SUMS", "weights_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_file(path: Path, digest: str, size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Unexpected byte count: {path}")
    if sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch; existing file left untouched: {path}")


def download_asset(entry: dict, directory: Path, *, verify_only: bool = False) -> Path:
    target = directory / entry["filename"]
    if target.exists() or verify_only:
        verify_file(target, entry["sha256"], entry["size_bytes"])
        return target
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "FlashH3VR-public-assets/1"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=target.name + ".", suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                count = 0
                for data in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(data)
                    if count > entry["size_bytes"]:
                        raise ValueError("Download exceeded the pinned asset size")
                    output.write(data)
        verify_file(temporary, entry["sha256"], entry["size_bytes"])
        # Opening exclusively also protects an existing file created meanwhile.
        with target.open("xb") as output, temporary.open("rb") as source:
            for data in iter(lambda: source.read(1024 * 1024), b""):
                output.write(data)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack_dense(archive: Path, entry: dict, directory: Path, *, verify_only: bool = False) -> None:
    """Validate the entire fixed bundle before writing any extracted member."""
    verify_file(archive, entry["sha256"], entry["size_bytes"])
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(BUNDLE_MEMBERS) or set(names) != BUNDLE_MEMBERS:
            raise ValueError("Unexpected or duplicate Dense bundle member")
        if sum(info.file_size for info in zipped.infolist()) > 16 * 1024 * 1024:
            raise ValueError("Unexpected Dense bundle uncompressed size")
        contents = {name: zipped.read(name) for name in names}
    checks = {}
    for line in contents["SHA256SUMS"].decode("ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name in checks:
            raise ValueError("Duplicate bundle checksum")
        checks[name] = digest
    if set(checks) != BUNDLE_MEMBERS - {"SHA256SUMS"}:
        raise ValueError("Incomplete bundle checksum manifest")
    for name, digest in checks.items():
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if checks[entry["weight_filename"]] != entry["weight_sha256"]:
        raise ValueError("Dense weight identity does not match the release")
    # Check all existing files first, so a mismatch cannot cause partial extraction.
    for name, data in contents.items():
        path = directory / name
        if path.exists() or verify_only:
            verify_file(path, hashlib.sha256(data).hexdigest(), len(data))
    if not verify_only:
        for name, data in contents.items():
            path = directory / name
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--asset", choices=("dense", "h3", "face", "all"), required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check existing downloads and extracted files without network access or writes")
    args = parser.parse_args(argv)
    assets = json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]
    names = list(assets) if args.asset == "all" else [args.asset]
    for name in names:
        entry = assets[name]
        print(f"{'Checking' if args.verify_only else 'Preparing'} {name}: {entry['filename']}", flush=True)
        path = download_asset(entry, args.models_dir, verify_only=args.verify_only)
        if entry["format"] == "dense_bundle":
            unpack_dense(path, entry, args.models_dir, verify_only=args.verify_only)
        print(f"Verified {name}: {entry['sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
