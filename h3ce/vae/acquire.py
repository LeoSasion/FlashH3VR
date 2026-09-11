"""Explicit acquisition of the one pinned visual VAE used by AI Toolkit."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
URL = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{REVISION}/vae/minimax_h3_video_vae_fp16.safetensors"
SHA256 = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
SIZE = 5207808496


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def acquire(path, *, allow_download=False):
    path = Path(path)
    if path.exists():
        actual = digest(path)
        if path.stat().st_size != SIZE or actual != SHA256:
            raise RuntimeError("E_COMPONENT_HASH: existing visual VAE differs from pinned asset")
        return {"path": str(path.resolve()), "sha256": actual, "size_bytes": SIZE, "reused": True}
    if not allow_download:
        raise RuntimeError("E_COMPONENT_MISSING: acquisition requires --allow-download")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    started = time.monotonic()
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == SIZE:
            break
        request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                if offset and not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server did not honor resume offset")
                if offset > SIZE:
                    raise RuntimeError("Partial artifact exceeds pinned size")
                last_progress = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                        offset += len(block)
                        if offset > SIZE:
                            raise RuntimeError("Remote artifact exceeds pinned size")
                        if time.monotonic() - last_progress >= 10:
                            print(json.dumps({"downloaded_bytes": offset, "total_bytes": SIZE}), flush=True)
                            last_progress = time.monotonic()
            break
        except (OSError, TimeoutError) as error:
            print(json.dumps({"attempt": attempt, "download_error": str(error)}), flush=True)
            if attempt == 4:
                raise
    actual = digest(partial)
    if partial.stat().st_size != SIZE or actual != SHA256:
        raise RuntimeError(f"E_COMPONENT_HASH: expected {SHA256}; locally computed {actual}")
    partial.replace(path)
    result = {"path": str(path.resolve()), "sha256": actual, "size_bytes": SIZE,
              "source_url": URL, "revision": REVISION, "elapsed_seconds": time.monotonic() - started,
              "reused": False}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    acquire(args.out, allow_download=args.allow_download)
