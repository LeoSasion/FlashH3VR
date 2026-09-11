"""Read-only source inventory. A source subdirectory is a conservative source group."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from h3ce.cache.keys import digest, file_sha256
from h3ce.errors import H3CEError
from .decode import decode_image
from .split import assign_splits

IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".gif"}


def image_fingerprint(rgb):
    image = Image.fromarray(np.floor(rgb * 255 + 0.5).astype(np.uint8)).convert("L")
    small = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def scan_sources(raw: Path, *, validation_fraction: float, seed: int, maximum: int,
                 budget_check=lambda: None, include_videos=False, source_color_declarations=None):
    raw = Path(raw).resolve()
    if not raw.is_dir():
        raise H3CEError("E_TARGET_REQUIRED", f"Provide trusted HQ images in {raw}.")
    paths = []
    for directory, dirs, files in os.walk(raw, followlinks=False):
        for name in [*dirs, *files]:
            item = Path(directory) / name
            if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                raise H3CEError("E_SOURCE_PATH", f"Source links require explicit materialization: {item}")
        paths.extend(Path(directory) / name for name in files
                     if Path(name).suffix.lower() in IMAGES | VIDEOS)
    paths.sort(key=lambda path: path.relative_to(raw).as_posix())
    if not paths:
        raise H3CEError("E_TARGET_REQUIRED", "No trusted HQ source images found.")
    if len(paths) > maximum:
        raise H3CEError("E_DATA_LIMIT", "Source count exceeds max_independent_frames; preparation is incomplete.")
    if not include_videos and any(path.suffix.lower() in VIDEOS for path in paths):
        raise H3CEError("E_NOT_IMPLEMENTED", "Video/animation preparation awaits PTS decoding and shot/track validation.")
    records = []
    for path in paths:
        budget_check()
        kind = "video" if path.suffix.lower() in VIDEOS else "image"
        if kind == "video":
            from .video import iter_video_frames
            declaration = (source_color_declarations or {}).get(str(path.resolve()))
            decoder = iter_video_frames(path, source_color_declaration=declaration)
            try:
                first = next(decoder)
                rgb = first.rgb
                transform = digest({"video_color": first.color_contract})
            finally:
                decoder.close()
        else:
            rgb, transform = decode_image(path)
        sha = file_sha256(path)
        relative = path.relative_to(raw)
        record = {"asset_id": digest([relative.as_posix(), sha]), "sha256": sha,
                  "kind": kind, "path": str(path),
                  "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                  "split": "train", "color_transform_id": transform,
                  "original_hw": list(rgb.shape[:2]), "pts": [], "shot_id": None,
                  "_dhash": image_fingerprint(rgb) if kind == "image" else None}
        records.append(record)
    return assign_splits(records, validation_fraction=validation_fraction, seed=seed)
