"""Explicit SDR image decoding. Video PTS decoding is not yet implemented."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from h3ce.errors import H3CEError


def decode_image(path: Path):
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise H3CEError("E_NOT_IMPLEMENTED", "Animated images need a timestamp-aware decoder.")
            if opened.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                raise H3CEError("E_COLOR_CONTRACT", f"Unsupported high-depth/color mode {opened.mode}: {path}")
            oriented = ImageOps.exif_transpose(opened)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                alpha = oriented.convert("RGBA").getchannel("A")
                if alpha.getextrema() != (255, 255):
                    raise H3CEError("E_COLOR_CONTRACT", "Transparent images require an explicit background contract.")
            rgb = oriented.convert("RGB")
            profile = opened.info.get("icc_profile")
            transform = "declared_sdr_srgb_exif_v1"
            if profile:
                try:
                    rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                                    ImageCms.createProfile("sRGB"), outputMode="RGB")
                except (ValueError, OSError, ImageCms.PyCMSError) as exc:
                    raise H3CEError("E_COLOR_CONTRACT", "Embedded ICC profile cannot be converted to sRGB.") from exc
                transform = "icc_to_srgb:" + hashlib.sha256(profile).hexdigest()
            return np.asarray(rgb, dtype=np.float32) / 255.0, transform
    except H3CEError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise H3CEError("E_DECODE", f"Cannot decode {path}: {exc}") from exc


def resize_rgb(rgb, hw, *, packed_channels=False):
    """Float-channel bicubic interpolation, keeping the external RGB range."""
    h, w = map(int, hw)
    if min(h, w) < 1:
        raise H3CEError("E_GEOMETRY", "Image dimensions must be positive.")
    if tuple(rgb.shape[:2]) == (h, w):
        return rgb.copy()
    # Pack once into contiguous float planes to avoid Pillow's strided .tobytes
    # on each HWC channel. The same F-mode bicubic implementation is retained.
    planes = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)) if packed_channels else None
    channels = [np.asarray(Image.fromarray(planes[c] if planes is not None else np.asarray(rgb[..., c], dtype=np.float32)).resize(
        (w, h), Image.Resampling.BICUBIC)) for c in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)


def working_canvas(rgb, longest_edge):
    h, w = rgb.shape[:2]
    scale = min(1.0, longest_edge / max(h, w))
    hw = [max(1, int(h * scale + 0.5)), max(1, int(w * scale + 0.5))]
    return resize_rgb(rgb, hw)
