"""Per-frame head geometry shared by training pairs and spatial inference.

The original frame is never resized during composition. Only the predicted
correction is inverse-resized, so a zero correction leaves every pixel intact.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def head_transform(face_xyxy, canvas_hw, *, frame_index, pts, side=512,
                   expansion_xy=(1.5, 2.0), provenance='input_detector'):
    h, w = map(int, canvas_hw)
    a, b, c, d = map(float, face_xyxy[:4])
    ex, ey = map(float, expansion_xy)
    if (min(h, w, side) < 1 or not all(math.isfinite(v) for v in (a, b, c, d, ex, ey, pts))
            or not (0 <= a < c <= w and 0 <= b < d <= h) or min(ex, ey) < 1):
        raise ValueError('Invalid face/canvas/expansion geometry')
    cx, cy = (a + c) / 2, (b + d) / 2
    cw, ch = (c - a) * ex, (d - b) * ey
    x0, y0 = max(0, math.floor(cx - cw / 2)), max(0, math.floor(cy - ch / 2))
    x1, y1 = min(w, math.ceil(cx + cw / 2)), min(h, math.ceil(cy + ch / 2))
    sh, sw = y1 - y0, x1 - x0
    scale = side / max(sh, sw)
    rh, rw = max(1, round(sh * scale)), max(1, round(sw * scale))
    left, top = (side - rw) // 2, (side - rh) // 2
    sx, sy = rw / sw, rh / sh
    tx, ty = left + (sx - 1) / 2 - sx * x0, top + (sy - 1) / 2 - sy * y0
    return {
        'version': 'dynamic_head_bucket_v1', 'frame_index': int(frame_index), 'pts': float(pts),
        'canvas_hw': [h, w], 'face_xyxy': [a, b, c, d], 'bbox_provenance': provenance,
        'expansion_xy': [ex, ey], 'crop_xyxy': [x0, y0, x1, y1], 'source_hw': [sh, sw],
        'bucket_hw': [side, side], 'resized_hw': [rh, rw], 'pad_lrtb': [left, side-rw-left, top, side-rh-top],
        'scale_xy': [sx, sy], 'requested_uniform_scale': scale,
        'original_to_bucket': [[sx, 0, tx], [0, sy, ty], [0, 0, 1]],
        'bucket_to_original': [[1/sx, 0, -tx/sx], [0, 1/sy, -ty/sy], [0, 0, 1]],
        'coordinates': 'pixel centers; crop xyxy exclusive upper edge; align_corners=False',
        'padding': 'replicate; padding excluded from target supervision',
        'forward_resize': 'bicubic_antialias_clamp01',
        'inverse_resize': 'area' if sh <= rh and sw <= rw else 'bicubic_antialias',
        'face_fraction_of_bucket': (c-a)*(d-b)*sx*sy/(side*side),
    }


def pack_frame(frame, transform):
    """BCHW full-canvas RGB -> BCHW spatial bucket, with identical X/Y geometry."""
    if frame.ndim != 4 or list(frame.shape[-2:]) != transform['canvas_hw']:
        raise ValueError('Frame and transform canvas differ')
    x0, y0, x1, y1 = transform['crop_xyxy']
    crop = frame[..., y0:y1, x0:x1]
    resized = F.interpolate(crop, size=transform['resized_hw'], mode='bicubic',
                            align_corners=False, antialias=True).clamp(0, 1)
    return F.pad(resized, tuple(transform['pad_lrtb']), mode='replicate')


def valid_bucket(transform, *, device=None):
    mask = torch.zeros(1, 1, *transform['bucket_hw'], device=device)
    left, _, top, _ = transform['pad_lrtb']
    h, w = transform['resized_hw']
    mask[..., top:top+h, left:left+w] = 1
    return mask


def pack_training_pair(x, y, transform):
    if x.shape != y.shape:
        raise ValueError('Aligned training pair must share its full-canvas geometry')
    return pack_frame(x, transform), pack_frame(y, transform), valid_bucket(transform, device=x.device)


def inverse_delta(delta, transform):
    if delta.ndim != 4 or list(delta.shape[-2:]) != transform['bucket_hw']:
        raise ValueError('Predicted correction and bucket dimensions differ')
    left, _, top, _ = transform['pad_lrtb']
    h, w = transform['resized_hw']
    content = delta[..., top:top+h, left:left+w]
    if transform['inverse_resize'] == 'area':
        return F.interpolate(content, size=transform['source_hw'], mode='area')
    return F.interpolate(content, size=transform['source_hw'], mode='bicubic',
                         align_corners=False, antialias=True)


def crop_feather(transform, *, device=None):
    h, w = transform['source_hw']
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    dy = torch.minimum(y, h - 1 - y) / max(1., h * .05)
    dx = torch.minimum(x, w - 1 - x) / max(1., w * .05)
    return torch.minimum(dy[:, None], dx[None, :]).clamp(0, 1)[None, None]


def paste_delta(original, delta, transform):
    """Inverse-map a bucket correction and feather it into this frame's own box."""
    if original.ndim != 4 or list(original.shape[-2:]) != transform['canvas_hw']:
        raise ValueError('Original frame and transform differ')
    if original.shape[:2] != delta.shape[:2]:
        raise ValueError('Original and correction batch/channels differ')
    x0, y0, x1, y1 = transform['crop_xyxy']
    restored = original.clone()
    correction = inverse_delta(delta, transform)
    restored[..., y0:y1, x0:x1] = original[..., y0:y1, x0:x1] + crop_feather(
        transform, device=original.device) * correction
    return restored
