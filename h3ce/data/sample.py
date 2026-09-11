"""Aspect-preserving fullbody/head-and-shoulder views, from one shared X."""
from __future__ import annotations

import math
import random

import numpy as np

from h3ce.cache.keys import digest
from .decode import resize_rgb


def _expand(box, xscale, yscale, hw):
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * xscale, (y2 - y1) * yscale
    return [max(0, math.floor(cx - w / 2)), max(0, math.floor(cy - h / 2)),
            min(hw[1], math.ceil(cx + w / 2)), min(hw[0], math.ceil(cy + h / 2))]


def view_geometries(person_boxes, face_boxes, working_hw, original_hw, sampling, target_id):
    cfg = sampling.model_dump() if hasattr(sampling, "model_dump") else sampling
    if len(person_boxes) != 1:
        return []
    rng = random.Random(int(digest([target_id, "view_geometry_v1"])[:16], 16))
    h, w = working_hw
    # Wide means the complete scene. A narrower body view still includes the entire person box.
    context = 1 + 2 * cfg["fullbody_context_fraction"]
    body = [0, 0, w, h] if rng.random() < cfg["fullbody_wide_probability"] else _expand(
        person_boxes[0], context, context, working_hw)
    crops = [("fullbody", body)]
    if len(face_boxes) == 1:
        box = face_boxes[0]
        face_pixels = min((box[2] - box[0]) * original_hw[1] / w,
                          (box[3] - box[1]) * original_hw[0] / h)
        if face_pixels >= cfg["min_face_hq_pixels"]:
            sx, sy = cfg["face_head_shoulders_expand_xy"]
            extra = 1 if rng.random() < cfg["face_close_probability"] else 1.5
            crops.append(("face", _expand(box, sx * extra, sy * extra, working_hw)))
    buckets = cfg["debug_buckets_hw"] if cfg["profile"] == "debug" else cfg["balanced_buckets_hw"]
    result = []
    for mode, crop in crops:
        ch, cw = crop[3] - crop[1], crop[2] - crop[0]
        bucket = min(buckets, key=lambda hw: (abs(math.log((hw[1] / hw[0]) / (cw / ch))), hw[0] * hw[1]))
        result.append({"mode": mode, "crop_xyxy": crop, "bucket_hw": list(bucket)})
    return result


def make_view(x, y, geometry, original_hw, scene_longest_edge=256):
    x1, y1, x2, y2 = geometry["crop_xyxy"]
    bh, bw = geometry["bucket_hw"]
    h, w = x.shape[:2]
    ch, cw = y2 - y1, x2 - x1
    scale = min(bw / cw, bh / ch)
    rh, rw = min(bh, int(ch * scale + 0.5)), min(bw, int(cw * scale + 0.5))
    oy, ox = (bh - rh) // 2, (bw - rw) // 2
    valid = np.zeros((bh, bw), dtype=np.float32)
    valid[oy:oy + rh, ox:ox + rw] = 1

    def letterbox(rgb):
        result = np.zeros((bh, bw, 3), dtype=np.float32)
        result[oy:oy + rh, ox:ox + rw] = resize_rgb(rgb[y1:y2, x1:x2], (rh, rw))
        return result

    scene_scale = min(1.0, scene_longest_edge / max(h, w))
    scene = resize_rgb(x, (max(1, int(h * scene_scale + 0.5)), max(1, int(w * scene_scale + 0.5))))
    sx, sy = original_hw[1] / w, original_hw[0] / h
    matrix = [[cw / rw * sx, 0, (x1 - ox * cw / rw) * sx],
              [0, ch / rh * sy, (y1 - oy * ch / rh) * sy], [0, 0, 1]]
    return {"x_crop": letterbox(x), "y_crop": letterbox(y), "pad_valid_map": valid,
            "scene_x": scene, "crop_to_original": matrix}

