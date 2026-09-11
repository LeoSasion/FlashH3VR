"""Deterministic CPU degradation of whole RGB working images and clips.

These are real pixel operations, independent of detection, crops, epochs, workers,
and H3. Inputs and ground truth are never changed in place. Reproducibility is
defined for the processor versions recorded in each resolved plan.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import PIL
from PIL import Image, features

from h3ce.errors import H3CEError


_ORDER = ["blur", "downsample", "noise", "compression", "upsample"]
_FIELDS = {
    "enabled", "version", "variants_per_image", "variants_per_clip",
    "include_clean_pair", "seed", "materialize", "resample_each_epoch",
    "recipe_order", "video_parameter_policy", "resolution", "blur", "noise",
    "compression", "max_resample_attempts",
}
_NESTED = {
    "resolution": {"mode", "ratio_range", "short_edge_range", "distribution",
                   "min_lr_side", "downsample_filter", "upsample_filter",
                   "allow_superresolution_of_target"},
    "blur": {"probability", "kind", "sigma_reference_px_range",
             "reference_short_edge", "truncate", "padding"},
    "noise": {"probability", "sigma_255_range", "realization"},
    "compression": {"kind", "probability", "jpeg_quality_range", "h264_crf_range"},
}


def canonical_json(value: Any) -> bytes:
    """The seed and identity serialization is UTF-8 canonical JSON, not hash()."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _error(message: str, code: str = "E_CONFIG") -> None:
    raise H3CEError(code, message)


def _degradation(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json")
    if not isinstance(config, Mapping):
        _error("degradation config must be a mapping or a Pydantic model")
    if "data" in config:
        config = config["data"]["degradation"]
    elif "degradation" in config:
        config = config["degradation"]
    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json")
    if not isinstance(config, Mapping):
        _error("data.degradation must be a mapping")
    return dict(config)


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        _error(f"{name} must be a mapping")
    missing, extra = expected - value.keys(), value.keys() - expected
    if missing or extra:
        _error(f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}")


def _number(value: Any, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        _error(f"{name} is outside its permitted range")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(f"{name} must be an integer >= {minimum}")
    return value


def _range(value: Any, name: str, minimum: float, maximum: float | None = None,
           integer: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _error(f"{name} must contain two ordered bounds")
    for bound in value:
        _number(bound, name, minimum, maximum)
        if integer and (isinstance(bound, bool) or not isinstance(bound, int)):
            _error(f"{name} must contain integer bounds")
    if value[0] > value[1]:
        _error(f"{name} must contain ordered bounds")


def validate_degradation(config: Any) -> dict[str, Any]:
    """Validate standalone dict callers as strictly as the project configuration."""
    d = _degradation(config)
    _keys(d, _FIELDS, "data.degradation")
    for name, fields in _NESTED.items():
        _keys(d[name], fields, f"data.degradation.{name}")
    for name in ("variants_per_image", "variants_per_clip", "max_resample_attempts"):
        _integer(d[name], name, 1)
    _integer(d["seed"], "seed")
    for name in ("enabled", "include_clean_pair", "materialize", "resample_each_epoch"):
        if not isinstance(d[name], bool):
            _error(f"{name} must be boolean")
    if d["version"] != "degrade_v2":
        _error("Only degradation version degrade_v2 is implemented", "E_NOT_IMPLEMENTED")
    if d["resample_each_epoch"] or d["recipe_order"] != _ORDER:
        _error("v2 requires fixed variants and blur/downsample/noise/compression/upsample order")
    if d["video_parameter_policy"] != "constant_per_clip":
        _error("Only constant_per_clip is implemented", "E_NOT_IMPLEMENTED")
    r, b, n, c = (d[k] for k in ("resolution", "blur", "noise", "compression"))
    if r["mode"] not in ("ratio", "short_edge"):
        _error("resolution.mode must be ratio or short_edge")
    _range(r["ratio_range"], "ratio_range", 0, 1)
    if r["ratio_range"][0] <= 0:
        _error("ratio_range must be strictly positive")
    _range(r["short_edge_range"], "short_edge_range", 1, integer=True)
    _integer(r["min_lr_side"], "min_lr_side", 1)
    if (r["distribution"] != "uniform" or r["downsample_filter"] != "area"
            or r["upsample_filter"] != "bicubic"):
        _error("Only uniform sampling, area downsampling and bicubic upsampling are implemented",
               "E_NOT_IMPLEMENTED")
    if r["allow_superresolution_of_target"] is not False:
        _error("v2 does not permit superresolution of the HQ target")
    for name, item in (("blur", b), ("noise", n), ("compression", c)):
        _number(item["probability"], f"{name}.probability", 0, 1)
    _range(b["sigma_reference_px_range"], "sigma_reference_px_range", 0)
    for name in ("reference_short_edge", "truncate"):
        if _number(b[name], f"blur.{name}", 0) <= 0:
            _error(f"blur.{name} must be positive")
    if b["kind"] != "gaussian_isotropic" or b["padding"] != "reflect":
        _error("Only isotropic Gaussian blur with reflect padding is implemented", "E_NOT_IMPLEMENTED")
    _range(n["sigma_255_range"], "sigma_255_range", 0)
    if n["realization"] != "independent_per_frame":
        _error("Only independent_per_frame noise is implemented", "E_NOT_IMPLEMENTED")
    if c["kind"] not in ("none", "jpeg", "h264"):
        _error("compression.kind must be none, jpeg or h264")
    if c["kind"] == "none" and c["probability"] != 0:
        _error("compression.probability must be zero when compression.kind is none")
    _range(c["jpeg_quality_range"], "jpeg_quality_range", 1, 100, integer=True)
    _range(c["h264_crf_range"], "h264_crf_range", 0, 51, integer=True)
    return d


def recipe_core(config: Any) -> dict[str, Any]:
    """Only fields affecting pixels enter the recipe; counts and inactive ranges do not."""
    d = validate_degradation(config)
    if not d["enabled"]:
        return {"version": d["version"], "enabled": False}
    r, b, n, c = (d[k] for k in ("resolution", "blur", "noise", "compression"))
    resolution = {k: r[k] for k in ("mode", "distribution", "min_lr_side",
                                   "downsample_filter", "upsample_filter")}
    active_range = "ratio_range" if r["mode"] == "ratio" else "short_edge_range"
    resolution[active_range] = list(r[active_range])
    compression = {"enabled": False}
    if c["kind"] != "none" and c["probability"] > 0:
        active_compression = "jpeg_quality_range" if c["kind"] == "jpeg" else "h264_crf_range"
        compression = {"kind": c["kind"], "probability": c["probability"],
                       active_compression: list(c[active_compression])}
    return {
        "version": d["version"], "enabled": True, "recipe_order": list(_ORDER),
        "video_parameter_policy": d["video_parameter_policy"], "resolution": resolution,
        "blur": dict(b) if b["probability"] > 0 else {"enabled": False},
        "noise": dict(n) if n["probability"] > 0 else {"enabled": False},
        "compression": compression, "max_resample_attempts": d["max_resample_attempts"],
    }


def recipe_hash(config: Any) -> str:
    return _digest(recipe_core(config))


def frame_noise_seed(variant_seed: int, frame_index: int) -> int:
    _integer(variant_seed, "variant_seed")
    _integer(frame_index, "frame_index")
    return int.from_bytes(hashlib.sha256(canonical_json([variant_seed, "noise", frame_index])).digest()[:8], "big")


def processor_contract() -> dict[str, Any]:
    """Include this contract in materialization cache keys when dependencies change."""
    return {"implementation": "numpy_pillow_rgb_v1", "numpy": np.__version__,
            "pillow": PIL.__version__, "jpeg_codec": features.version_codec("jpg"),
            "blur": "separable_gaussian_float32_numpy_reflect",
            "downsample": "exact_pixel_area_float64_accumulation",
            "upsample": "pillow_float32_bicubic_per_channel",
            "noise_rng": "numpy_pcg64", "clip_zero_one": ["after_noise", "after_upsample"],
            "jpeg_quantization": "round_half_up_uint8", "jpeg_subsampling": 0}


def plan_variants(config: Any, target_id: str, kind: str, working_hw: list[int] | tuple[int, int],
                  global_seed: int | None = None) -> list[dict[str, Any]]:
    """Sample N/M fixed whole-frame recipes, followed by an optional clean identity."""
    d = validate_degradation(config)
    if not isinstance(target_id, str) or not target_id:
        _error("target_id must be a nonempty string")
    if kind not in ("image", "video"):
        _error("kind must be image or video", "E_DEGRADE_MEDIA")
    if not isinstance(working_hw, (list, tuple)) or len(working_hw) != 2:
        _error("working_hw must be [height, width]", "E_DEGRADE_RANGE")
    h, w = (_integer(v, "working_hw dimension", 1) for v in working_hw)
    seed_base = d["seed"] if global_seed is None else _integer(global_seed, "global_seed")
    if d["enabled"] and d["compression"]["kind"] == "h264":
        if kind == "image":
            _error("H.264 may only be selected for real video", "E_DEGRADE_MEDIA")
        _error("H.264 whole-clip roundtrip needs a locked ffmpeg/color contract and is not implemented",
               "E_NOT_IMPLEMENTED")
    core_hash = recipe_hash(d)
    count = d["variants_per_image"] if kind == "image" else d["variants_per_clip"]
    plans = [_plan(d, target_id, kind, h, w, seed_base, core_hash, i)
             for i in range(count)] if d["enabled"] else []
    if d["include_clean_pair"]:
        identity_hash = _digest({"version": "degrade_v2", "identity": True})
        plans.append({
            "variant_id": _digest([target_id, identity_hash, seed_base, -1]),
            "target_id": target_id, "variant_index": -1,
            "seed": int.from_bytes(hashlib.sha256(canonical_json(
                [seed_base, target_id, identity_hash, -1])).digest()[:8], "big"),
            "recipe_hash": identity_hash, "lr_hw": [h, w], "clean_pair": True,
            "resolved_operations": {"media": kind, "working_hw": [h, w], "order": [],
                                    "identity": True, "processor": processor_contract()},
        })
    return plans


def _plan(d: dict[str, Any], target_id: str, kind: str, h: int, w: int,
          seed_base: int, core_hash: str, index: int) -> dict[str, Any]:
    seed = int.from_bytes(hashlib.sha256(canonical_json(
        [seed_base, target_id, core_hash, index])).digest()[:8], "big")
    rng, short = random.Random(seed), min(h, w)
    r, b, n, c = (d[k] for k in ("resolution", "blur", "noise", "compression"))
    for attempt in range(d["max_resample_attempts"]):
        if r["mode"] == "ratio":
            q = rng.uniform(*r["ratio_range"])
        else:
            low, high = max(r["min_lr_side"], r["short_edge_range"][0]), min(short, r["short_edge_range"][1])
            if low > high:
                _error("short_edge range has no intersection with the source/minimum LR dimensions",
                       "E_DEGRADE_RANGE")
            q = rng.randint(low, high) / short
        hl, wl = math.floor(h * q + 0.5), math.floor(w * q + 0.5)
        if min(hl, wl) >= r["min_lr_side"]:
            break
    else:
        _error("No valid LR dimensions within max_resample_attempts; dimensions were not clamped",
               "E_DEGRADE_RANGE")
    sigma_ref = rng.uniform(*b["sigma_reference_px_range"]) if rng.random() < b["probability"] else 0.0
    sigma = sigma_ref * short / b["reference_short_edge"]
    radius = math.ceil(b["truncate"] * sigma) if sigma else 0
    if radius >= short:
        _error("Gaussian reflect radius must be smaller than both working dimensions", "E_BLUR_KERNEL")
    noise_sigma = rng.uniform(*n["sigma_255_range"]) if rng.random() < n["probability"] else 0.0
    compression: dict[str, Any] = {"kind": "none"}
    if c["kind"] != "none" and rng.random() < c["probability"]:
        compression = {"kind": "jpeg", "quality": rng.randint(*c["jpeg_quality_range"]),
                       "subsampling": 0, "optimize": False, "progressive": False}
    operations = {
        "media": kind, "working_hw": [h, w], "order": list(_ORDER), "identity": False,
        "q_requested": q, "q_effective_yx": [hl / h, wl / w], "retry_count": attempt,
        "blur": {"sigma_reference_px": sigma_ref, "sigma_actual_px": sigma,
                 "radius": radius, "kernel": 2 * radius + 1, "padding": "reflect"},
        "noise": {"sigma_255": noise_sigma, "sigma_zero_one": noise_sigma / 255.0,
                  "realization": "independent_per_frame", "seed_derivation": "sha256_json_seed_noise_frame_big_endian_u64"},
        "compression": compression, "processor": processor_contract(),
    }
    return {"variant_id": _digest([target_id, core_hash, seed_base, index]), "target_id": target_id,
            "variant_index": index, "seed": seed, "recipe_hash": core_hash,
            "resolved_operations": operations, "lr_hw": [hl, wl], "clean_pair": False}


def _gaussian_rgb(frame: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    if sigma == 0:
        return frame.copy()
    if radius < 1 or radius >= min(frame.shape[:2]):
        _error("Gaussian reflect radius must be smaller than both dimensions", "E_BLUR_KERNEL")
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    weights = (weights / weights.sum()).astype(np.float32)
    result = frame
    for axis in (0, 1):
        padding = [(0, 0)] * 3
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="reflect")
        blurred = np.zeros_like(result)
        for i, weight in enumerate(weights):
            sl = [slice(None)] * 3
            sl[axis] = slice(i, i + result.shape[axis])
            blurred += weight * padded[tuple(sl)]
        result = blurred
    return result


def _area_axis(frame: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Integrate each piecewise constant pixel over every target pixel footprint."""
    values = np.moveaxis(frame, axis, 0)
    old_length = values.shape[0]
    if length == old_length:
        return frame.copy()
    edges = np.arange(length + 1, dtype=np.float64) * (old_length / length)
    edges[-1] = old_length
    starts = np.floor(edges).astype(np.int64)
    fractions = (edges - starts).reshape((-1,) + (1,) * (values.ndim - 1))
    prefix = np.concatenate((np.zeros((1,) + values.shape[1:], dtype=np.float64),
                             np.cumsum(values, axis=0, dtype=np.float64)), axis=0)
    integrals = prefix[starts] + fractions * values[np.minimum(starts, old_length - 1)]
    result = ((integrals[1:] - integrals[:-1]) / (old_length / length)).astype(np.float32)
    return np.moveaxis(result, 0, axis)


def _area_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return _area_axis(_area_axis(frame, hw[1], 1), hw[0], 0)


def _bicubic_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if frame.shape[:2] == hw:
        return frame.copy()
    return np.stack([np.asarray(Image.fromarray(frame[..., c]).resize(
        (hw[1], hw[0]), resample=Image.Resampling.BICUBIC), dtype=np.float32)
        for c in range(3)], axis=-1)


def _jpeg_rgb(frame: np.ndarray, quality: int) -> np.ndarray:
    quantized = np.floor(np.clip(frame, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    with io.BytesIO() as buffer:
        Image.fromarray(quantized).save(buffer, format="JPEG", quality=quality,
                                        subsampling=0, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def degrade_rgb(frames: np.ndarray, plan: Mapping[str, Any]) -> np.ndarray:
    """Execute a resolved recipe; return float32 RGB in [0, 1] with unchanged shape."""
    if not isinstance(frames, np.ndarray) or frames.dtype != np.float32:
        _error("degrade_rgb expects a float32 NumPy array", "E_DEGRADE_INPUT")
    if frames.ndim not in (3, 4) or frames.shape[-1] != 3 or any(d == 0 for d in frames.shape):
        _error("degrade_rgb expects H,W,3 or T,H,W,3 with nonempty dimensions", "E_DEGRADE_INPUT")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        _error("RGB values must be finite and in [0,1]", "E_DEGRADE_INPUT")
    operations = plan["resolved_operations"]
    hw = tuple(operations["working_hw"])
    source = frames[None] if frames.ndim == 3 else frames
    if source.shape[1:3] != hw:
        _error("Input dimensions differ from the resolved working canvas", "E_DEGRADE_INPUT")
    if operations["media"] == "image" and len(source) != 1:
        _error("An image plan cannot process a multi-frame clip", "E_DEGRADE_MEDIA")
    if plan["clean_pair"]:
        if operations.get("identity") is not True or operations["order"]:
            _error("Clean pairs must resolve to an identity recipe", "E_DEGRADE_INPUT")
        return frames.copy()
    if operations["processor"] != processor_contract():
        _error("Resolved processor versions differ from the running implementation; regenerate the plan",
               "E_DEGRADE_PROCESSOR")
    if operations["order"] != _ORDER:
        _error("Resolved operation order violates the v2 contract", "E_DEGRADE_INPUT")
    lr = tuple(plan["lr_hw"])
    if len(lr) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in lr):
        _error("Resolved LR dimensions must be positive integers", "E_DEGRADE_RANGE")
    if lr[0] > hw[0] or lr[1] > hw[1]:
        _error("Resolved LR dimensions cannot exceed the HQ canvas", "E_DEGRADE_RANGE")
    blur, noise, compression = (operations[k] for k in ("blur", "noise", "compression"))
    if compression["kind"] == "h264":
        if operations["media"] == "image":
            _error("H.264 may only be selected for real video", "E_DEGRADE_MEDIA")
        _error("H.264 whole-clip execution is not implemented", "E_NOT_IMPLEMENTED")
    if compression["kind"] not in ("none", "jpeg"):
        _error("Unsupported resolved compression", "E_DEGRADE_INPUT")
    output = np.empty_like(source)
    for frame_index, frame in enumerate(source):
        degraded = _area_rgb(_gaussian_rgb(frame, blur["sigma_actual_px"], blur["radius"]), lr)
        if noise["sigma_zero_one"] > 0:
            rng = np.random.Generator(np.random.PCG64(frame_noise_seed(plan["seed"], frame_index)))
            degraded += rng.normal(0, noise["sigma_zero_one"], degraded.shape).astype(np.float32)
            np.clip(degraded, 0, 1, out=degraded)
        if compression["kind"] == "jpeg":
            degraded = _jpeg_rgb(degraded, compression["quality"])
        output[frame_index] = np.clip(_bicubic_rgb(degraded, hw), 0, 1)
    return output[0] if frames.ndim == 3 else output
