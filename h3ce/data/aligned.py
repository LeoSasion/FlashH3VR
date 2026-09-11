"""Explicit original-degradation / AI-target pairs; enhanced pixels never feed X.

This narrowly scoped contract retains the original target and degraded variant in
the same manifest. Generated detail is a pseudo target, not verified ground truth.
"""
from pathlib import Path

import numpy as np

from h3ce.cache.keys import file_sha256, stage_key
from h3ce.cache.store import assert_no_links
from h3ce.data.decode import decode_image, working_canvas
from h3ce.data.degrade import degrade_rgb, plan_variants
from h3ce.errors import H3CEError


def require(ok, message):
    if not ok:
        raise H3CEError("E_ALIGNED_PAIR", message)


def paired_target_id(contract):
    return stage_key("working_targets", aligned_original_pair=contract)


def paired_variant_plan(original_plan, target_id):
    require(original_plan.get("clean_pair") is False, "AI target must not create an identity/clean pair")
    return {**original_plan, "target_id": target_id,
            "variant_id": stage_key("aligned_variant", original_variant=original_plan["variant_id"], target_id=target_id),
            "degradation_source": "unenhanced_original",
            "original_variant_id": original_plan["variant_id"]}


def validate_aligned_target(target, targets, sources):
    try:
        contract = target["pair_contract"]
        require(set(contract) == {"version", "degradation_input", "original_target_id", "original_sha256",
                                  "enhanced_path", "enhanced_sha256", "geometry", "quality"},
                "Unknown or incomplete original-pair contract")
        require(contract["version"] == 1 and contract["degradation_input"] == "unenhanced_original"
                and contract["quality"] == "ai_enhanced_pseudo_target", "Explicit original-only degradation is required")
        require(target["target_id"] == paired_target_id(contract), "Aligned target content/geometry key differs")
        origin = targets[contract["original_target_id"]]
        require(origin["target_kind"] == "hq_self_degrade" and origin["asset_id"] == target["asset_id"],
                "Paired X and Y must retain the same original asset")
        require(all(target[k] == origin[k] for k in ("working_hw", "person_boxes", "face_boxes", "bbox_provenance")),
                "Aligned target changes original geometry")
        source = sources[origin["asset_id"]]
        require(contract["original_sha256"] == source["sha256"], "Pair references a different original")
        require(contract["geometry"] == {"mode": "shared_canvas_no_warp", "working_hw": target["working_hw"],
                                         "original_hw": source["original_hw"]}, "Unknown aligned coordinate contract")
        require(all(a <= b for a, b in zip(target["working_hw"], source["original_hw"])), "Do not upscale the original")
        path = Path(contract["enhanced_path"])
        assert_no_links(path)
        require(path.is_file() and file_sha256(path) == contract["enhanced_sha256"], "Enhanced file failed checksum")
        pixels, _ = decode_image(path)
        require(list(pixels.shape[:2]) == target["working_hw"], "Target must use the native generated canvas without upscaling")
        return origin
    except (KeyError, TypeError, ValueError) as exc:
        raise H3CEError("E_ALIGNED_PAIR", f"Malformed aligned target: {exc}") from exc


def expected_aligned_plans(target, targets, sources, config):
    require(config.data.mode == "aligned_pairs" and bool(config.paths.pairs_manifest),
            "Mixed AI pairs require an explicitly configured pair manifest")
    origin = validate_aligned_target(target, targets, sources)
    return [paired_variant_plan(p, target["target_id"]) for p in plan_variants(
        config.data.degradation, origin["target_id"], "image", origin["working_hw"],
        global_seed=config.data.degradation.seed) if not p["clean_pair"]]


def validate_aligned_variant(variant, target, targets, sources, variants, config=None):
    origin = validate_aligned_target(target, targets, sources)
    original = variants.get(variant.get("original_variant_id"))
    require(original is not None and original["target_id"] == origin["target_id"],
            "Aligned pair is missing its original degraded variant")
    # Reuse the SAME committed complete X, not just an equal-looking JPEG.
    expected = paired_variant_plan(original, target["target_id"])
    require(all(variant.get(k) == v for k, v in expected.items()), "Pair must reuse the unchanged original degradation")
    if config is not None:
        plans = expected_aligned_plans(target, targets, sources, config)
        require(any(all(variant.get(k) == v for k, v in p.items()) for p in plans), "Aligned recipe differs from config")


def audit_aligned_pixels(target, origin, source, variant, x, y, original_y):
    raw, _ = decode_image(Path(source["path"]))
    require(file_sha256(Path(source["path"])) == source["sha256"], "Original changed during pair loading")
    require(np.array_equal(original_y, working_canvas(raw, max(origin["working_hw"]))),
            "Degradation origin is not the unenhanced original")
    generated, _ = decode_image(Path(target["pair_contract"]["enhanced_path"]))
    require(np.array_equal(y, generated), "AI target pixels differ from the native generated image")
    require(np.array_equal(x, degrade_rgb(original_y, variant)), "X was not degraded from the unenhanced original")
