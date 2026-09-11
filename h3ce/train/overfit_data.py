"""Bounded degraded overfit examples and their optional exact clean companions."""
from __future__ import annotations

from collections import defaultdict
import math

from h3ce.errors import H3CEError


# Pixel-cache roles are keyed by the variant-derived view ID, including Y and
# the padding map. Their CONTENT is still checked by CachedImageDataset.audit.
# Target latent keys/paths are pixel-content addressed and must remain equal.
_VARIANT_DERIVED_FIELDS = frozenset({
    "view_id", "variant_id", "x_crop_path", "y_crop_path", "scene_x_path",
    "pad_valid_map", "z_input_key", "z_input_path",
})
_REQUIRED_VIEW_FIELDS = frozenset({
    "view_id", "variant_id", "asset_id", "target_id", "mode", "bucket_hw",
    "crop_xyxy", "crop_to_original", "encoder_contract_id", "latent_execution_contract",
    "z_target_key", "z_target_path", "split", "media_kind", "latent_status", "record_type",
    "x_crop_path", "y_crop_path", "scene_x_path", "pad_valid_map", "z_input_key", "z_input_path",
})


def _require(condition, message):
    if not condition:
        raise H3CEError("E_OVERFIT_DATA", message)


def validate_overfit_dataset(dataset):
    """Validate metadata after the dataset's mandatory full pixel/cache audit.

    There are 8–16 total views and 8–16 distinct original images. Each original
    contributes exactly one degraded view and, optionally, one clean companion.
    This does not replace pixel identity, geometry or latent provenance audits.
    """
    views = dataset.views
    _require(8 <= len(views) <= 16, "Overfit requires 8–16 total views including clean companions")
    groups = defaultdict(lambda: {"degraded": [], "clean": []})
    seen = set()
    try:
        for view in views:
            _require(_REQUIRED_VIEW_FIELDS.issubset(view) and all(view[field] for field in _REQUIRED_VIEW_FIELDS),
                     "Overfit view lacks complete source, geometry, Encoder or pixel/latent cache metadata")
            _require(view["view_id"] not in seen, "Overfit view identifiers must be unique")
            seen.add(view["view_id"])
            variant = dataset.variants[view["variant_id"]]
            target = dataset.targets[view["target_id"]]
            source = dataset.sources[view["asset_id"]]
            _require(variant["variant_id"] == view["variant_id"]
                     and variant["target_id"] == view["target_id"]
                     and target["target_id"] == view["target_id"]
                     and target["asset_id"] == view["asset_id"]
                     and source["asset_id"] == view["asset_id"],
                     "Overfit view changes its variant/target/source parent")
            _require(type(variant["clean_pair"]) is bool, "Clean-pair flag must be a boolean")
            kind = "clean" if variant["clean_pair"] else "degraded"
            groups[view["asset_id"]][kind].append(view)
        _require(8 <= len(groups) <= 16, "Overfit requires 8–16 distinct original images")
        for pair in groups.values():
            _require(len(pair["degraded"]) == 1,
                     "Each overfit original requires exactly one degraded view; clean-only originals are forbidden")
            _require(len(pair["clean"]) <= 1, "Each overfit original allows at most one clean companion")
            if pair["clean"]:
                degraded, clean = pair["degraded"][0], pair["clean"][0]
                signature = lambda view: {key: value for key, value in view.items()
                                          if key not in _VARIANT_DERIVED_FIELDS}
                _require(signature(degraded) == signature(clean),
                         "Clean companion must retain the degraded view's exact source, target, mode, geometry, Encoder and target latent")
        _require({pair["degraded"][0]["mode"] for pair in groups.values()} == {"fullbody", "face"},
                 "Overfit degraded views must cover both fullbody and face modes")
    except (KeyError, TypeError, AttributeError) as exc:
        raise H3CEError("E_OVERFIT_DATA", f"Malformed overfit parent records: {exc}") from exc
    return {"views": len(views), "original_images": len(groups),
            "degraded": len(groups), "clean": len(views) - len(groups)}


def summarize_probe(totals, rgb, pair_kinds, asset_ids, view_ids):
    """Keep the historical aggregate fields and expose each pair kind separately."""
    _require(bool(totals) and len({len(values) for values in (totals, rgb, pair_kinds, asset_ids, view_ids)}) == 1,
             "Probe metric arrays must be nonempty and aligned")
    _require(all(kind in {"degraded", "clean"} for kind in pair_kinds), "Unknown probe pair kind")
    _require(all(math.isfinite(value) for value in [*totals, *rgb]), "Probe metrics must be finite")
    by_kind = {}
    for kind in ("degraded", "clean"):
        indices = [i for i, value in enumerate(pair_kinds) if value == kind]
        by_kind[kind] = {"count": len(indices),
            "mean_total": sum(totals[i] for i in indices)/len(indices) if indices else None,
            "mean_rgb": sum(rgb[i] for i in indices)/len(indices) if indices else None,
            "per_view_total": [totals[i] for i in indices], "per_view_rgb": [rgb[i] for i in indices],
            "view_ids": [view_ids[i] for i in indices]}
    return {"mean_total": sum(totals)/len(totals), "mean_rgb": sum(rgb)/len(rgb),
            "per_view_total": totals, "per_view_rgb": rgb,
            "counts": {"views": len(totals), "original_images": len(set(asset_ids)),
                       "degraded": by_kind["degraded"]["count"], "clean": by_kind["clean"]["count"]},
            "by_pair_kind": by_kind, "scope": "fixed_training_subset_not_independent_validation"}


def overfit_probe_passed(*, step, gradients, pixel_gradient_max, before, after):
    """Clean improvement must never hide degraded regression at the overfit gate.

    Historical no-clean probe snapshots lack by_pair_kind. They remain readable,
    but are never treated as a degraded baseline for a current mixed probe.
    """
    before_groups, after_groups = before.get("by_pair_kind"), after.get("by_pair_kind")
    has_clean = any(probe.get("counts", {}).get("clean", 0) > 0
                    or probe.get("by_pair_kind", {}).get("clean", {}).get("count", 0) > 0
                    for probe in (before, after))
    if has_clean and (before_groups is None or after_groups is None):
        return False
    baseline = before_groups["degraded"] if before_groups is not None else before
    candidate = after_groups["degraded"] if after_groups is not None else after
    if baseline.get("count", 1) < 1 or candidate.get("count", 1) < 1:
        return False
    if before_groups is not None and after_groups is not None:
        if baseline.get("view_ids") != candidate.get("view_ids") or baseline["count"] != candidate["count"]:
            return False
    values = [baseline.get("mean_total"), baseline.get("mean_rgb"), candidate.get("mean_total"),
              candidate.get("mean_rgb"), gradients.get("spatial"), gradients.get("scene"), pixel_gradient_max]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return False
    return (step >= 2 and gradients["spatial"] > 0 and gradients["scene"] > 0 and pixel_gradient_max > 0
            and candidate["mean_total"] < baseline["mean_total"]
            and candidate["mean_rgb"] <= baseline["mean_rgb"])
