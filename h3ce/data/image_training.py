"""Image-only pretraining data checks and frozen-encoder cache binding.

No optimizer, backward pass, refiner or training stage is imported or run here.
An injected encoder in a unit test is a cache-contract fixture, never H3 evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from h3ce.cache.keys import digest, file_sha256, latent_key, stage_key
from h3ce.cache.store import assert_no_links
from h3ce.data.degrade import plan_variants
from h3ce.data.aligned import expected_aligned_plans, validate_aligned_variant, audit_aligned_pixels
from h3ce.errors import H3CEError


def require(condition, message):
    if not condition:
        raise H3CEError("E_IMAGE_PREPARATION", message)


def indexed_rows(records, kind, key):
    rows = [row for row in records if row.get("record_type") == kind]
    require(all(key in row for row in rows), f"Missing {key} in {kind}")
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {kind} identifiers")
    return result


def inspect_image_manifest(records, config, raw_root):
    require(config.native_temporal.mode == "frozen" and config.native_temporal.decoder_adapter is None,
            "Image preparation requires frozen original H3 VAE without a Decoder adapter")
    require(not config.vae.encoder_trainable and config.cache.cache_encoder_latents,
            "Frozen Encoder latent caching must be enabled")
    sources = indexed_rows(records, "source_asset", "asset_id")
    targets = indexed_rows(records, "prepared_target", "target_id")
    variants = indexed_rows(records, "degraded_variant", "variant_id")
    views = indexed_rows(records, "prepared_pixel_view", "view_id")
    require(bool(sources and targets and variants and views), "Preparation requires a nonempty complete pixel manifest")
    require(len(records) == sum(map(len, (sources, targets, variants, views))), "Unexpected manifest record type")
    group_splits, hash_splits = defaultdict(set), defaultdict(set)
    for source in sources.values():
        require(source["kind"] == "image" and source["pts"] == [], "This preparation accepts still images only")
        require(source["split"] in {"train", "val", "test"}, "Unknown source split")
        path = Path(source["path"])
        assert_no_links(path)
        require(path.resolve().is_relative_to(Path(raw_root).resolve()), "Source is outside configured raw")
        require(path.is_file() and file_sha256(path) == source["sha256"], "Source content differs from the preparation manifest")
        group_splits[source["source_group"]].add(source["split"])
        hash_splits[source["sha256"]].add(source["split"])
    require(all(len(splits) == 1 for splits in [*group_splits.values(), *hash_splits.values()]),
            "Same source group or duplicate content crosses splits")
    by_target, by_variant = defaultdict(list), defaultdict(list)
    for variant in variants.values():
        require(variant["target_id"] in targets, "Variant references a missing target")
        by_target[variant["target_id"]].append(variant)
    for target in targets.values():
        require(target["asset_id"] in sources, "Target references a missing source")
        require(target["valid_frames"] == 1 and target["real_pts"] == [] and target["source_frame_indices"] == [0],
                "Still image target contains video timing")
        require(len(target["person_boxes"]) == 1 and len(target["face_boxes"]) <= 1,
                "Ambiguous subject cannot enter the training preparation manifest")
        require(target["target_kind"] in {"hq_self_degrade", "aligned_pair"}, "Unknown target provenance")
        expected = (expected_aligned_plans(target, targets, sources, config) if target["target_kind"] == "aligned_pair"
                    else plan_variants(config.data.degradation, target["target_id"], "image", target["working_hw"],
                                       global_seed=config.data.degradation.seed))
        actual = {row["variant_id"]: row for row in by_target[target["target_id"]]}
        require(set(actual) == {row["variant_id"] for row in expected}, "Incomplete or stale degradation plan")
        for plan in expected:
            require(all(actual[plan["variant_id"]].get(k) == v for k, v in plan.items()),
                    "Recorded degradation does not match the effective configuration")
            if target["target_kind"] == "aligned_pair":
                validate_aligned_variant(actual[plan["variant_id"]], target, targets, sources, variants, config)
    for view in views.values():
        require(view["variant_id"] in variants, "View references a missing variant")
        require(view["mode"] in {"fullbody", "face"}, "Unknown image view mode")
        target = targets[variants[view["variant_id"]]["target_id"]]
        h, w = target["working_hw"]
        x1, y1, x2, y2 = view["crop_xyxy"]
        require(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, "Crop is outside the complete working image")
        require(len(view["bucket_hw"]) == 2 and all(type(side) is int and side >= 32 and side % 32 == 0
                                                   for side in view["bucket_hw"]), "Invalid H3 image canvas")
        if view["mode"] == "fullbody":
            box = target["person_boxes"][0]
            require(x1 <= box[0] and y1 <= box[1] and x2 >= box[2] and y2 >= box[3],
                    "Fullbody crop must contain the complete person including the head")
        else:
            require(len(target["face_boxes"]) == 1, "Face view has no detected face")
        by_variant[view["variant_id"]].append(view)
    require(all(by_variant[key] for key in variants), "A prepared variant has no views")
    return {"sources": sources, "targets": targets, "variants": variants, "views": views}


def cache_reference(store, path, stage, owner):
    path = Path(path)
    entry = store.get(path.stem, pin_manifest=owner)
    require(entry is not None and entry["stage"] == stage and Path(entry["absolute_path"]).resolve() == path.resolve(),
            "Manifest cache reference does not match its indexed stage/path")
    return entry


def rgb_array(entry, hw=None):
    array = np.load(entry["absolute_path"], allow_pickle=False)
    require(array.dtype == np.float32 and array.ndim == 3 and array.shape[-1] == 3,
            "Expected one float32 HWC RGB image")
    require(hw is None or list(array.shape[:2]) == list(hw), "Pixel shape differs from geometry")
    require(bool(np.isfinite(array).all()) and float(array.min()) >= 0 and float(array.max()) <= 1,
            "Pixels are outside finite RGB [0,1]")
    return array


def audit_pixel_cache(index, store, owner, *, budget_check=lambda: None):
    entries = {}
    def check(path, stage):
        if path not in entries:
            budget_check()
            entries[path] = cache_reference(store, path, stage, owner)
        return entries[path]
    for target in index["targets"].values():
        rgb_array(check(target["y_path"], "working_targets"), target["working_hw"])
    for variant in index["variants"].values():
        entry = check(variant["x_work_path"], "variants")
        require(entry["key"] == variant["cache_key"] and entry["sha256"] == variant["output_sha256"],
                "Variant cache hash differs from manifest")
        rgb_array(entry, index["targets"][variant["target_id"]]["working_hw"])
        target = index["targets"][variant["target_id"]]
        if target["target_kind"] == "aligned_pair":
            origin = index["targets"][target["pair_contract"]["original_target_id"]]
            audit_aligned_pixels(target, origin, index["sources"][target["asset_id"]], variant,
                rgb_array(entry), rgb_array(check(target["y_path"], "working_targets")),
                rgb_array(check(origin["y_path"], "working_targets")))
    for view in index["views"].values():
        arrays = {}
        for field, role in (("x_crop_path", "x_crop"), ("y_crop_path", "y_crop"),
                            ("scene_x_path", "scene_x"), ("pad_valid_map", "pad_valid_map")):
            entry = check(view[field], "views")
            require(entry["key"] == stage_key("views", view_id=view["view_id"], role=role),
                    "View role/cache relationship changed")
            arrays[role] = (np.load(entry["absolute_path"], allow_pickle=False) if role == "pad_valid_map"
                            else rgb_array(entry, None if role == "scene_x" else view["bucket_hw"]))
        mask = arrays["pad_valid_map"]
        require(mask.dtype == np.float32 and list(mask.shape) == view["bucket_hw"]
                and bool(np.isin(mask, [0, 1]).all()) and bool(mask.any()), "Invalid pixel padding mask")
        require(not arrays["x_crop"][mask == 0].any() and not arrays["y_crop"][mask == 0].any(),
                "Artificial letterbox pixels are not the declared zero fill")
        variant = index["variants"][view["variant_id"]]
        if variant["clean_pair"]:
            require(np.array_equal(arrays["x_crop"], arrays["y_crop"]), "Clean pair is not an exact identity pair")
    return entries


def encoded_pixel_key(pixel_sha256, encoder_contract_id, execution_contract=None):
    # N, optimizer, role, Decoder adapter and ordering cannot change an Encoder input.
    pixel_contract = digest({"npy_sha256": pixel_sha256, "kind": "image", "layout": "float32_hwc_rgb_zero_one",
                             "encoder_input": "B3THW_T1", "binding_version": 2,
                             "encoder_execution": execution_contract})
    return latent_key(pixel_contract, encoder_contract_id)


def bind_image_latents(index, store, owner, encoder_contract_id, encode_pixel, *, execution_contract=None, budget_check=lambda: None,
                       progress=lambda _: None):
    counts = Counter()
    result, keys = [], set()
    for number, view in enumerate(index["views"].values(), 1):
        output = {**view, "record_type": "training_view", "latent_status": "materialized",
                  "encoder_contract_id": encoder_contract_id, "media_kind": "image"}
        variant = index["variants"][view["variant_id"]]
        target = index["targets"][variant["target_id"]]
        source = index["sources"][target["asset_id"]]
        output.update(target_id=target["target_id"], asset_id=source["asset_id"], split=source["split"])
        shape = (1, 24, 1, view["bucket_hw"][0] // 16, view["bucket_hw"][1] // 16)
        for role, field in (("input", "x_crop_path"), ("target", "y_crop_path")):
            budget_check()
            pixel = cache_reference(store, view[field], "views", owner)
            key = encoded_pixel_key(pixel["sha256"], encoder_contract_id, execution_contract)
            entry = store.get(key, pin_manifest=owner)
            if entry is None:
                latent = encode_pixel(rgb_array(pixel, view["bucket_hw"]))
                require(isinstance(latent, np.ndarray) and latent.dtype == np.float32
                        and latent.shape == shape and bool(np.isfinite(latent).all()), "Encoder returned an invalid image latent")
                entry = store.put_array("latents", key, latent, pin_manifest=owner,
                    metadata={"encoder_contract_id": encoder_contract_id, "pixel_sha256": pixel["sha256"],
                              "encoder_execution": execution_contract,
                              "kind": "image", "shape": list(shape), "dtype": "float32"})
                counts["encoded"] += 1
            else:
                counts["cache_hits"] += 1
            require(entry["stage"] == "latents" and entry["metadata"].get("encoder_contract_id") == encoder_contract_id
                    and entry["metadata"].get("pixel_sha256") == pixel["sha256"]
                    and entry["metadata"].get("encoder_execution") == execution_contract, "Stale latent provenance")
            latent = np.load(entry["absolute_path"], allow_pickle=False)
            require(latent.shape == shape and latent.dtype == np.float32 and bool(np.isfinite(latent).all()),
                    "Cached latent shape/dtype/content is invalid")
            output[f"z_{role}_key"] = key
            output[f"z_{role}_path"] = entry["absolute_path"]
            keys.add(key)
        result.append(output)
        output["latent_execution_contract"] = execution_contract
        if number % 20 == 0 or number == len(index["views"]):
            progress({"completed_views": number, "total_views": len(index["views"]), **counts})
    return result, keys, {**counts, "unique_latents": len(keys)}


def select_overfit_views(index, training_views, count=16, seed=42):
    require(type(count) is int and 8 <= count <= 16 and count % 2 == 0, "Select an even number of 8..16 pairs")
    # Equal face/body quota; distinct original images; broad scene/framing coverage.
    candidates = [v for v in training_views if v["split"] == "train"
                  and not index["variants"][v["variant_id"]]["clean_pair"]]
    selected, used_assets = [], set()
    for mode in ("fullbody", "face"):
        coverage = Counter()
        for _ in range(count // 2):
            pool = [v for v in candidates if v["mode"] == mode and v["asset_id"] not in used_assets]
            require(bool(pool), "Insufficient distinct images with both requested overfit view modes")
            def order(view):
                source = Path(index["sources"][view["asset_id"]]["path"])
                scene, framing = source.parent.name, source.stem.rsplit("_", 1)[-1]
                variant = index["variants"][view["variant_id"]]
                return (0 if mode == "fullbody" and framing == "wide" else 1,
                        coverage[scene], variant["resolved_operations"]["q_requested"],
                        digest([seed, view["view_id"]]))
            chosen = min(pool, key=order)
            selected.append(chosen)
            used_assets.add(chosen["asset_id"])
            coverage[Path(index["sources"][chosen["asset_id"]]["path"]).parent.name] += 1
    return selected


def subset_records(index, views):
    variant_ids = {v["variant_id"] for v in views}
    # An AI-only evaluation still needs the original X provenance, but must not
    # silently add its parent's training view to the evaluation sample count.
    pending = list(variant_ids)
    while pending:
        key = pending.pop()
        require(key in index["variants"], "Subset references a missing variant")
        parent = index["variants"][key].get("original_variant_id")
        if parent is not None and parent not in variant_ids:
            require(parent in index["variants"], "Aligned subset is missing its original degradation")
            variant_ids.add(parent)
            pending.append(parent)
    target_ids = {index["variants"][key]["target_id"] for key in variant_ids}
    asset_ids = {index["targets"][key]["asset_id"] for key in target_ids}
    return ([v for k, v in index["sources"].items() if k in asset_ids]
            + [v for k, v in index["targets"].items() if k in target_ids]
            + [v for k, v in index["variants"].items() if k in variant_ids] + views)
