"""Strict, checksummed still-image batches for frozen-Encoder bootstrap.

Samples already contain the microbatch dimension. Do not apply a DataLoader's
default collator: it would introduce a second batch axis. No model is loaded here.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from h3ce.cache.keys import file_sha256, stage_key
from h3ce.cache.store import CacheStore, assert_no_links
from h3ce.data.degrade import plan_variants
from h3ce.data.aligned import validate_aligned_target, validate_aligned_variant, audit_aligned_pixels
from h3ce.data.image_training import encoded_pixel_key
from h3ce.data.sample import make_view
from h3ce.errors import H3CEError


def _require(condition, message):
    if not condition:
        raise H3CEError("E_TRAINING_DATA", message)


def _indexed(records, kind, field):
    rows = [r for r in records if r.get("record_type") == kind]
    _require(all(isinstance(r.get(field), str) and r[field] for r in rows), f"Missing {field}")
    result = {r[field]: r for r in rows}
    _require(len(result) == len(rows), f"Duplicate {kind} identifiers")
    return result


def _rgb(array, hw=None):
    _require(array.dtype == np.float32 and array.ndim == 3 and array.shape[-1] == 3,
             "Expected float32 HWC RGB cache")
    _require(hw is None or list(array.shape[:2]) == list(hw), "RGB geometry differs from manifest")
    _require(np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1,
             "RGB cache must be finite and in [0,1]")
    return array


def _image_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).unsqueeze(2)


def _hw(value, *, canvas=False):
    _require(isinstance(value, list) and len(value) == 2
             and all(type(side) is int and side > 0 for side in value), "Invalid image dimensions")
    if canvas:
        _require(all(side >= 32 and side % 32 == 0 for side in value), "Canvas must be aligned to 32")
    return value


def projected_box_mask(boxes, bucket_hw, crop_to_original, original_hw, working_hw, valid):
    """Project detector boxes from the working image onto pixel centers.

    Boxes are geometry supervision only. The full person region includes its head.
    """
    h, w = bucket_hw
    yy, xx = np.indices((h, w), dtype=np.float64)
    grid = np.stack((xx + .5, yy + .5, np.ones_like(xx)), axis=-1)
    projected = grid @ np.asarray(crop_to_original, dtype=np.float64).T
    px, py = projected[..., 0] / projected[..., 2], projected[..., 1] / projected[..., 2]
    result = np.zeros((h, w), dtype=np.float32)
    sx, sy = original_hw[1] / working_hw[1], original_hw[0] / working_hw[0]
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        result[(px >= x1*sx) & (px < x2*sx) & (py >= y1*sy) & (py < y2*sy)] = 1
    return result * valid


class CachedImageDataset(Dataset):
    """Load an immutable training manifest; verify every consumed cache object.

    ``verify_pixels`` additionally reproduces the declared crop/letterbox and
    scene from its complete degraded X and trusted target Y. The default is on;
    hashes, role binding, latent provenance and shape checks are never optional.
    Subsets (such as the 16-pair overfit manifest) may contain fewer variants than
    the full preparation, but every included recipe must match the config.
    """

    def __init__(self, manifest_path, cache, *, config=None, raw_root=None, split="train",
                 expected_manifest_sha256=None, expected_encoder_contract_id=None,
                 expected_execution_contract=None, verify_pixels=True):
        assert_no_links(Path(manifest_path))
        self.manifest_path = Path(manifest_path).resolve()
        _require(self.manifest_path.is_file(), "Training manifest does not exist")
        self.manifest_sha256 = file_sha256(self.manifest_path)
        _require(expected_manifest_sha256 is None or self.manifest_sha256 == expected_manifest_sha256,
                 "Training manifest SHA256 differs from the expected checkpoint/preparation")
        try:
            records = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, ValueError) as exc:
            raise H3CEError("E_TRAINING_DATA", f"Cannot read training manifest: {exc}") from exc
        _require(all(isinstance(row, dict) for row in records), "Manifest rows must be objects")
        self.sources = _indexed(records, "source_asset", "asset_id")
        self.targets = _indexed(records, "prepared_target", "target_id")
        self.variants = _indexed(records, "degraded_variant", "variant_id")
        all_views = _indexed(records, "training_view", "view_id")
        _require(all((self.sources, self.targets, self.variants, all_views)), "Incomplete training manifest")
        _require(len(records) == sum(len(d) for d in (self.sources, self.targets, self.variants, all_views)),
                 "Unexpected manifest record type; prepared pixels are not materialized training latents")
        _require(split in {None, "train", "val", "test"}, "Unknown requested split")
        self.store = cache if isinstance(cache, CacheStore) else CacheStore(Path(cache), create=False)
        self.verify_pixels = verify_pixels
        self.config = config
        if config is not None:
            _require(config.native_temporal.mode == "frozen" and config.native_temporal.decoder_adapter is None,
                     "Image bootstrap requires frozen original H3 VAE")
            _require(not config.vae.encoder_trainable and config.cache.cache_encoder_latents,
                     "Materialized image latents require a fixed Encoder")
            if raw_root is None:
                raw_root = Path(config.project.root).resolve() / config.paths.raw
        self.raw_root = Path(raw_root).resolve() if raw_root is not None else None
        self._validate_records(all_views)
        self.views = [row for row in all_views.values() if split is None or row["split"] == split]
        _require(bool(self.views), f"No materialized image views in requested split {split}")
        self.encoder_contract_id = self.views[0]["encoder_contract_id"]
        self.execution_contract = self.views[0]["latent_execution_contract"]
        _require(expected_encoder_contract_id is None or expected_encoder_contract_id == self.encoder_contract_id,
                 "Cached Encoder contract differs from current H3")
        _require(expected_execution_contract is None or expected_execution_contract == self.execution_contract,
                 "Cached Encoder execution environment differs from the requested contract")
        _require(all(v["encoder_contract_id"] == self.encoder_contract_id
                     and v["latent_execution_contract"] == self.execution_contract for v in all_views.values()),
                 "Mixed Encoder or execution contracts in one manifest")
        self.source_groups = {split_name: sorted({s["source_group"] for s in self.sources.values()
                                                 if s["split"] == split_name}) for split_name in ("train", "val", "test")}

    def _validate_records(self, all_views):
        groups, hashes = defaultdict(set), defaultdict(set)
        try:
            for source in self.sources.values():
                _require(source["kind"] == "image" and source["pts"] == [], "Still-image dataset cannot masquerade as video")
                _require(source["split"] in {"train", "val", "test"}, "Unknown source split")
                _require(isinstance(source["source_group"], str) and source["source_group"], "Source group is missing")
                _hw(source["original_hw"])
                path = Path(source["path"])
                assert_no_links(path)
                _require(self.raw_root is None or path.resolve().is_relative_to(self.raw_root), "Source escapes configured raw root")
                _require(path.is_file() and file_sha256(path) == source["sha256"], "Original source failed checksum")
                groups[source["source_group"]].add(source["split"])
                hashes[source["sha256"]].add(source["split"])
            _require(all(len(s) == 1 for s in [*groups.values(), *hashes.values()]),
                     "One source group or duplicate content crosses train/validation/test splits")
            for target in self.targets.values():
                _require(target["asset_id"] in self.sources, "Target references missing source")
                _require(target["valid_frames"] == 1 and target["real_pts"] == [] and target["source_frame_indices"] == [0],
                         "Image target contains video timing")
                _require(target["target_kind"] in {"hq_self_degrade", "aligned_pair"}, "Target lacks trusted HQ/alignment provenance")
                if target["target_kind"] == "aligned_pair":
                    validate_aligned_target(target, self.targets, self.sources)
                _require(target["bbox_provenance"] in {"source_detector", "input_detector", "propagated"}, "Missing box provenance")
                h, w = _hw(target["working_hw"])
                _require(len(target["person_boxes"]) == 1 and len(target["face_boxes"]) <= 1, "Ambiguous subject in training target")
                for box in target["person_boxes"] + target["face_boxes"]:
                    _require(len(box) >= 4 and np.isfinite(box).all()
                             and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h, "Invalid detector box")
            for variant in self.variants.values():
                _require(variant["target_id"] in self.targets, "Variant references missing target")
                target = self.targets[variant["target_id"]]
                if target["target_kind"] == "aligned_pair":
                    validate_aligned_variant(variant, target, self.targets, self.sources, self.variants, self.config)
                if self.config is not None and target["target_kind"] == "hq_self_degrade":
                    plans = plan_variants(self.config.data.degradation, target["target_id"], "image", target["working_hw"],
                                          global_seed=self.config.data.degradation.seed)
                    expected = next((p for p in plans if p["variant_id"] == variant["variant_id"]), None)
                    _require(expected is not None and all(variant.get(k) == v for k, v in expected.items()),
                             "Degradation plan differs from the effective training config")
            for view in all_views.values():
                for field in ("x_crop_path", "y_crop_path", "scene_x_path", "pad_valid_map",
                              "z_input_key", "z_target_key", "z_input_path", "z_target_path"):
                    _require(isinstance(view.get(field), str) and view[field], f"Missing TrainingView field {field}")
                _require(view["variant_id"] in self.variants, "View references missing variant")
                target = self.targets[self.variants[view["variant_id"]]["target_id"]]
                source = self.sources[target["asset_id"]]
                _require(view["asset_id"] == source["asset_id"] and view["target_id"] == target["target_id"]
                         and view["split"] == source["split"], "View changes its source/target/split")
                _require(view["media_kind"] == "image" and view["latent_status"] == "materialized", "Image latents are not materialized")
                _require(isinstance(view["encoder_contract_id"], str) and view["encoder_contract_id"], "Missing Encoder contract")
                _require(isinstance(view["latent_execution_contract"], dict) and view["latent_execution_contract"],
                         "Missing explicit Encoder execution contract")
                _require(view["mode"] in {"fullbody", "face"}, "Unknown view mode")
                _hw(view["bucket_hw"], canvas=True)
                if self.config is not None:
                    sampling = self.config.data.sampling
                    buckets = sampling.debug_buckets_hw if sampling.profile == "debug" else sampling.balanced_buckets_hw
                    _require(view["bucket_hw"] in buckets, "View canvas differs from active sampling buckets")
                h, w = target["working_hw"]
                x1, y1, x2, y2 = view["crop_xyxy"]
                _require(all(type(v) is int for v in (x1, y1, x2, y2)) and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h,
                         "Invalid declared integer crop geometry")
                matrix = np.asarray(view["crop_to_original"], dtype=np.float64)
                _require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
                         and np.array_equal(matrix[2], [0, 0, 1]) and np.linalg.det(matrix) > 0, "Invalid crop coordinate transform")
                if view["mode"] == "fullbody":
                    box = target["person_boxes"][0]
                    _require(x1 <= box[0] and y1 <= box[1] and x2 >= box[2] and y2 >= box[3],
                             "Fullbody crop must include the complete detected person and head")
                else:
                    _require(len(target["face_boxes"]) == 1, "Face view lacks detected face")
        except (KeyError, TypeError, ValueError) as exc:
            raise H3CEError("E_TRAINING_DATA", f"Malformed training manifest: {exc}") from exc

    def __len__(self):
        return len(self.views)

    def _entry(self, path, stage, *, key=None, sha256=None):
        path = Path(path)
        assert_no_links(path)
        entry = self.store.get(path.stem)
        _require(entry is not None and entry["stage"] == stage and Path(entry["absolute_path"]).resolve() == path.resolve(),
                 "Cache reference differs from its committed stage/path")
        _require(key is None or entry["key"] == key, "Cache role/key differs from manifest")
        _require(sha256 is None or entry["sha256"] == sha256, "Cache SHA256 differs from manifest")
        return entry

    @staticmethod
    def _array(entry):
        try:
            return np.load(entry["absolute_path"], allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise H3CEError("E_TRAINING_DATA", f"Cannot read array cache: {exc}") from exc

    def __getitem__(self, index):
        _require(file_sha256(self.manifest_path) == self.manifest_sha256, "Manifest changed after dataset construction")
        view = self.views[index]
        variant = self.variants[view["variant_id"]]
        target = self.targets[view["target_id"]]
        source = self.sources[view["asset_id"]]
        arrays, entries = {}, {}
        for role, field in (("x_crop", "x_crop_path"), ("y_crop", "y_crop_path"),
                            ("scene_x", "scene_x_path"), ("pad_valid_map", "pad_valid_map")):
            entry = self._entry(view[field], "views", key=stage_key("views", view_id=view["view_id"], role=role))
            entries[role], arrays[role] = entry, self._array(entry)
        for role in ("x_crop", "y_crop", "scene_x"):
            _rgb(arrays[role], None if role == "scene_x" else view["bucket_hw"])
        valid = arrays["pad_valid_map"]
        _require(valid.dtype == np.float32 and list(valid.shape) == view["bucket_hw"]
                 and np.isin(valid, [0, 1]).all() and valid.any(), "Invalid pixel padding mask")
        _require(not arrays["x_crop"][valid == 0].any() and not arrays["y_crop"][valid == 0].any(),
                 "Artificial letterbox pixels differ from declared zero fill")
        scene_scale = min(1., 256 / max(target["working_hw"]))
        expected_scene_hw = [max(1, int(side*scene_scale+.5)) for side in target["working_hw"]]
        _require(list(arrays["scene_x"].shape[:2]) == expected_scene_hw, "Scene must retain complete degraded X aspect ratio")
        if variant["clean_pair"]:
            _require(np.array_equal(arrays["x_crop"], arrays["y_crop"]), "Clean pair differs from identity")
        if self.verify_pixels:
            xentry = self._entry(variant["x_work_path"], "variants", key=variant["cache_key"], sha256=variant["output_sha256"])
            yentry = self._entry(target["y_path"], "working_targets", key=target["target_id"])
            if target["target_kind"] == "aligned_pair":
                origin = self.targets[target["pair_contract"]["original_target_id"]]
                origin_entry = self._entry(origin["y_path"], "working_targets", key=origin["target_id"])
                audit_aligned_pixels(target, origin, source, variant, self._array(xentry),
                                     self._array(yentry), self._array(origin_entry))
            expected = make_view(_rgb(self._array(xentry), target["working_hw"]),
                                 _rgb(self._array(yentry), target["working_hw"]), view, source["original_hw"])
            for role in arrays:
                _require(np.array_equal(arrays[role], expected[role]), f"{role} does not reproduce declared X/Y geometry")
            _require(np.array_equal(view["crop_to_original"], expected["crop_to_original"]),
                     "Crop coordinate matrix differs from pixel geometry")
        latents = {}
        for role, pixel_role in (("input", "x_crop"), ("target", "y_crop")):
            key = encoded_pixel_key(entries[pixel_role]["sha256"], self.encoder_contract_id, self.execution_contract)
            _require(view[f"z_{role}_key"] == key, "Latent key differs from input pixels/Encoder/execution contract")
            entry = self._entry(view[f"z_{role}_path"], "latents", key=key)
            metadata = entry["metadata"]
            _require(metadata.get("encoder_contract_id") == self.encoder_contract_id
                     and metadata.get("pixel_sha256") == entries[pixel_role]["sha256"]
                     and metadata.get("encoder_execution") == self.execution_contract
                     and metadata.get("kind") == "image", "Stale latent provenance")
            array = self._array(entry)
            shape = (1, 24, 1, view["bucket_hw"][0]//16, view["bucket_hw"][1]//16)
            _require(array.dtype == np.float32 and array.shape == shape and np.isfinite(array).all()
                     and metadata.get("shape") == list(shape) and metadata.get("dtype") == "float32", "Invalid normalized image latent")
            latents[f"z_{role}"] = torch.from_numpy(array)
        masks = {name: torch.from_numpy(projected_box_mask(target[field], view["bucket_hw"], view["crop_to_original"],
                  source["original_hw"], target["working_hw"], valid))[None, None, None]
                 for name, field in (("person_mask", "person_boxes"), ("face_mask", "face_boxes"))}
        return {**latents, **masks, "x": _image_tensor(arrays["x_crop"]), "y": _image_tensor(arrays["y_crop"]),
                "scene": _image_tensor(arrays["scene_x"]), "valid": torch.from_numpy(valid)[None, None, None],
                "geometry": torch.tensor(view["crop_to_original"], dtype=torch.float32)[None],
                "original_hw": torch.tensor(source["original_hw"], dtype=torch.float32)[None],
                "bucket_hw": tuple(view["bucket_hw"]), "view_id": view["view_id"], "mode": view["mode"],
                "asset_id": view["asset_id"], "source_group": source["source_group"], "split": source["split"],
                "encoder_contract_id": self.encoder_contract_id, "latent_execution_contract": self.execution_contract,
                "clean_pair": variant["clean_pair"],
                "target_kind": target["target_kind"],
                "supervision_group": ("ai_paired" if target["target_kind"] == "aligned_pair"
                                      else "original_clean" if variant["clean_pair"] else "original_degraded")}

    def audit(self, *, progress=lambda _: None, budget_check=lambda: None):
        """Verify all selected views without loading H3 or starting optimization."""
        modes = defaultdict(int)
        for number in range(len(self)):
            budget_check()
            sample = self[number]
            modes[sample["mode"]] += 1
            progress({"checked_views": number + 1, "total_views": len(self)})
        return {"views": len(self), "modes": dict(modes), "manifest_sha256": self.manifest_sha256,
                "encoder_contract_id": self.encoder_contract_id, "execution_contract": self.execution_contract,
                "source_groups": self.source_groups, "pixel_geometry_reproduced": bool(self.verify_pixels)}
