"""M0 image/video preparation, using real locked detectors in the public CLI."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import PIL

from h3ce.budget import PreparationBudget
from h3ce.cache.keys import canonical_json, digest, file_sha256, stage_key
from h3ce.cache.store import CacheStore, assert_separate_paths, atomic_write
from h3ce.data.decode import decode_image, working_canvas
from h3ce.data.degrade import degrade_rgb, plan_variants, processor_contract
from h3ce.data.detect_yolo11 import Yolo11Detectors
from h3ce.data.manifest import write_manifest
from h3ce.data.sample import make_view, view_geometries
from h3ce.data.scan import scan_sources
from h3ce.errors import H3CEError


def prepare_images(config, root: Path, run_dir: Path, *, detectors=None, include_videos=False,
                   source_color_declarations=None):
    """Dependency injection is for isolated CPU tests; CLI always loads YOLO11."""
    start = time.monotonic()
    root, run_dir = Path(root).resolve(), Path(run_dir)
    paths = {key: root / getattr(config.paths, key)
             for key in ("raw", "tmp", "runs", "exports", "models")}
    assert_separate_paths(*paths.values())
    if config.data.mode != "hq_self_degrade":
        raise H3CEError("E_NOT_IMPLEMENTED", "Aligned-pair geometry validation is not implemented yet.")
    if config.data.task != "restoration":
        raise H3CEError("E_TARGET_REQUIRED", "Harmonization requires trusted aligned lighting targets.")
    if not config.data.degradation.materialize:
        raise H3CEError("E_NOT_IMPLEMENTED", "Lazy variant materialization is not implemented yet.")
    if config.cache.quota_gib is not None:
        raise H3CEError("E_NOT_IMPLEMENTED", "Automatic cache quota enforcement is not implemented yet.")
    store = CacheStore(paths["tmp"], protected=[paths[k] for k in ("raw", "runs", "exports", "models")])
    report = {"stage": "prepare_media_m0" if include_videos else "prepare_images_m0", "status": "incomplete", "acceptance": "not_real_model_acceptance",
              "raw_sources": 0, "independent_source_groups": 0, "targets": 0, "degraded_variants": 0,
              "clean_pairs": 0, "views": 0, "cache_hits": 0, "cache_writes": 0, "quarantine": [],
              "source_real_frames": 0, "video_sources": 0, "video_targets": 0, "video_views_with_pixel_changes": 0,
              "notes": ["Source subdirectories are groups; root-level files form one conservative group.",
                        "dHash merges near-duplicate groups conservatively; independent-source review is still needed.",
                        "Prepared pixel views are not yet materialized as H3 latent TrainingViews."]}
    records, references = [], []
    code_hashes = {p: file_sha256(Path(__file__).parent / p) for p in (
        "data/decode.py", "data/sample.py", "data/degrade.py")}
    decode_processor = {"numpy": np.__version__, "pillow": PIL.__version__, "code": code_hashes["data/decode.py"]}
    degradation_processor = {"runtime": processor_contract(), "code": code_hashes["data/degrade.py"]}
    view_processor = {"decode": decode_processor, "code": code_hashes["data/sample.py"]}
    pin_manifest = str(run_dir.resolve())

    def cached_array(stage, key, compute, metadata=None):
        entry = store.get(key, pin_manifest=pin_manifest)
        if entry is None:
            entry = store.put_array(stage, key, compute(), metadata=metadata, pin_manifest=pin_manifest)
            report["cache_writes"] += 1
        else:
            report["cache_hits"] += 1
        references.append(key)
        return entry

    def detect(rgb, key):
        entry = store.get(key, pin_manifest=pin_manifest)
        if entry is None:
            result = detectors.detect(np.floor(rgb * 255 + 0.5).astype(np.uint8))
            entry = store.put_json("detections", key, result, pin_manifest=pin_manifest)
            report["cache_writes"] += 1
        else:
            result = json.loads(Path(entry["absolute_path"]).read_text(encoding="utf-8"))
            report["cache_hits"] += 1
        references.append(key)
        return result

    def materialize_variants(source, target_id, y, geometries, detection_key, budget, *, kind="image", processor=None):
        hw = list(y.shape[-3:-1])
        plans = plan_variants(config.data.degradation, target_id, kind, hw,
                              global_seed=config.data.degradation.seed)
        if not plans:
            raise H3CEError("E_TARGET_REQUIRED", "No synthetic or clean pairs requested.")
        for plan in plans:
            budget.check()
            variant_key = stage_key("variants", variant_id=plan["variant_id"], processor=degradation_processor)
            entry = cached_array("variants", variant_key, lambda: degrade_rgb(y, plan), metadata=plan)
            x = store.read_array(variant_key)
            records.append({"record_type": "degraded_variant", **plan, "cache_key": variant_key,
                "x_work_path": entry["absolute_path"], "output_sha256": entry["sha256"]})
            report["clean_pairs" if plan["clean_pair"] else "degraded_variants"] += 1
            for geometry in geometries:
                view_id = stage_key("views", variant_key=variant_key, detection_key=detection_key,
                    geometry=geometry, sampling=config.data.sampling.model_dump(mode="json"),
                    scene_size=config.model.scene_context.longest_edge, processor=processor or view_processor)
                if kind == "video":
                    from h3ce.data.video import make_video_view
                    positions = geometry["target_frame_positions"]
                    arrays = make_video_view(x[positions], y[positions], geometry, source["original_hw"], config.model.scene_context.longest_edge)
                else:
                    arrays = make_view(x, y, geometry, source["original_hw"], config.model.scene_context.longest_edge)
                view = {"record_type": "prepared_pixel_view", "view_id": view_id, "variant_id": plan["variant_id"],
                    **geometry, "crop_to_original": arrays["crop_to_original"], "latent_status": "not_computed"}
                for array_name, field in (("scene_x", "scene_x_path"), ("x_crop", "x_crop_path"),
                                         ("y_crop", "y_crop_path"), ("pad_valid_map", "pad_valid_map")):
                    key = stage_key("views", view_id=view_id, role=array_name)
                    array_entry = cached_array("views", key, lambda name=array_name: arrays[name])
                    view[field] = array_entry["absolute_path"]
                records.append(view)
                report["views"] += 1

    def video_source(source, budget):
        from h3ce.data.video import prepare_video_clips, video_processor_contract
        from h3ce.data.video_stream import iter_video_targets
        implementation = {name: file_sha256(Path(__file__).parent / name) for name in
                          ("data/video.py", "data/video_stream.py", "data/track.py")}
        native_processor = {"decode": decode_processor, "runtime": video_processor_contract(),
                            "video_code": implementation["data/video.py"], "stream_code": implementation["data/video_stream.py"]}
        video_view_processor = {"image_view": view_processor, "video_code": implementation}
        source["pts"].clear()
        source["shot_detection_targets"] = []
        exact_pts, bases = [], []
        report["video_sources"] += 1
        declaration = (source_color_declarations or {}).get(str(Path(source["path"]).resolve()))
        maximum = config.data.max_independent_frames - report["source_real_frames"]
        if maximum < 1:
            raise H3CEError("E_DATA_LIMIT", "Global source frame limit exhausted; preparation is incomplete.")
        for decoded in iter_video_targets(source["path"], capacity=config.native_temporal.train_clip_frames,
                working_long_edge_max=config.data.working_long_edge_max, max_frames=maximum,
                budget_check=budget.check, source_color_declaration=declaration):
            source["pts"].extend(decoded.pts)
            exact_pts.extend(decoded.metadata["pts_integer"])
            bases.extend(decoded.metadata["time_bases"])
            # Retain decisions for rejected/singleton targets as well as accepted views.
            source["shot_detection_targets"].append(decoded.metadata["shot_detection"])
            report["source_real_frames"] += len(decoded.frames)
            if len(decoded.frames) < 2:
                report["quarantine"].append({"asset_id": source["asset_id"],
                    "source_frame_indices": decoded.source_frame_indices, "reason": "single_frame_at_shot_or_source_tail"})
                continue
            target_id = stage_key("working_targets", asset_id=source["asset_id"], source_sha=source["sha256"],
                color=decoded.color_transform_id, source_frame_indices=decoded.source_frame_indices,
                pts_integer=decoded.metadata["pts_integer"], time_bases=decoded.metadata["time_bases"],
                longest_edge=config.data.working_long_edge_max, processor=native_processor)
            target_entry = cached_array("working_targets", target_id, lambda: decoded.frames)
            y = store.read_array(target_id)
            detection_keys, detections = [], []
            for index, frame in zip(decoded.source_frame_indices, y):
                key = stage_key("detections", target_id=target_id, source_frame_index=index, detector=detectors.contract_id)
                detection_keys.append(key)
                detections.append(detect(frame, key))
            sampling = prepare_video_clips(decoded, detections, config.data.sampling,
                                          clip_frames=config.native_temporal.train_clip_frames, asset_id=source["asset_id"])
            report["quarantine"].extend({"asset_id": source["asset_id"], **row,
                "local_window_shot_index": row.get("shot_index"), "shot_index": decoded.metadata["shot_index"]}
                for row in sampling.quarantined)
            if not sampling.clips:
                continue
            hw = list(y.shape[1:3])
            shot_id = digest([source["asset_id"], "shot", decoded.metadata["shot_index"]])
            records.append({"record_type": "prepared_target", "target_id": target_id, "asset_id": source["asset_id"],
                "shot_id": shot_id, "working_hw": hw, "real_pts": decoded.pts,
                "source_frame_indices": decoded.source_frame_indices, "valid_frames": len(y),
                "target_kind": "hq_self_degrade", "person_boxes": [d["person_boxes"] for d in detections],
                "face_boxes": [d["face_boxes"] for d in detections], "bbox_provenance": "source_detector",
                "geometry_id": digest({"original_hw": source["original_hw"], "working_hw": hw}),
                "y_path": target_entry["absolute_path"], "video_metadata": decoded.metadata})
            report["targets"] += 1
            report["video_targets"] += 1
            geometries = []
            for clip in sampling.clips:
                if not clip.metadata["identical_rgb_frames"]:
                    report["video_views_with_pixel_changes"] += 1
                positions = [decoded.source_frame_indices.index(index) for index in clip.source_frame_indices]
                geometries.extend({**geometry, "shot_id": shot_id, "real_pts": clip.real_pts,
                    "source_frame_indices": clip.source_frame_indices, "valid_frames": clip.valid_frames,
                    "target_frame_positions": positions, "frame_validity": {**clip.metadata,
                        "local_window_shot_index": clip.metadata["shot_index"], "shot_index": decoded.metadata["shot_index"]}}
                    for geometry in clip.geometries)
            materialize_variants(source, target_id, y, geometries, digest(detection_keys), budget,
                                 kind="video", processor=video_view_processor)
            atomic_write(run_dir / "progress.json", canonical_json({"last_asset_id": source["asset_id"], **report}))
        source["pts_integer"] = exact_pts
        source["time_bases"] = bases

    with PreparationBudget(paths["runs"], config.project.budget_seconds) as budget:
        try:
            sources = scan_sources(paths["raw"], validation_fraction=config.data.split.validation_fraction,
                                   seed=config.data.split.seed, maximum=config.data.max_independent_frames,
                                   budget_check=budget.check, include_videos=include_videos,
                                   source_color_declarations=source_color_declarations)
            if config.native_temporal.mode == "finetune" and not any(source["kind"] == "video" for source in sources):
                raise H3CEError("E_REAL_VIDEO_REQUIRED", "Native Decoder tuning requires verified real continuous video.")
            report["raw_sources"] = len(sources)
            report["independent_source_groups"] = len({source["source_group"] for source in sources})
            report["source_splits"] = {split: sum(source["split"] == split for source in sources) for split in ("train", "val", "test")}
            detectors = detectors if detectors is not None else Yolo11Detectors(config, root)
            report["detector_contract_id"] = detectors.contract_id
            for source in sources:
                budget.check()
                if file_sha256(Path(source["path"])) != source["sha256"]:
                    raise H3CEError("E_SOURCE_CHANGED", "Source changed after scanning; restart preparation.")
                if source["kind"] == "video":
                    video_source(source, budget)
                    continue
                report["source_real_frames"] += 1
                if report["source_real_frames"] > config.data.max_independent_frames:
                    raise H3CEError("E_DATA_LIMIT", "Global source frame limit exceeded; preparation is incomplete.")
                target_id = stage_key("working_targets", asset_id=source["asset_id"], source_sha=source["sha256"],
                    color=source["color_transform_id"], longest_edge=config.data.working_long_edge_max, processor=decode_processor)
                target_entry = cached_array("working_targets", target_id, lambda: working_canvas(
                    decode_image(Path(source["path"]))[0], config.data.working_long_edge_max))
                y = store.read_array(target_id)
                hw = list(y.shape[:2])
                detection_key = stage_key("detections", target_id=target_id, detector=detectors.contract_id)
                detections = detect(y, detection_key)
                person, face = detections["person_boxes"], detections["face_boxes"]
                if len(person) != 1 or len(face) > 1:
                    report["quarantine"].append({"asset_id": source["asset_id"], "person_count": len(person),
                        "face_count": len(face), "reason": "missing_or_ambiguous_subject"})
                    continue
                geometry_id = digest({"original_hw": source["original_hw"], "working_hw": hw})
                records.append({"record_type": "prepared_target", "target_id": target_id, "asset_id": source["asset_id"],
                    "shot_id": None, "working_hw": hw, "real_pts": [], "source_frame_indices": [0], "valid_frames": 1,
                    "target_kind": "hq_self_degrade", "person_boxes": person, "face_boxes": face,
                    "bbox_provenance": "source_detector", "geometry_id": geometry_id, "y_path": target_entry["absolute_path"]})
                report["targets"] += 1
                geometries = view_geometries(person, face, hw, source["original_hw"], config.data.sampling, target_id)
                materialize_variants(source, target_id, y, geometries, detection_key, budget)
                # Journal only committed rows; final manifest is published after every source completes.
                atomic_write(run_dir / "progress.json", canonical_json({"last_asset_id": source["asset_id"], **report}))
            if report["targets"] == 0:
                raise H3CEError("E_TARGET_REQUIRED", "All sources were quarantined; no accepted HQ targets.")
            if config.native_temporal.mode == "finetune" and report["video_views_with_pixel_changes"] == 0:
                raise H3CEError("E_REAL_VIDEO_REQUIRED", "No continuous accepted video view contains changing source frames; repeated stills cannot start native adaptation.")
            for source in sources:
                if file_sha256(Path(source["path"])) != source["sha256"]:
                    raise H3CEError("E_SOURCE_CHANGED", "Source changed during preparation; manifest was not published.")
            records[:0] = [{"record_type": "source_asset", **source} for source in sources]
            write_manifest(run_dir / "manifest.jsonl", records, store, references)
            report["status"] = "completed_with_quarantine" if report["quarantine"] else "completed"
        except BaseException as exc:
            report["failure"] = exc.as_dict() if isinstance(exc, H3CEError) else {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - start
            report["budget_used_seconds"] = budget.used
            report["cache"] = store.inspect()
            atomic_write(run_dir / "prepare_report.json", canonical_json(report))
    return report


def prepare_project(config, root, run_dir, *, detectors=None, source_color_declarations=None):
    return prepare_images(config, root, run_dir, detectors=detectors, include_videos=True,
                          source_color_declarations=source_color_declarations)
