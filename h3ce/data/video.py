"""PTS-preserving SDR decoding and conservative preparation of real video clips.

No resampling, interpolation, synthetic motion, identity network, or temporal
neural branch is introduced here. Tail padding is metadata until the H3 bridge
needs it, and therefore cannot become an independent training observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from h3ce.cache.keys import digest
from h3ce.errors import H3CEError
from .decode import resize_rgb, working_canvas
from .sample import make_view, view_geometries
from .track import TRACK_POLICY, associate_subjects, box_union


DECODE_VERSION = "pyav_pts_sdr_v1"
SHOT_POLICY = {"version": "rgb_histogram_cut_v1", "thumbnail_hw": [48, 48], "histogram_bins": 16,
               "minimum_mean_rgb_delta": 0.25, "minimum_histogram_distance": 0.45,
               "kind": "nonlearned_heuristic_requires_review"}
CROP_POLICY = {"version": "fixed_union_greedy_shorten_v1", "maximum_union_to_median_box_area": 1.8,
               "minimum_real_frames": 2, "identity_verified": False}


def _pyav():
    try:
        import av
    except ImportError as exc:
        raise H3CEError("E_DEPENDENCY", "Video decoding requires the locked PyAV runtime.") from exc
    return av


def video_processor_contract():
    av = _pyav()
    return {"decoder": DECODE_VERSION, "pyav": av.__version__,
            "ffmpeg_libraries": {name: list(version) for name, version in av.library_versions.items()},
            "input_timing": "decoded_frame_pts_times_frame_time_base",
            "output_rgb": "float32_srgb_zero_one", "fps_resampling": False,
            "shot_policy": SHOT_POLICY, "tracking_policy": TRACK_POLICY, "crop_policy": CROP_POLICY}


def validate_pts(pts):
    """Accept an arbitrary finite origin, including zero; every dt must be > 0."""
    if not pts or any(value is None or not math.isfinite(float(value)) for value in pts):
        raise H3CEError("E_VIDEO_PTS", "Every decoded frame needs a finite source PTS.")
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise H3CEError("E_VIDEO_PTS", "Source PTS intervals must be strictly positive; timestamps are not repaired.")


def _color_values(frame, codec):
    result = {}
    for name in ("colorspace", "color_primaries", "color_trc", "color_range"):
        value = getattr(frame, name, None)
        unspecified = {2} if name != "color_range" else {0}
        if value is None or int(value) in unspecified:
            codec_value = getattr(codec, name, None)
            if codec_value is not None:
                value = codec_value
        result[name] = None if value is None else int(value)
    return result


def _apply_color_declaration(values, declaration):
    """Apply an explicit provenance-bearing declaration only to missing tags."""
    if declaration is None:
        return values
    required = {"matrix", "transfer", "primaries", "range", "provenance", "color_accuracy_verified"}
    if (not isinstance(declaration, dict) or set(declaration) != required
            or not isinstance(declaration["provenance"], str) or not declaration["provenance"].strip()
            or not isinstance(declaration["color_accuracy_verified"], bool)):
        raise H3CEError("E_COLOR_CONTRACT", "Source color declarations require matrix, transfer, primaries, range, provenance and color_accuracy_verified.")
    choices = {"matrix": {"ITU709": 1, "ITU601": 5}, "transfer": {"bt709": 1, "srgb": 13},
               "primaries": {"bt709": 1}, "range": {"MPEG": 1, "JPEG": 2}}
    names = {"matrix": "colorspace", "transfer": "color_trc", "primaries": "color_primaries", "range": "color_range"}
    resolved = dict(values)
    for name, options in choices.items():
        if declaration[name] not in options:
            raise H3CEError("E_COLOR_CONTRACT", f"Unsupported declared video color {name}.")
        desired, field_name = options[declaration[name]], names[name]
        source = values[field_name]
        unspecified = {None, 0} if field_name == "color_range" else {None, 2}
        equivalent = field_name == "colorspace" and source in {5, 6} and desired == 5
        if source not in unspecified and source != desired and not equivalent:
            raise H3CEError("E_COLOR_CONTRACT", "Explicit color declaration conflicts with a known source tag.",
                           {"field": field_name, "source": source, "declaration": desired})
        resolved[field_name] = desired
    return resolved


def resolve_video_color(frame, codec, source_color_declaration=None):
    """Resolve only transforms implemented below; never relabel HDR as SDR."""
    values = _color_values(frame, codec)
    source_values = dict(values)
    pixel_format = frame.format
    if values["color_trc"] in {16, 18} or values["color_primaries"] in {9, 11, 12, 22}:
        raise H3CEError("E_COLOR_CONTRACT", "HDR/wide-gamut video requires an explicit implemented color transform.", values)
    if any(component.bits > 8 for component in pixel_format.components):
        raise H3CEError("E_COLOR_CONTRACT", "High-depth video requires an explicit implemented color transform.", values)
    if any(component.is_alpha for component in pixel_format.components):
        raise H3CEError("E_COLOR_CONTRACT", "Video alpha requires an explicit compositing contract.")
    if source_color_declaration is not None:
        if pixel_format.is_rgb:
            raise H3CEError("E_COLOR_CONTRACT", "A YUV matrix declaration cannot override a native RGB source.")
        values = _apply_color_declaration(values, source_color_declaration)
    if pixel_format.is_rgb:
        if values["color_primaries"] not in {None, 1, 2} or values["color_trc"] not in {None, 1, 2, 13}:
            raise H3CEError("E_COLOR_CONTRACT", "Unsupported RGB video primaries or transfer.", values)
        return {**values, "matrix": "native_rgb", "range": "full", "transfer":
                "bt709_to_srgb" if values["color_trc"] == 1 else "project_declared_sdr_srgb",
                "pixel_format": pixel_format.name}
    matrices = {1: "ITU709", 5: "ITU601", 6: "ITU601"}
    if (values["colorspace"] not in matrices or values["color_primaries"] != 1
            or values["color_trc"] not in {1, 13} or values["color_range"] not in {1, 2}):
        raise H3CEError("E_COLOR_CONTRACT", "YUV video requires explicit supported SDR matrix, BT.709 primaries, transfer and range tags.", values)
    return {**values, "matrix": matrices[values["colorspace"]],
            "range": "MPEG" if values["color_range"] == 1 else "JPEG",
            "transfer": "bt709_to_srgb" if values["color_trc"] == 1 else "srgb_identity",
            "pixel_format": pixel_format.name, "source_color_tags": source_values,
            "source_color_declaration": source_color_declaration}


def _decode_rgb(frame, contract, *, reformatter=None, reuse_buffers=False):
    kwargs = {"format": "rgb24"}
    if contract["matrix"] != "native_rgb":
        kwargs.update(src_colorspace=contract["matrix"], src_color_range=contract["range"],
                      dst_color_range="JPEG")
    raw = (frame.to_ndarray(**kwargs) if reformatter is None
           else reformatter.reformat(frame, **kwargs).to_ndarray())
    if reuse_buffers:
        # This is a fresh owned float allocation; the decoded source is untouched.
        rgb = raw.astype(np.float32)
        rgb /= 255.0
    else:
        rgb = raw.astype(np.float32) / 255.0
    if contract["transfer"] == "bt709_to_srgb":
        linear = np.where(rgb < 0.081, rgb / 4.5, ((rgb + 0.099) / 1.099) ** (1 / 0.45))
        rgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    if reuse_buffers:
        if rgb.dtype != np.float32:
            raise H3CEError("E_COLOR_CONTRACT", "Float buffer reuse requires the original FP32 arithmetic domain.")
        np.clip(rgb, 0, 1, out=rgb)
        return rgb
    return np.clip(rgb, 0, 1).astype(np.float32)


@dataclass
class DecodedVideoFrame:
    rgb: np.ndarray
    source_frame_index: int
    pts: float
    pts_integer: int
    time_base: tuple[int, int]
    original_hw: tuple[int, int]
    color_contract: dict


def iter_video_frames(path, *, working_long_edge_max=None, max_frames=None, budget_check=None,
                      source_color_declaration=None, reuse_reformatter=False, reuse_buffers=False):
    """Decode in presentation order without a guessed FPS or regenerated PTS.

    Consumers may stream frames to a cache; a frame limit is an explicit failure,
    never an apparently complete shortened source. No audio is decoded or changed.
    """
    av = _pyav()
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise H3CEError("E_VIDEO_LIMIT", "max_frames must be a positive integer.")
    try:
        with av.open(str(Path(path)), mode="r") as container:
            if len(container.streams.video) != 1:
                raise H3CEError("E_VIDEO_STREAM", "Exactly one video stream is required; choose a stream explicitly upstream.")
            stream = container.streams.video[0]
            codec = stream.codec_context
            # The converter belongs to this stream; checks and arithmetic stay the same.
            reformatter = av.video.reformatter.VideoReformatter() if reuse_reformatter else None
            sar = stream.sample_aspect_ratio
            if sar is not None and sar != 1:
                raise H3CEError("E_GEOMETRY", "Non-square video pixels need an explicit geometry transform.")
            previous_time, shape, color = None, None, None
            count = 0
            for index, frame in enumerate(container.decode(stream)):
                if budget_check is not None:
                    budget_check()
                if max_frames is not None and index >= max_frames:
                    raise H3CEError("E_VIDEO_LIMIT", "Source exceeds the explicit frame limit; no partial video accepted.", {"maximum": max_frames})
                if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                    raise H3CEError("E_VIDEO_PTS", "Video frame has no valid source PTS/time base.", {"frame_index": index})
                exact_time = frame.pts * frame.time_base
                if previous_time is not None and exact_time <= previous_time:
                    raise H3CEError("E_VIDEO_PTS", "Source PTS intervals must be strictly positive.", {"frame_index": index})
                previous_time = exact_time
                if getattr(frame, "interlaced_frame", False):
                    raise H3CEError("E_GEOMETRY", "Interlaced video requires an explicit deinterlacing contract.")
                rotation = getattr(frame, "rotation", 0)
                side_data_types = [str(item.type).upper() for item in frame.side_data]
                if rotation or any("DISPLAYMATRIX" in item for item in side_data_types) or stream.metadata.get("rotate", "0") not in {"0", "0.0"}:
                    raise H3CEError("E_GEOMETRY", "Video display rotation needs an explicit geometry transform.")
                if any(any(marker in item for marker in ("MASTERING_DISPLAY", "CONTENT_LIGHT", "DYNAMIC_HDR", "DOVI")) for item in side_data_types):
                    raise H3CEError("E_COLOR_CONTRACT", "HDR side metadata requires an explicit implemented color transform.")
                current_hw = (frame.height, frame.width)
                current_color = resolve_video_color(frame, codec, source_color_declaration)
                if shape is not None and shape != current_hw:
                    raise H3CEError("E_GEOMETRY", "Video dimensions change mid-stream; split sources explicitly.")
                if color is not None and color != current_color:
                    raise H3CEError("E_COLOR_CONTRACT", "Video color contract changes mid-stream.")
                shape, color = current_hw, current_color
                if reuse_buffers:
                    rgb = _decode_rgb(frame, color, reformatter=reformatter, reuse_buffers=True)
                else:
                    rgb = (_decode_rgb(frame, color) if reformatter is None else
                           _decode_rgb(frame, color, reformatter=reformatter))
                if (working_long_edge_max is not None
                        and not (reuse_buffers and max(rgb.shape[:2]) <= working_long_edge_max)):
                    rgb = working_canvas(rgb, working_long_edge_max)
                count += 1
                yield DecodedVideoFrame(rgb, index, float(exact_time), int(frame.pts),
                                        (frame.time_base.numerator, frame.time_base.denominator), current_hw, color)
            if count == 0:
                raise H3CEError("E_DECODE", "Video contains no decodable frames.")
    except H3CEError:
        raise
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        raise H3CEError("E_DECODE", f"Cannot decode video {path}: {exc}") from exc


@dataclass
class DecodedVideo:
    frames: np.ndarray
    pts: list[float]
    source_frame_indices: list[int]
    original_hw: list[int]
    color_transform_id: str
    metadata: dict = field(default_factory=dict)


def decode_video(path, *, working_long_edge_max=None, max_frames=None, budget_check=None,
                  source_color_declaration=None):
    frames = list(iter_video_frames(path, working_long_edge_max=working_long_edge_max,
                                    max_frames=max_frames, budget_check=budget_check,
                                    source_color_declaration=source_color_declaration))
    metadata = {"source_kind": "video", "timing_source": "container_decoded_frame_pts",
                "source_path": str(Path(path).resolve()), "decoder_contract": video_processor_contract(),
                "pts_integer": [frame.pts_integer for frame in frames],
                "time_bases": [list(frame.time_base) for frame in frames],
                "color_contract": frames[0].color_contract, "audio_policy": "source_untouched_not_decoded",
                "real_motion_verified": False, "human_source_review_required": True}
    pts = [frame.pts for frame in frames]
    validate_pts(pts)
    return DecodedVideo(np.stack([frame.rgb for frame in frames]), pts,
                        [frame.source_frame_index for frame in frames], list(frames[0].original_hw),
                        digest({"decoder": DECODE_VERSION, "color": frames[0].color_contract}), metadata)


def detect_shots(frames, *, packed_channels=False):
    """Flag large pixel/histogram changes; this heuristic is explicitly recorded."""
    if len(frames) == 0:
        raise H3CEError("E_DECODE", "Cannot detect shots in an empty video.")
    indices, transitions = [0], []
    previous = resize_rgb(frames[0], SHOT_POLICY["thumbnail_hw"], packed_channels=packed_channels)
    bins = SHOT_POLICY["histogram_bins"]

    def histogram(rgb):
        return np.stack([np.histogram(rgb[..., channel], bins=bins, range=(0, 1))[0]
                         for channel in range(3)]).astype(float) / rgb.shape[0] / rgb.shape[1]

    previous_hist = histogram(previous)
    for index, frame in enumerate(frames[1:], 1):
        current = resize_rgb(frame, SHOT_POLICY["thumbnail_hw"], packed_channels=packed_channels)
        current_hist = histogram(current)
        pixel_delta = float(np.abs(current - previous).mean())
        histogram_distance = float(np.abs(current_hist - previous_hist).sum(axis=1).mean() / 2)
        cut = (pixel_delta >= SHOT_POLICY["minimum_mean_rgb_delta"] and
               histogram_distance >= SHOT_POLICY["minimum_histogram_distance"])
        transitions.append({"frame_index": index, "mean_rgb_delta": pixel_delta,
                            "histogram_distance": histogram_distance, "cut": bool(cut)})
        indices.append(indices[-1] + int(cut))
        previous, previous_hist = current, current_hist
    return indices, {"policy": SHOT_POLICY, "transitions": transitions}


def h3_padding_plan(real_frames):
    if not isinstance(real_frames, int) or real_frames < 2:
        raise H3CEError("E_REAL_VIDEO_REQUIRED", "A video clip needs at least two source frames; single frames use the image path.")
    padded = 5 + 17 * max(0, math.ceil((real_frames - 5) / 17))
    return {"valid_frames": real_frames, "padded_frame_count": padded,
            "tail_padding_frames": padded - real_frames, "padding": "repeat_last_at_bridge_only",
            "valid_frame_mask": [True] * real_frames + [False] * (padded - real_frames),
            "valid_motion_mask": [True] * (real_frames - 1) + [False] * (padded - real_frames),
            "padding_pts": None}


@dataclass
class VideoClip:
    frames: np.ndarray
    shot_id: str
    real_pts: list[float]
    source_frame_indices: list[int]
    person_boxes: list[list[list[float]]]
    face_boxes: list[list[list[float]]]
    geometries: list[dict]
    valid_frames: int
    metadata: dict


@dataclass
class VideoPreparation:
    clips: list[VideoClip]
    quarantined: list[dict]
    metadata: dict


def _union_ratio(boxes):
    union = box_union(boxes)
    area = (union[2] - union[0]) * (union[3] - union[1])
    return float(area / np.median([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]))


def prepare_video_clips(decoded, detections, sampling, *, clip_frames=5, asset_id=None):
    """Split shots/tracks into disjoint continuous clips with fixed crop unions.

    A greedy prefix is shortened before its body or complete-face union exceeds
    1.8 times the median box area. One-frame remainders are quarantined, not
    repeated to impersonate motion. Missing faces preserve fullbody clips.
    """
    frames, pts, source_indices = decoded.frames, decoded.pts, decoded.source_frame_indices
    validate_pts(pts)
    if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.float32 or
            not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1):
        raise H3CEError("E_DECODE", "Decoded video must be finite float32 RGB [T,H,W,3] in [0,1].")
    if len(frames) != len(pts) or len(frames) != len(detections) or len(frames) != len(source_indices):
        raise H3CEError("E_VIDEO_PTS", "Video frames, timestamps, indices and detections must align.")
    if (any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in source_indices) or
            any(right != left + 1 for left, right in zip(source_indices, source_indices[1:]))):
        raise H3CEError("E_REAL_VIDEO_REQUIRED", "Source frame indices must be contiguous; independent images are not video.")
    if decoded.metadata.get("source_kind") != "video" or decoded.metadata.get("timing_source") != "container_decoded_frame_pts":
        raise H3CEError("E_REAL_VIDEO_REQUIRED", "Video preparation requires container-backed source timing provenance.")
    if not isinstance(clip_frames, int) or clip_frames < 5 or (clip_frames - 5) % 17:
        raise H3CEError("E_VIDEO_SHAPE", "Video clip capacity must follow H3 T=17n+5.")
    cfg = sampling.model_dump() if hasattr(sampling, "model_dump") else sampling
    hw = list(frames.shape[1:3])
    shot_indices, shot_report = detect_shots(frames)
    tracks, quarantined = associate_subjects(detections, shot_indices, hw)
    for item in quarantined:
        item["source_frame_index"] = source_indices[item["frame_index"]]
        item["pts"] = pts[item["frame_index"]]
    clips = []
    source_id = asset_id or digest([decoded.metadata.get("source_path"), source_indices, pts])
    for track in tracks:
        cursor = 0
        while cursor < len(track.frame_indices):
            candidates = track.frame_indices[cursor:cursor + clip_frames]
            indices = []
            body_boxes, all_faces, seen_faces = [], [], []
            stop_reason = "capacity_or_track_end"
            for index in candidates:
                body = detections[index]["person_boxes"][0]
                faces = detections[index]["face_boxes"]
                proposed_bodies = body_boxes + [body]
                proposed_faces = all_faces + [faces[0]] if faces and len(all_faces) == len(indices) else []
                too_large = (_union_ratio(proposed_bodies) > CROP_POLICY["maximum_union_to_median_box_area"] or
                             (len(proposed_faces) == len(indices) + 1 and _union_ratio(proposed_faces) > CROP_POLICY["maximum_union_to_median_box_area"]))
                if indices and too_large:
                    stop_reason = "crop_union_too_large"
                    break
                indices.append(index)
                body_boxes = proposed_bodies
                all_faces = proposed_faces
                seen_faces.extend(faces)
            cursor += len(indices)
            if len(indices) < CROP_POLICY["minimum_real_frames"]:
                index = indices[0]
                quarantined.append({"frame_index": index, "source_frame_index": source_indices[index], "pts": pts[index],
                                    "shot_index": track.shot_index, "reason": "fewer_than_two_continuous_real_frames"})
                continue
            body_union = box_union(body_boxes + seen_faces)
            face_union = []
            if len(all_faces) == len(indices):
                face_pixels = [min((box[2] - box[0]) * decoded.original_hw[1] / hw[1],
                                   (box[3] - box[1]) * decoded.original_hw[0] / hw[0]) for box in all_faces]
                if min(face_pixels) >= cfg["min_face_hq_pixels"]:
                    face_union = [box_union(all_faces)]
            current_source_indices = [source_indices[index] for index in indices]
            shot_id = digest([source_id, "shot", track.shot_index])
            geometry_seed = digest([source_id, current_source_indices, "fixed_clip_union"])
            geometries = view_geometries([body_union], face_union, hw, decoded.original_hw, sampling, geometry_seed)
            metadata = {"shot_index": track.shot_index, "stop_reason": stop_reason, "crop_policy": CROP_POLICY,
                        "body_union_to_median_area": _union_ratio(body_boxes), "body_union": body_union,
                        "face_union": face_union, "face_view_requires_detection_in_every_real_frame": True,
                        "bbox_provenance": "source_detector", "real_motion_verified": False,
                        "identical_rgb_frames": bool(all(np.array_equal(frames[indices[0]], frames[index]) for index in indices[1:])),
                        **h3_padding_plan(len(indices))}
            if "pts_integer" in decoded.metadata and "time_bases" in decoded.metadata:
                metadata["pts_integer"] = [decoded.metadata["pts_integer"][index] for index in indices]
                metadata["time_bases"] = [decoded.metadata["time_bases"][index] for index in indices]
            # Tracks and prefixes are contiguous. Slicing keeps the pixel buffer
            # shared with the decoded video instead of copying an entire clip.
            clips.append(VideoClip(frames[indices[0]:indices[-1] + 1], shot_id, [pts[index] for index in indices],
                                   current_source_indices, [detections[index]["person_boxes"] for index in indices],
                                   [detections[index]["face_boxes"] for index in indices], geometries, len(indices), metadata))
    return VideoPreparation(clips, quarantined, {"shot_detection": shot_report, "tracking_policy": TRACK_POLICY,
                            "crop_policy": CROP_POLICY, "source_real_frames": len(frames),
                            "accepted_real_frames": sum(clip.valid_frames for clip in clips),
                            "clip_capacity": clip_frames, "real_motion_verified": False,
                            "human_source_review_required": True})


def make_video_view(x, y, geometry, original_hw, scene_longest_edge=256):
    """Apply one shared geometry to all real frames; scene always reads X."""
    if x.shape != y.shape or x.ndim != 4 or len(x) < 2:
        raise H3CEError("E_GEOMETRY", "Video X/Y must share a nonempty real clip geometry.")
    views = [make_view(xframe, yframe, geometry, original_hw, scene_longest_edge)
             for xframe, yframe in zip(x, y)]
    return {"x_crop": np.stack([view["x_crop"] for view in views]),
            "y_crop": np.stack([view["y_crop"] for view in views]),
            "scene_x": np.stack([view["scene_x"] for view in views]),
            "pad_valid_map": np.stack([view["pad_valid_map"] for view in views]),
            "crop_to_original": views[0]["crop_to_original"]}
