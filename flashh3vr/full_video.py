"""Bounded full-frame SDR video restoration with the adopted Dense head model.

Only real source frames cross the public API.  H3 tail padding lives inside
DenseRestorer, and overlapping windows blend predictions of the *same* frame.
The detector supplies geometry only; it is not an identity model.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from h3ce.data import continuous_head
from h3ce.data.stable_head import POLICY, stabilize_head_boxes
from h3ce.data.video import detect_shots, iter_video_frames
from h3ce.infer.video_output import encode_srgb_h264
from scripts.video_benchmark_math import blend_weights, chunk_plan

from .backend import H3_WEIGHT_SHA256, sha256_file
from .dense import DENSE_WEIGHT_SHA256
from .inference import BUCKETS, DenseRestorer, REAL_VIDEO_FRAMES


FACE_WEIGHT_SHA256 = "6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661"
FACE_RUNTIME = "ultralytics==8.4.142"
MIN_FACE_SIDE = 64


class PinnedFaceDetector:
    """Load one external, SHA-pinned YOLO11 face model without a person model."""

    def __init__(self, weights: str | Path, *, device: str = "cuda:0") -> None:
        from h3ce.data.detect_yolo11 import _get_yolo_class, _validate_architecture

        weights = Path(weights)
        if not weights.is_file() or sha256_file(weights) != FACE_WEIGHT_SHA256:
            raise ValueError("Face detector asset missing or SHA256 differs from the pinned YOLO11 face model")
        factory = _get_yolo_class(FACE_RUNTIME)
        self.model = factory(str(weights), task="detect")
        _, self.class_id = _validate_architecture(
            self.model, {"architecture": "yolo11m-detect-face"}, "face"
        )
        self.device = device

    def detect_frames(self, rgb: np.ndarray) -> list[list[list[float]]]:
        from h3ce.data.detect_yolo11 import _boxes_from_result

        if (rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.float32
                or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1))):
            raise ValueError("Face detector needs finite float32 RGB [T,H,W,3] in [0,1]")
        height, width = rgb.shape[1:3]
        output = []
        for start in range(0, len(rgb), 16):
            batch = [np.uint8(np.clip(frame, 0, 1) * 255 + .5) for frame in rgb[start:start + 16]]
            results = self.model.predict(
                source=[np.ascontiguousarray(frame[:, :, ::-1]) for frame in batch],
                imgsz=960, conf=0.25, iou=0.5, classes=[self.class_id],
                stream=False, save=False, verbose=False, augment=False, half=False,
                device=self.device,
            )
            if len(results) != len(batch):
                raise ValueError("Face detector changed source frame count")
            output.extend(_boxes_from_result(result, class_id=self.class_id, height=height,
                                             width=width, role="face") for result in results)
        return output


def _eligible_face(boxes: Sequence[Sequence[float]], canvas_hw: tuple[int, int]) -> tuple[list[float] | None, str]:
    height, width = canvas_hw
    eligible = []
    for box in boxes:
        if len(box) not in (4, 5) or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("Invalid face detector box")
        x0, y0, x1, y1 = map(float, box[:4])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Face detector box outside source canvas")
        if min(x1 - x0, y1 - y0) >= MIN_FACE_SIDE:
            eligible.append([x0, y0, x1, y1])
    if len(eligible) == 1:
        return eligible[0], "unique_face"
    return None, "missing_face" if not eligible else "ambiguous_faces"


def plan_full_video(face_boxes: Sequence[Sequence[Sequence[float]]], pts: Sequence[float],
                    shot_ids: Sequence[int], canvas_hw: tuple[int, int], *, side: int) -> dict:
    """Freeze per-frame geometry and eligible same-shot real-frame windows."""
    count = len(pts)
    if (side not in BUCKETS or count < 2 or len(face_boxes) != count or len(shot_ids) != count
            or any(not math.isfinite(float(value)) for value in pts)
            or any(right <= left for left, right in zip(pts, pts[1:]))):
        raise ValueError("Need matching real frames, finite increasing PTS and a supported bucket")
    selected, reasons = [], []
    for boxes in face_boxes:
        box, reason = _eligible_face(boxes, canvas_hw)
        selected.append(box)
        reasons.append(reason)
    stable = stabilize_head_boxes(selected, pts, shot_ids, canvas_hw, side=side)
    transforms = [None] * count
    for index, record in enumerate(stable):
        if record is None:
            continue
        transform = continuous_head.head_transform(
            record["used_face_xyxy"], canvas_hw, frame_index=index,
            pts=pts[index], side=side, provenance="stabilized_input_detector",
        )
        x0, y0, x1, y1 = transform["crop_xyxy"]
        a, b, c, d = selected[index]
        if not (x0 <= a and y0 <= b and x1 >= c and y1 >= d):
            raise ValueError("Stabilized crop lost the detected face")
        transforms[index] = transform
    segments, active = [], []
    for index, box in enumerate(selected):
        boundary = (index > 0 and (shot_ids[index] != shot_ids[index - 1]
                    or pts[index] - pts[index - 1] > POLICY["max_gap_seconds"]))
        if boundary or box is None:
            if active:
                segments.append(active)
                active = []
        if box is not None:
            active.append(index)
    if active:
        segments.append(active)
    chunks, used = [], []
    for indices in segments:
        if len(indices) == 1:
            reasons[indices[0]] = "isolated_face_no_video_context"
            transforms[indices[0]] = None
            continue
        used.extend(indices)
        for chunk in chunk_plan([shot_ids[indices[0]]] * len(indices),
                                size=REAL_VIDEO_FRAMES, overlap=5):
            real_frames = chunk["stop"] - chunk["start"]
            context_frames = 5 if real_frames <= 5 else REAL_VIDEO_FRAMES
            chunks.append({**chunk, "start": chunk["start"] + indices[0],
                           "stop": chunk["stop"] + indices[0],
                           "valid_frames": real_frames,
                           "h3_context_frames": context_frames,
                           "padded_frames": context_frames,
                           "padding_frames_added": context_frames - real_frames})
    return {"side": side, "pts": list(map(float, pts)), "shot_ids": list(shot_ids),
            "transforms": transforms, "face_decisions": reasons,
            "used_frames": used, "skipped_frames": {i: reasons[i] for i in range(count) if i not in used},
            "chunks": chunks, "max_gap_seconds": POLICY["max_gap_seconds"],
            "window_overlap_real_frames": 5,
            "short_window_tail_padding": "repeat_last_inside_H3_only; no invented source PTS"}


@torch.no_grad()
def restore_full_video_frames(frames: torch.Tensor, pts: Sequence[float],
                              face_boxes: Sequence[Sequence[Sequence[float]]],
                              shot_ids: Sequence[int], restorer: DenseRestorer, *, side: int,
                              plan: dict | None = None) -> dict:
    """Restore bounded full-scene frames; skipped frames remain bitwise unchanged."""
    if (frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.float32
            or frames.device.type != "cpu" or frames.shape[0] != len(pts)
            or not bool(torch.isfinite(frames).all())
            or bool((frames < 0).any()) or bool((frames > 1).any())):
        raise ValueError("Expected finite CPU float32 RGB [T,3,H,W] real video frames")
    plan = plan or plan_full_video(face_boxes, pts, shot_ids, tuple(frames.shape[-2:]), side=side)
    if plan["side"] != side or plan["pts"] != list(pts) or len(plan["transforms"]) != len(frames):
        raise ValueError("Precomputed geometry does not match these real source frames")
    if not plan["used_frames"]:
        raise ValueError("No eligible two-frame head segment; refusing an unchanged output")
    output = frames.clone()
    pending: dict[int, tuple[torch.Tensor, float]] = {}
    window_reports = []
    chunks = plan["chunks"]
    for chunk_index, chunk in enumerate(chunks):
        start, stop = chunk["start"], chunk["stop"]
        bucket = torch.stack([
            continuous_head.pack_frame(frames[index:index + 1], plan["transforms"][index])[0]
            for index in range(start, stop)
        ])
        model_input = bucket.permute(1, 0, 2, 3)[None].contiguous().to(restorer.device)
        restored = restorer.restore_video_window(model_input, pts=pts[start:stop])
        if restored.shape != model_input.shape or not bool(torch.isfinite(restored).all()):
            raise ValueError("Dense restorer returned an invalid real-frame window")
        native_plan = getattr(restorer, "last_plan", None)
        if (native_plan is not None and "valid_frames" in native_plan
                and (native_plan["valid_frames"] != chunk["valid_frames"]
                     or native_plan["h3_context_frames"] != chunk["h3_context_frames"])):
            raise ValueError("H3 real-frame or padded-context count changed")
        delta = (restored - model_input).detach().to(device="cpu")[0].permute(1, 0, 2, 3)
        window_reports.append({"start": start, "stop": stop,
                               "valid_frames": chunk["valid_frames"],
                               "h3_context_frames": chunk["h3_context_frames"],
                               "padded_frames": chunk["padded_frames"],
                               "padding_frames_added": chunk["padding_frames_added"],
                               "padded_frames_meaning": "total H3 context frames, including real frames",
                               "native_plan": native_plan})
        weights = blend_weights(chunk)
        for offset, index in enumerate(range(start, stop)):
            weighted = delta[offset] * float(weights[offset])
            if index in pending:
                prior, total_weight = pending[index]
                pending[index] = (prior + weighted, total_weight + float(weights[offset]))
            else:
                pending[index] = (weighted, float(weights[offset]))
        next_start = chunks[chunk_index + 1]["start"] if chunk_index + 1 < len(chunks) else len(frames)
        for index in sorted(i for i in pending if i < next_start):
            correction, total_weight = pending.pop(index)
            if not math.isclose(total_weight, 1.0, abs_tol=1e-6, rel_tol=1e-6):
                raise ValueError("Same-frame window overlap did not partition unit weight")
            pasted = continuous_head.paste_delta(
                frames[index:index + 1], (correction / total_weight)[None],
                plan["transforms"][index],
            )
            output[index:index + 1] = pasted
    if pending:
        raise ValueError("Unfinished overlap predictions")
    return {"prediction": output, "plan": plan, "window_reports": window_reports}


def restore_full_video_file(source: str | Path, destination: str | Path, *,
                            h3_weights: str | Path, dense_weights: str | Path,
                            face_weights: str | Path, side: int = 448,
                            max_frames: int, working_long_edge: int = 0,
                            device: str = "cuda:0") -> dict:
    """Decode, detect, restore, paste, encode and record a finite SDR MP4."""
    import av

    source, destination = Path(source), Path(destination)
    receipt = destination.with_suffix(".flashh3vr.json")
    if (not source.is_file() or destination.suffix.lower() != ".mp4"
            or destination.resolve() == source.resolve()
            or destination.exists() or receipt.exists()):
        raise ValueError("Source must exist and output MP4/receipt must be new distinct paths")
    if (type(max_frames) is not int or max_frames < 2
            or type(working_long_edge) is not int or
            (working_long_edge != 0 and working_long_edge < 256)
            or side not in BUCKETS):
        raise ValueError("Use supported bucket, long edge 0 or >=256 and explicit max_frames >=2")
    with av.open(str(source)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio):
            raise ValueError("Exactly one video stream and no audio stream are supported")
        rate = container.streams.video[0].average_rate
        if rate is None or Fraction(rate) <= 0:
            raise ValueError("Source needs a positive nominal video rate")
    decoded = list(iter_video_frames(source, working_long_edge_max=working_long_edge or None,
                                     max_frames=max_frames))
    if len(decoded) < 2:
        raise ValueError("Full-video restoration requires at least two real frames")
    arrays = np.stack([frame.rgb for frame in decoded])
    if arrays.shape[1] % 2 or arrays.shape[2] % 2:
        raise ValueError("Working canvas dimensions must be even for H.264 output")
    pts = [frame.pts for frame in decoded]
    shots, shot_report = detect_shots(list(arrays))
    detector = PinnedFaceDetector(face_weights, device=device)
    faces = detector.detect_frames(arrays)
    plan = plan_full_video(faces, pts, shots, tuple(arrays.shape[1:3]), side=side)
    if not plan["used_frames"]:
        raise ValueError("No eligible head segment; no unchanged-copy output was written")
    restorer = DenseRestorer(h3_weights=h3_weights, dense_weights=dense_weights, device=device)
    frames = torch.from_numpy(np.ascontiguousarray(arrays.transpose(0, 3, 1, 2)))
    result = restore_full_video_frames(frames, pts, faces, shots, restorer, side=side, plan=plan)
    integer_pts = [frame.pts_integer for frame in decoded]
    time_bases = [list(frame.time_base) for frame in decoded]
    encode_srgb_h264(destination, result["prediction"], integer_pts, time_bases, rate=rate)
    public_plan = {key: value for key, value in plan.items() if key != "transforms"}
    receipt.write_text(json.dumps({
        "source": str(source.resolve()), "output": str(destination.resolve()),
        "source_sha256": sha256_file(source),
        "h3_weight_sha256": H3_WEIGHT_SHA256,
        "dense_weight_sha256": DENSE_WEIGHT_SHA256,
        "face_weight_sha256": FACE_WEIGHT_SHA256,
        "source_frame_indices": [frame.source_frame_index for frame in decoded],
        "source_pts_integer": integer_pts, "source_time_bases": time_bases,
        "source_color": decoded[0].color_contract,
        "face_detections_xyxy_conf": faces,
        "source_canvas_hw": list(decoded[0].original_hw),
        "working_output_canvas_hw": list(arrays.shape[1:3]),
        "working_long_edge_max": working_long_edge or None,
        "shot_detection": shot_report,
        "plan": public_plan, "transforms": plan["transforms"],
        "window_reports": result["window_reports"],
        "output_encoding": "H.264 CRF18 limited BT709 SDR, no audio",
        "scope": "finite full-frame video, eligible head regions only; no long-video/FPS certification",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(destination.resolve()), "receipt": str(receipt.resolve()),
            "source_frames": len(decoded), "restored_frames": len(plan["used_frames"]),
            "skipped_frames": len(plan["skipped_frames"]), "windows": len(plan["chunks"])}
