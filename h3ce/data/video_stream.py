"""Bounded, detector-independent video targets: shot cuts then fixed capacity.

The canonical whole-frame degradation target does not change when detections or
view policies change. Geometry may shorten views within that same degraded clip.
"""
from __future__ import annotations

import numpy as np

from h3ce.cache.keys import digest
from h3ce.errors import H3CEError
from .video import DecodedVideo, SHOT_POLICY, detect_shots, iter_video_frames, video_processor_contract


def iter_video_targets(path, *, capacity=5, working_long_edge_max=2048, max_frames=None,
                       budget_check=None, source_color_declaration=None):
    if type(capacity) is not int or capacity < 5 or (capacity - 5) % 17:
        raise H3CEError("E_VIDEO_SHAPE", "Video target capacity must follow T=17n+5.")
    buffer, transitions, previous, shot_index, shot_origin = [], [], None, 0, None

    def target(items, shot, decisions, origin):
        color = items[0].color_contract
        return DecodedVideo(np.stack([item.rgb for item in items]), [item.pts for item in items],
            [item.source_frame_index for item in items], list(items[0].original_hw),
            digest({"decoder": video_processor_contract()["decoder"], "color": color}),
            {"source_kind": "video", "timing_source": "container_decoded_frame_pts",
             "source_path": str(path), "pts_integer": [item.pts_integer for item in items],
             "time_bases": [list(item.time_base) for item in items], "color_contract": color,
             "shot_index": shot, "target_capacity": capacity,
             "target_sampling": "fixed_capacity_inside_pixel_detected_shot",
             "shot_detection": {"policy": dict(SHOT_POLICY), "shot_index": shot,
                 "shot_start_source_frame_index": origin[0], "shot_start_pts": origin[1],
                 "transitions": list(decisions)},
             "real_motion_verified": False, "audio_policy": "source_untouched_not_decoded"})

    for frame in iter_video_frames(path, working_long_edge_max=working_long_edge_max,
            max_frames=max_frames, budget_check=budget_check, source_color_declaration=source_color_declaration):
        decision = None
        if previous is not None:
            _, scores = detect_shots([previous.rgb, frame.rgb])
            score = scores["transitions"][0]
            # Each decision belongs to its destination frame. This keeps the
            # cross-buffer/cross-shot edge exactly once, with global provenance.
            decision = {"from_source_frame_index": previous.source_frame_index,
                "to_source_frame_index": frame.source_frame_index,
                "from_pts": previous.pts, "to_pts": frame.pts,
                "from_pts_integer": previous.pts_integer, "to_pts_integer": frame.pts_integer,
                "from_time_base": list(previous.time_base), "to_time_base": list(frame.time_base),
                "mean_rgb_delta": score["mean_rgb_delta"],
                "histogram_distance": score["histogram_distance"], "cut": score["cut"]}
        cut = decision is not None and decision["cut"]
        if cut:
            if buffer:
                yield target(buffer, shot_index, transitions, shot_origin)
                buffer, transitions = [], []
            shot_index += 1
        if shot_origin is None or cut:
            shot_origin = (frame.source_frame_index, frame.pts)
        buffer.append(frame)
        if decision is not None:
            transitions.append(decision)
        previous = frame
        if len(buffer) == capacity:
            yield target(buffer, shot_index, transitions, shot_origin)
            buffer, transitions = [], []
    if buffer:
        yield target(buffer, shot_index, transitions, shot_origin)
