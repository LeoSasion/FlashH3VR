"""Conservative single-subject association using box IoU only.

This is geometric continuity, not identity recognition. An ambiguous frame or an
association failure is quarantined and resets continuity; it is never bridged by
an invented box or an identity embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from h3ce.errors import H3CEError


TRACK_POLICY = {"version": "iou_single_subject_v1", "minimum_iou": 0.1,
                "missing_person": "quarantine_and_reset", "ambiguous_subject": "quarantine_and_reset",
                "association_failure": "quarantine_transition_and_reset", "identity_verified": False}


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def box_union(boxes):
    if not boxes:
        raise H3CEError("E_GEOMETRY", "Cannot form a crop union without source boxes.")
    result = [min(box[0] for box in boxes), min(box[1] for box in boxes),
              max(box[2] for box in boxes), max(box[3] for box in boxes)]
    if all(len(box) >= 5 for box in boxes):
        result.append(min(box[4] for box in boxes))
    return result


def validate_boxes(boxes, hw):
    height, width = hw
    for box in boxes:
        if len(box) not in (4, 5) or not all(math.isfinite(value) for value in box):
            raise H3CEError("E_GEOMETRY", "Video boxes must be finite xyxy with optional confidence.")
        x1, y1, x2, y2 = box[:4]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise H3CEError("E_GEOMETRY", "Video detector box falls outside the working canvas.")
        if len(box) == 5 and not 0 <= box[4] <= 1:
            raise H3CEError("E_GEOMETRY", "Video box confidence is outside [0,1].")


@dataclass
class SubjectTrack:
    shot_index: int
    frame_indices: list[int]


def associate_subjects(detections, shot_indices, working_hw):
    """Return continuous tracks and quarantines, using local frame positions.

    Source frame numbering and PTS are kept by the video sampler. Exactly one
    person and zero or one geometrically associated face are accepted per frame.
    """
    if len(detections) != len(shot_indices):
        raise H3CEError("E_GEOMETRY", "Detection and shot counts differ.")
    tracks, quarantined = [], []
    active, previous_person, previous_shot = None, None, None
    for index, (detected, shot) in enumerate(zip(detections, shot_indices)):
        if shot != previous_shot:
            active, previous_person = None, None
        previous_shot = shot
        if detected.get("bbox_provenance") != "source_detector":
            raise H3CEError("E_GEOMETRY", "Training video boxes must come from a declared source detector.")
        people, faces = detected["person_boxes"], detected["face_boxes"]
        validate_boxes(people, working_hw)
        validate_boxes(faces, working_hw)
        reason = None
        if len(people) != 1 or len(faces) > 1:
            reason = "missing_or_ambiguous_subject"
        elif faces:
            face, person = faces[0], people[0]
            center = ((face[0] + face[2]) / 2, (face[1] + face[3]) / 2)
            if not (person[0] <= center[0] <= person[2] and person[1] <= center[1] <= person[3]):
                reason = "face_not_geometrically_associated_with_person"
        if reason is None and previous_person is not None:
            overlap = box_iou(previous_person, people[0])
            if overlap < TRACK_POLICY["minimum_iou"]:
                reason = "geometry_association_lost"
        if reason:
            quarantined.append({"frame_index": index, "shot_index": shot, "reason": reason,
                                "person_count": len(people), "face_count": len(faces)})
            active, previous_person = None, None
            continue
        if active is None:
            active = SubjectTrack(shot, [])
            tracks.append(active)
        active.frame_indices.append(index)
        previous_person = people[0]
    return tracks, quarantined
