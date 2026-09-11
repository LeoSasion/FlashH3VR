"""Conservative multiple-face continuity using geometry only, without reidentification."""
import math
from .track import box_iou, validate_boxes

POLICY = {'version': 'mutually_unique_face_iou_v1', 'minimum_iou': .3,
          'minimum_face_side': 64., 'max_gap_seconds': .1,
          'missing_or_ambiguous': 'terminate_without_propagation',
          'identity_verified': False, 'reidentification': False}


def associate_face_geometry(records, pts, canvas_hw):
    """Retain only mutually unique associations; new faces start new tracks.

    Detection order and confidence do not select a winner. Crossing, duplicate,
    missing, or disconnected boxes never produce a forced identity assignment.
    A complete track can be selected for native-video processing by its actual
    frame indices; partial tracks remain explicitly recorded.
    """
    if not records or len(records) != len(pts):
        raise ValueError('Matching nonempty detections and PTS required')
    if any(not math.isfinite(t) for t in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('Strictly increasing finite PTS required')
    tracks, events, active = [], [], {}
    previous_shot = None

    def finish(track_id, frame, reason):
        tracks[track_id]['termination'] = {'at_frame': frame, 'reason': reason}

    for i, record in enumerate(records):
        if record.get('bbox_provenance') != 'input_detector':
            raise ValueError('Input YOLO11 detection provenance required')
        if record['frame_index'] != i or record['pts'] != pts[i]:
            raise ValueError('Frame indices or PTS changed')
        shot = record['shot_id']; boxes = record['all_faces']
        validate_boxes(boxes, canvas_hw)
        if i and (shot != previous_shot or pts[i] - pts[i-1] > POLICY['max_gap_seconds']):
            reason = 'shot_change' if shot != previous_shot else 'pts_gap'
            for tid in active:
                finish(tid, i, reason)
            active = {}
        previous_shot = shot
        eligible = sorted([(j, b) for j, b in enumerate(boxes)
                           if min(b[2]-b[0], b[3]-b[1]) >= POLICY['minimum_face_side']],
                          key=lambda item: tuple(item[1][:4]))
        for j, b in enumerate(boxes):
            if min(b[2]-b[0], b[3]-b[1]) < POLICY['minimum_face_side']:
                events.append({'frame_index': i, 'detection_index': j, 'reason': 'face_below_minimum_side'})
        candidates = {tid: [k for k, (_, b) in enumerate(eligible)
                            if box_iou(previous, b) >= POLICY['minimum_iou']]
                      for tid, previous in active.items()}
        owners = {k: [tid for tid, hits in candidates.items() if k in hits] for k in range(len(eligible))}
        following, used, blocked = {}, set(), set()
        for tid, hits in candidates.items():
            if len(hits) == 1 and len(owners[hits[0]]) == 1:
                k = hits[0]; j, box = eligible[k]; used.add(k)
                entry = {'frame_index': i, 'pts': pts[i], 'detection_index': j,
                         'face_xyxy': list(map(float, box[:4])),
                         'previous_iou': box_iou(active[tid], box)}
                tracks[tid]['entries'].append(entry); following[tid] = box
            else:
                reason = 'ambiguous_geometry' if hits else 'missing_or_overlap_lost'
                finish(tid, i, reason); blocked.update(hits)
                events.append({'frame_index': i, 'track_id': tid, 'reason': reason,
                               'candidate_detection_indices': [eligible[k][0] for k in hits]})
        for k, (j, box) in enumerate(eligible):
            if k in used:
                continue
            overlaps = [other for other, (_, b) in enumerate(eligible)
                        if other != k and box_iou(box, b) >= POLICY['minimum_iou']]
            if k in blocked or overlaps:
                events.append({'frame_index': i, 'detection_index': j, 'reason': 'ambiguous_new_detection'})
                continue
            tid = len(tracks)
            tracks.append({'track_id': tid, 'shot_id': shot, 'identity_verified': False,
                'entries': [{'frame_index': i, 'pts': pts[i], 'detection_index': j,
                             'face_xyxy': list(map(float, box[:4])), 'previous_iou': None}],
                'termination': None})
            following[tid] = box
        active = following
    for tid in active:
        finish(tid, len(records), 'end_of_sequence')
    return {'policy': dict(POLICY), 'tracks': tracks, 'events': events,
            'complete_track_ids': [t['track_id'] for t in tracks
                                   if [e['frame_index'] for e in t['entries']] == list(range(len(records)))]}
