"""Offline five-frame geometric stabilization; no learned temporal model."""
import math
import numpy as np
from h3ce.data.head_bucket import head_transform

POLICY = {'version':'local_linear_head_v1','radius':2,'max_gap_seconds':.1,
          'maximum_center_deviation_fraction':.1,'maximum_log_size_deviation':.08,
          'expansion_xy':[1.5,2.0], 'future_frames':2}


def stabilize_head_boxes(boxes, pts, shot_ids, canvas_hw, *, side=256):
    """Return explicit raw/used geometry; None splits tracks without propagation.

    Center and log(width/height) use weighted local linear regression in real
    PTS. Large departures use the observed box; raw face coverage is required.
    Two future frames are needed: this is an offline, bounded-lookahead method.
    """
    n=len(boxes);h,w=canvas_hw
    if n!=len(pts) or n!=len(shot_ids) or side not in (256,512):
        raise ValueError('Length or head size mismatch')
    if any(not math.isfinite(t) for t in pts) or any(b<=a for a,b in zip(pts,pts[1:])):
        raise ValueError('Strictly increasing finite source PTS required')
    params={};segments=[];active=[]
    for i,box in enumerate(boxes):
        if i and (shot_ids[i]!=shot_ids[i-1] or pts[i]-pts[i-1]>POLICY['max_gap_seconds']):
            if active:segments.append(active);active=[]
        if box is None:
            if active:segments.append(active);active=[]
            continue
        if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Invalid raw box')
        a,b,c,d=map(float,box)
        if not (0<=a<c<=w and 0<=b<d<=h):raise ValueError('Raw box outside canvas')
        params[i]=np.array([(a+c)/2,(b+d)/2,math.log(c-a),math.log(d-b)])
        active.append(i)
    if active:segments.append(active)
    output=[None]*n
    for segment in segments:
        for position,i in enumerate(segment):
            indices=segment[max(0,position-2):position+3]
            times=np.asarray([pts[j]-pts[i] for j in indices],dtype=np.float64)
            values=np.stack([params[j] for j in indices])
            weights=np.asarray([3-abs(j-i) for j in indices],dtype=np.float64)
            if len(indices)==1: fitted=params[i].copy()
            else:
                design=np.stack([np.ones_like(times),times],axis=1)
                fitted=np.linalg.lstsq(design*np.sqrt(weights[:,None]),values*np.sqrt(weights[:,None]),rcond=None)[0][0]
            reason='local_linear'
            distance=np.linalg.norm(fitted[:2]-params[i][:2])
            if (distance>POLICY['maximum_center_deviation_fraction']*min(math.exp(params[i][2]),math.exp(params[i][3]))
                    or np.max(abs(fitted[2:]-params[i][2:]))>POLICY['maximum_log_size_deviation']):
                fitted=params[i].copy();reason='large_change_keep_observation'
            cx,cy,lw,lh=fitted;sw,sh=math.exp(lw),math.exp(lh)
            used=[max(0.,cx-sw/2),max(0.,cy-sh/2),min(float(w),cx+sw/2),min(float(h),cy+sh/2)]
            transform=head_transform(used,canvas_hw,frame_index=i,pts=pts[i],side=side,provenance='stabilized_input_detector')
            a,b,c,d=transform['crop_xyxy'];ra,rb,rc,rd=boxes[i]
            if not (a<=ra and b<=rb and c>=rc and d>=rd):
                used=list(map(float,boxes[i]));reason='coverage_guard_keep_observation'
                transform=head_transform(used,canvas_hw,frame_index=i,pts=pts[i],side=side,provenance='stabilized_input_detector')
            output[i]={'policy':dict(POLICY),'frame_index':i,'pts':float(pts[i]),'shot_id':shot_ids[i],
                'raw_bbox_provenance':'input_detector','raw_face_xyxy':list(map(float,boxes[i])),
                'used_face_xyxy':used,'contributors':indices,'decision':reason,'transform':transform}
    return output


def validate_stable_records(records, pts, canvas_hw, *, side):
    if not records or any(r is None for r in records):raise ValueError('Split missing/ambiguous detections before native head H3')
    if any(r.get('raw_bbox_provenance')!='input_detector' for r in records):raise ValueError('Input detection provenance required')
    shots=[r['shot_id'] for r in records]
    if any(s!=shots[0] for s in shots):raise ValueError('Split shots before native head H3')
    expected=stabilize_head_boxes([r['raw_face_xyxy'] for r in records],pts,shots,canvas_hw,side=side)
    if expected!=records:raise ValueError('Stable trajectory/geometry/PTS changed')
    return [r['transform'] for r in records]
