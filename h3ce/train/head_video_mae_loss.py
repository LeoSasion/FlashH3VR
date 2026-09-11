"""Explicit absolute target frame-difference supervision for the MAE research path.

This is a loss on real video, not a temporal network or a smoothing filter.
The caller supplies original-canvas predictions and nonlearned head support.
"""
from __future__ import annotations
import math
import torch


def target_transition_mae(prediction, target, support, pts, shot_ids, *,
                           valid_frames=None, max_gap_seconds=.1):
    """Equal-weight valid transitions; each union-head mask is area-normalized.

    No pair spans cuts, missing frames, repeated padding, or a declared time gap.
    Support is T1HW in source coordinates, not a moving crop coordinate system.
    PTS are validated and preserved; differences are per transition, not velocity.
    """
    if (prediction.ndim != 4 or prediction.shape[1] != 3 or prediction.shape != target.shape
            or prediction.shape[0] < 2 or support.shape != (prediction.shape[0],1,*prediction.shape[-2:])):
        raise ValueError('Expected matching real-video T3HW and original-canvas T1HW support')
    n=prediction.shape[0]
    if (prediction.device != target.device or prediction.device != support.device
            or not prediction.is_floating_point() or target.dtype != prediction.dtype
            or not support.is_floating_point() or target.requires_grad or support.requires_grad):
        raise ValueError('Matching float devices/dtypes and fixed target/support required')
    if (len(pts) != n or len(shot_ids) != n or any(not math.isfinite(t) for t in pts)
            or any(b <= a for a,b in zip(pts,pts[1:]))):
        raise ValueError('Strictly increasing source PTS and one shot id per frame required')
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError('Positive finite gap required')
    valid=[True]*n if valid_frames is None else list(valid_frames)
    if len(valid) != n or any(type(v) is not bool for v in valid):
        raise ValueError('Explicit boolean real-frame validity required')
    if (not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or not torch.isfinite(support).all() or torch.any(support < 0) or torch.any(support > 1)):
        raise ValueError('Finite pixels and support in [0,1] required')
    indices=[i for i in range(1,n) if valid[i-1] and valid[i]
             and shot_ids[i-1] == shot_ids[i] and pts[i]-pts[i-1] <= max_gap_seconds]
    if not indices:raise ValueError('No valid real within-shot transitions')
    current=torch.tensor(indices,device=prediction.device,dtype=torch.long)
    mask=torch.maximum(support[current],support[current-1]).to(prediction.dtype)
    area=mask.flatten(1).sum(1)
    if torch.any(area <= 0):raise ValueError('A selected transition has no head support')
    residual=prediction-target
    error=residual[current]-residual[current-1]
    # Absolute error matches the validation metric; exact zero uses torch abs subgradient0.
    per_pair=(error.abs()*mask).flatten(1).sum(1)/(3*area)
    return {'total':per_pair.mean(),'per_transition':per_pair,'pair_indices':[(i-1,i) for i in indices],
            'support_area':area,'eligible_transitions':len(indices),'error_function':'absolute','pts_units':'seconds; per-transition difference'}
