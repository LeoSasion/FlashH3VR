"""Stable target-relative video loss for an explicitly additive correction.

This optional path does not change sealed prediction-based experiments.
It retains the existing temporal eligibility and support normalization rules.
"""
from __future__ import annotations
import torch
from .head_video_mae_loss import target_transition_mae


def target_transition_mae_from_correction(input_rgb, target, correction, support,
                                          pts, shot_ids, *, valid_frames=None,
                                          max_gap_seconds=.1):
    """Compute (X-Y)+correction without cancelling a tiny correction in X+c-Y.

    All tensors are in original frame coordinates. The caller supplies the
    already inverse-warped and feathered correction, not the head-bucket output.
    Source/target are fixed; only correction may carry a trainable graph.
    """
    if (input_rgb.shape != target.shape or input_rgb.shape != correction.shape
            or input_rgb.ndim != 4 or input_rgb.shape[1] != 3):
        raise ValueError('Matching original-coordinate T3HW input, target and correction required')
    if (not input_rgb.is_floating_point() or target.dtype != input_rgb.dtype
            or correction.dtype != input_rgb.dtype
            or input_rgb.device != target.device or correction.device != input_rgb.device
            or input_rgb.requires_grad or target.requires_grad):
        raise ValueError('Matching float devices/dtypes and fixed source/target required')
    if not all(torch.isfinite(t).all() for t in (input_rgb, target, correction)):
        raise ValueError('Finite input, target and correction required')
    residual = (input_rgb - target) + correction
    result = target_transition_mae(residual, torch.zeros_like(target), support, pts, shot_ids,
                                   valid_frames=valid_frames, max_gap_seconds=max_gap_seconds)
    return {**result, 'residual_formulation': 'input_minus_target_plus_pasted_correction'}
