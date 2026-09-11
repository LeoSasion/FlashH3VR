"""Spatial restoration with the same frozen H3 decoder on both delta branches."""
from __future__ import annotations

import math
import torch

from h3ce.errors import H3CEError
from h3ce.vae.bridge import FrameMeta, LatentBatch


def outer_box_feather(valid, fraction=0.05):
    """Fade the outer valid rectangle, including letterboxed crop boundaries."""
    if valid.ndim != 5 or valid.shape[1] != 1 or not 0 < fraction <= 0.5:
        raise H3CEError("E_MASK_CONTRACT", "Expected a B1THW rectangular validity mask")
    result = torch.zeros_like(valid, dtype=torch.float32)
    for b in range(valid.shape[0]):
        for t in range(valid.shape[2]):
            mask = valid[b, 0, t]
            if not torch.all((mask == 0) | (mask == 1)):
                raise H3CEError("E_MASK_CONTRACT", "Pixel validity must be binary")
            points = torch.nonzero(mask, as_tuple=False)
            if not len(points):
                raise H3CEError("E_MASK_CONTRACT", "No valid crop pixels")
            y0, x0 = points.min(0).values.tolist()
            y1, x1 = (points.max(0).values + 1).tolist()
            if int(mask.sum()) != (y1-y0)*(x1-x0):
                raise H3CEError("E_MASK_CONTRACT", "Validity must describe one outer rectangle, not a skin mask")
            height, width = y1-y0, x1-x0
            y = torch.arange(height, device=valid.device, dtype=torch.float32)
            x = torch.arange(width, device=valid.device, dtype=torch.float32)
            dy = torch.minimum(y, height-1-y) / max(1., height*fraction)
            dx = torch.minimum(x, width-1-x) / max(1., width*fraction)
            result[b, 0, t, y0:y1, x0:x1] = torch.minimum(dy[:, None], dx[None, :]).clamp(0, 1)
    return result


def decoded_delta(x_base, positive, negative, feather, strength=1.0):
    if not math.isfinite(strength) or positive.shape != x_base.shape or negative.shape != x_base.shape:
        raise H3CEError("E_OUTPUT_CONTRACT", "Decoded branches must match the input canvas and strength must be finite")
    # Deliberately no clamp: out-of-range errors remain visible to the loss.
    return x_base + strength * feather * (positive - negative)


def move_sample(sample, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}


def refine_latent(model, sample, *, autocast_enabled=False):
    device_type = sample["z_input"].device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        delta = model(sample["z_input"], sample["scene"], sample["geometry"], sample["original_hw"])
    return sample["z_input"].float() + delta.float(), delta


def restore_pixels(bridge, sample, z_prediction, *, grad, strength=1.0):
    pack = bridge.current_codec_pack()
    if pack.trainable or any(p.requires_grad for p in bridge.backend.model.parameters()):
        raise H3CEError("E_FROZEN_VAE_REQUIRED", "Image bootstrap freezes the entire native H3 VAE")
    latent = LatentBatch(sample["z_input"], FrameMeta("image", (0.,)), 1, 1,
                         tuple(sample["bucket_hw"]), sample["encoder_contract_id"])
    # grad=True is essential on the positive path during pixel training.
    # Both branches use the exact same pack and numerical implementation.
    positive = bridge.decode_latent(latent.with_tensor(z_prediction), grad=grad, codec_pack=pack)
    negative = bridge.decode_latent(latent, grad=grad, codec_pack=pack)
    return decoded_delta(sample["x"], positive, negative, outer_box_feather(sample["valid"]), strength), pack
