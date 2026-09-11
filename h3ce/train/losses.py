"""Mask-normalized restoration losses, with no clamp of training predictions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from h3ce.errors import H3CEError


def _require(condition, message):
    if not condition:
        raise H3CEError("E_LOSS_CONTRACT", message)


def charbonnier(error, epsilon=.001):
    _require(math.isfinite(epsilon) and epsilon > 0, "Charbonnier epsilon must be positive")
    return torch.sqrt(error.float().square() + epsilon**2)


def region_error(error, weight, epsilon=1e-12):
    """Mean over weighted scalar elements, invariant to mask size or channel count."""
    _require(error.ndim == weight.ndim and error.shape[0] == weight.shape[0]
             and error.shape[2:] == weight.shape[2:] and weight.shape[1] in {1, error.shape[1]},
             "Region weights must match batch/time/space and have one or matching channels")
    _require(torch.isfinite(weight).all() and (weight >= 0).all(), "Region weights must be finite and nonnegative")
    weight = weight.float().expand_as(error)
    return (weight * error.float()).sum() / weight.sum().clamp_min(epsilon)


def _valid(valid, reference):
    _require(reference.ndim == 5 and valid.ndim == 5 and valid.shape[1] == 1
             and valid.shape[0] == reference.shape[0] and valid.shape[2:] == reference.shape[2:],
             "Expected B,C,T,H,W with B,1,T,H,W padding mask")
    _require(torch.isfinite(valid).all() and (valid >= 0).all() and (valid <= 1).all()
             and valid.sum() > 0, "Invalid or empty valid-pixel mask")
    return valid.float()


def latent_loss(prediction, target, valid, epsilon=.001):
    _require(prediction.shape == target.shape and prediction.ndim == 5 and prediction.shape[1] == 24,
             "Expected matching B,24,Tz,Hl,Wl normalized latents")
    _require(torch.isfinite(prediction).all() and torch.isfinite(target).all(), "Nonfinite normalized latent loss input")
    _require(valid.ndim == 5 and valid.shape[0] == prediction.shape[0] and valid.shape[1] == 1,
             "Invalid latent padding weights")
    _require(torch.isfinite(valid).all() and (valid >= 0).all() and (valid <= 1).all() and valid.sum() > 0,
             "Invalid latent padding mask")
    _require(prediction.shape[2] == valid.shape[2], "Image bootstrap cannot guess temporal latent mask correspondence")
    weights = F.interpolate(valid.float(), size=prediction.shape[2:], mode="area")
    return region_error(charbonnier(prediction-target, epsilon), weights)


def srgb_to_linear(value):
    # Extend the SDR transfer curve outside [0,1], preserving out-of-range errors.
    # Guard the unused power branch against negative fractional-power NaNs.
    value = value.float()
    return torch.where(value <= .04045, value / 12.92,
                       ((value + .055) / 1.055).clamp_min(0).pow(2.4))


def _spatial(value):
    b, c, t, h, w = value.shape
    return value.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)


def _restore(value, b, t):
    return value.reshape(b, t, *value.shape[1:]).permute(0, 2, 1, 3, 4)


def _gaussian(value, sigma):
    radius = max(1, math.ceil(3*sigma))
    _require(radius < min(value.shape[-2:]), "Image is too small for the declared reflect Gaussian kernel")
    coordinate = torch.arange(-radius, radius+1, device=value.device, dtype=torch.float32)
    kernel = torch.exp(-.5*(coordinate/sigma).square())
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    value = F.conv2d(F.pad(value.float(), (radius, radius, 0, 0), mode="reflect"),
                     kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(F.pad(value, (0, 0, radius, radius), mode="reflect"),
                    kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def lighting_target_loss(prediction, target, valid, epsilon=.001):
    """Compare low frequencies to trusted Y in linear RGB, never to the input X.

    Normalized masked filtering prevents artificial letterbox pixels from leaking
    into valid boundary pixels. Filtering is spatial only and independent per T.
    """
    _require(prediction.shape == target.shape and prediction.ndim == 5 and prediction.shape[1] == 3,
             "Lighting loss requires matching RGB videos/images")
    _require(torch.isfinite(prediction).all() and torch.isfinite(target).all(), "Nonfinite lighting loss input")
    valid = _valid(valid, prediction)
    b, _, t, h, w = prediction.shape
    weights = _spatial(valid)
    x, y = _spatial(srgb_to_linear(prediction)), _spatial(srgb_to_linear(target))
    losses = []
    for reference_sigma in (4., 16., 32.):
        sigma = reference_sigma * min(h, w) / 512
        normalizer = _gaussian(weights, sigma).clamp_min(1e-12)
        low_x = _gaussian(x*weights, sigma) / normalizer
        low_y = _gaussian(y*weights, sigma) / normalizer
        losses.append(region_error(charbonnier(_restore(low_x-low_y, b, t), epsilon), valid))
    return torch.stack(losses).mean()


class ApplicationLoss(nn.Module):
    """Whole valid RGB plus equally weighted, independently normalized box terms.

    ``rgb`` = global RGB + person-box RGB + face-box RGB for regions provided and
    visible. This makes supervision explicit without inventing new config fields;
    the outer configured RGB coefficient applies to their sum. Fullbody regions
    always retain the head. Missing LPIPS must be resolved to weight zero upstream.
    """

    def __init__(self, config_losses, *, perceptual=None):
        super().__init__()
        self.weights = config_losses.model_dump() if hasattr(config_losses, "model_dump") else dict(config_losses)
        self.perceptual = perceptual
        _require(self.weights.get("region_normalization", "weight_sum") == "weight_sum", "Unknown region normalization")
        for name in ("rgb", "latent", "perceptual", "lighting_target"):
            _require(name in self.weights and math.isfinite(self.weights[name]) and self.weights[name] >= 0,
                     f"Invalid {name} loss weight")
        _require(self.weights["perceptual"] == 0 or perceptual is not None,
                 "LPIPS is unavailable: set resolved training.losses.perceptual explicitly to 0")
        if perceptual is not None:
            perceptual.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.perceptual is not None:
            self.perceptual.eval()
        return self

    def forward(self, prediction, target, z_prediction, z_target, valid, person_mask=None, face_mask=None):
        _require(prediction.shape == target.shape and prediction.ndim == 5 and prediction.shape[1] == 3,
                 "Application loss requires matching B,3,T,H,W RGB")
        _require(torch.isfinite(prediction).all() and torch.isfinite(target).all()
                 and torch.isfinite(z_prediction).all() and torch.isfinite(z_target).all(), "Nonfinite loss input")
        valid = _valid(valid, prediction)
        epsilon = self.weights.get("charbonnier_epsilon", .001)
        error = charbonnier(prediction-target, epsilon)
        rgb = region_error(error, valid)
        regions = {}
        for name, mask in (("person", person_mask), ("face", face_mask)):
            if mask is not None:
                _require(mask.shape == valid.shape and torch.isfinite(mask).all() and (mask >= 0).all()
                         and (mask <= 1).all(), "Invalid additional box supervision mask")
                regions[name] = region_error(error, mask*valid)
                rgb = rgb + regions[name]
        latent = latent_loss(z_prediction, z_target, valid, epsilon)
        lighting = lighting_target_loss(prediction, target, valid, epsilon) if self.weights["lighting_target"] else rgb.new_zeros(())
        perceptual = rgb.new_zeros(())
        if self.weights["perceptual"]:
            # LPIPS only sees each image's valid rectangle; no artificial padding.
            values = []
            for x, y, mask in zip(_spatial(prediction), _spatial(target), _spatial(valid)):
                points = torch.nonzero(mask[0] > 0)
                _require(points.numel() > 0, "Perceptual frame has no valid pixels")
                y1, x1 = points.amin(dim=0).tolist()
                y2, x2 = (points.amax(dim=0)+1).tolist()
                _require(bool((mask[0, y1:y2, x1:x2] == 1).all()), "LPIPS requires a rectangular valid crop")
                values.append(self.perceptual(x[None, :, y1:y2, x1:x2]*2-1,
                                              y[None, :, y1:y2, x1:x2]*2-1).float().mean())
            perceptual = torch.stack(values).mean()
        total = self.weights["rgb"]*rgb + self.weights["latent"]*latent + self.weights["perceptual"]*perceptual + self.weights["lighting_target"]*lighting
        return {"total": total, "rgb": rgb, "latent": latent, "lighting_target": lighting,
                "perceptual": perceptual, **{f"rgb_{name}": value for name, value in regions.items()}}
