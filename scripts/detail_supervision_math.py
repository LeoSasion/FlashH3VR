"""Candidate spatial-detail diagnostics; not an enabled or validated training loss.

Work in sRGB working pixels, without clamping. This does not replace the existing
linear-RGB lighting objective. Each B/T frame is cropped to its binary rectangular
valid area before filtering; artificial padding never enters the Gaussian. Sigma
uses that rectangle's short edge, and only the interior at least one kernel radius
from its boundary contributes. There is no temporal filtering or model mutation.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F

from h3ce.errors import H3CEError


def _require(condition, message):
    if not condition:
        raise H3CEError("E_DETAIL_DIAGNOSTIC", message)


def _positive(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value) and value > 0, f"{name} must be finite and positive")


def _floating_tensor(value, name):
    _require(isinstance(value, torch.Tensor) and value.is_floating_point()
             and value.numel() > 0, f"{name} must be a nonempty floating tensor")
    _require(bool(torch.isfinite(value).all()), f"{name} contains nonfinite values")


def gaussian_lowpass_rectangle(value, *, sigma):
    """Filter an already-cropped NCHW rectangle, with reflect and truncate=3.

    Half/bfloat16 arithmetic is promoted to float32; float64 is retained for
    numerical gradient checks. Callers must not pass artificial padding here.
    """
    _floating_tensor(value, "rectangle")
    _require(value.ndim == 4 and min(value.shape) > 0, "Rectangle must have NCHW shape")
    _positive(sigma, "sigma")
    _require(math.isfinite(3 * sigma), "Gaussian radius overflowed")
    radius = math.ceil(3 * sigma)
    _require(radius < min(value.shape[-2:]), "Rectangle is too small for reflect Gaussian")
    value = value if value.dtype == torch.float64 else value.float()
    position = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-0.5 * (position / sigma).square())
    _require(bool(torch.isfinite(kernel).all()) and bool(kernel.sum() > 0), "Invalid Gaussian kernel")
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode="reflect"),
                     horizontal, groups=channels)
    return F.conv2d(F.pad(value, (0, 0, radius, radius), mode="reflect"),
                    vertical, groups=channels)


def _validate_masks(prediction, valid, person_mask, face_mask):
    expected = (prediction.shape[0], 1, *prediction.shape[2:])
    for name, mask in (("valid", valid), ("person", person_mask), ("face", face_mask)):
        if mask is None and name != "valid":
            continue
        _require(isinstance(mask, torch.Tensor) and mask.shape == expected
                 and mask.device == prediction.device, f"{name} mask must match B1THW and device")
        _require(mask.dtype == torch.bool or mask.is_floating_point(), f"{name} mask must be floating or bool")
        _require(bool(torch.isfinite(mask).all()) and bool((mask >= 0).all())
                 and bool((mask <= 1).all()), f"{name} mask must be finite in [0,1]")
        if name == "valid":
            _require(bool(((mask == 0) | (mask == 1)).all()), "Valid mask must be binary rectangular")
        _require(not mask.requires_grad, f"{name} mask is metadata and must not require gradients")


def detail_loss_components(prediction, target, valid, person_mask=None, face_mask=None, *,
                           sigma_reference=2.0, reference_short_edge=512, epsilon=.001):
    """Return independent global/person/face normalized terms and their total.

    Shapes are B3THW with B1THW masks. Every B/T valid mask must be one nonempty
    filled rectangle. Region masks may be fractional; a region with no interior
    support contributes zero. Means aggregate weighted scalar pixels over B/T,
    rather than equally weighting differently sized frames. Charbonnier's constant
    epsilon floor is retained. No result here proves useful detail restoration.
    """
    _floating_tensor(prediction, "prediction")
    _floating_tensor(target, "target")
    _require(prediction.ndim == 5 and prediction.shape[1] == 3
             and target.shape == prediction.shape and target.device == prediction.device,
             "Expected matching B3THW prediction/target on one device")
    for name, value in (("sigma_reference", sigma_reference),
                        ("reference_short_edge", reference_short_edge), ("epsilon", epsilon)):
        _positive(value, name)
    _validate_masks(prediction, valid, person_mask, face_mask)
    dtype = torch.float64 if torch.float64 in (prediction.dtype, target.dtype) else torch.float32
    prediction, target = prediction.to(dtype=dtype), target.to(dtype=dtype)
    masks = {"global": valid}
    if person_mask is not None:
        masks["person"] = person_mask
    if face_mask is not None:
        masks["face"] = face_mask
    numerators = {name: prediction.new_zeros(()) for name in masks}
    denominators = {name: prediction.new_zeros(()) for name in masks}
    for batch in range(prediction.shape[0]):
        for frame in range(prediction.shape[2]):
            points = torch.nonzero(valid[batch, 0, frame] == 1)
            _require(points.numel() > 0, "Every frame needs a nonempty valid rectangle")
            y1, x1 = points.amin(dim=0).tolist()
            y2, x2 = (points.amax(dim=0) + 1).tolist()
            _require(bool((valid[batch, 0, frame, y1:y2, x1:x2] == 1).all()),
                     "Valid mask must be a filled rectangle without holes")
            sigma = sigma_reference * min(y2 - y1, x2 - x1) / reference_short_edge
            _require(math.isfinite(sigma) and sigma > 0 and math.isfinite(3 * sigma),
                     "Scaled Gaussian sigma/radius is not finite and positive")
            radius = math.ceil(3 * sigma)
            _require(min(y2 - y1, x2 - x1) > 2 * radius,
                     "Valid rectangle has no interior beyond the Gaussian radius")
            predicted_crop = prediction[batch:batch+1, :, frame, y1:y2, x1:x2]
            target_crop = target[batch:batch+1, :, frame, y1:y2, x1:x2]
            predicted_high = predicted_crop - gaussian_lowpass_rectangle(predicted_crop, sigma=sigma)
            target_high = target_crop - gaussian_lowpass_rectangle(target_crop, sigma=sigma)
            residual = (predicted_high - target_high)[..., radius:-radius, radius:-radius]
            error = torch.sqrt(residual.square() + epsilon**2)
            _require(bool(torch.isfinite(error).all()), "Nonfinite candidate detail error")
            for name, mask in masks.items():
                weight = mask[batch:batch+1, :, frame,
                              y1+radius:y2-radius, x1+radius:x2-radius].to(dtype=dtype)
                numerators[name] = numerators[name] + (weight * error).sum()
                denominators[name] = denominators[name] + weight.sum() * prediction.shape[1]
    terms = {name: numerator / denominators[name].clamp_min(torch.finfo(dtype).tiny)
             for name, numerator in numerators.items()}
    terms["total"] = sum(terms.values(), prediction.new_zeros(()))
    _require(all(bool(torch.isfinite(value)) for value in terms.values()), "Candidate detail reduction overflowed")
    return terms


def detail_loss(prediction, target, valid, person_mask=None, face_mask=None, *,
                sigma_reference=2.0, reference_short_edge=512, epsilon=.001):
    """Candidate diagnostic scalar, without changing the project's training loss."""
    return detail_loss_components(prediction, target, valid, person_mask, face_mask,
                                  sigma_reference=sigma_reference,
                                  reference_short_edge=reference_short_edge,
                                  epsilon=epsilon)["total"]


def gradient_pair_statistics(first, second):
    """Read detached gradient norm/dot/cosine in float64; zero norm gives None.

    These statistics do not call autograd or perform an optimizer update. The
    caller determines the space being measured; latent frequency is not assumed
    equivalent to image-space frequency.
    """
    _floating_tensor(first, "first gradient")
    _floating_tensor(second, "second gradient")
    _require(first.shape == second.shape and first.device == second.device,
             "Gradient tensors must have identical shapes and devices")
    first, second = first.detach().double().reshape(-1), second.detach().double().reshape(-1)
    first_norm, second_norm = float(torch.linalg.vector_norm(first)), float(torch.linalg.vector_norm(second))
    dot = float(torch.dot(first, second))
    _require(all(math.isfinite(x) for x in (first_norm, second_norm, dot)), "Gradient statistics overflowed")
    cosine = None if first_norm == 0 or second_norm == 0 else max(-1., min(1., dot / first_norm / second_norm))
    return {"first_norm": first_norm, "second_norm": second_norm, "dot": dot, "cosine": cosine}
