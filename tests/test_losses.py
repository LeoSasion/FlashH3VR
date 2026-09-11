"""Numerical forward-only loss tests; no backward or optimizer updates."""
import math

import pytest
import torch

from h3ce.config import ApplicationLosses
from h3ce.errors import H3CEError
from h3ce.train.losses import (ApplicationLoss, charbonnier, latent_loss,
                                lighting_target_loss, region_error, srgb_to_linear)


def _inputs():
    x = torch.full((1, 3, 1, 32, 32), .25)
    y = torch.full_like(x, .5)
    z = torch.zeros(1, 24, 1, 2, 2)
    valid = torch.ones(1, 1, 1, 32, 32)
    return x, y, z, z.clone(), valid


def test_charbonnier_uses_epsilon_squared_and_weight_sum_including_channels():
    error = torch.ones(1, 3, 1, 4, 4)
    mask = torch.ones(1, 1, 1, 4, 4)
    assert region_error(charbonnier(error), mask).item() == pytest.approx(math.sqrt(1+1e-6))
    mask[:, :, :, :2] = 0
    assert region_error(error, mask).item() == 1
    assert region_error(error, mask*0).item() == 0


def test_latent_mask_excludes_fully_artificial_cells_and_weights_partial_cells():
    z = torch.ones(1, 24, 1, 2, 2)
    target = torch.zeros_like(z)
    z[:, :, :, 0, :] = 1000
    valid = torch.ones(1, 1, 1, 32, 32)
    valid[:, :, :, :16] = 0
    assert latent_loss(z, target, valid).item() == pytest.approx(math.sqrt(1+1e-6))


def test_linear_transfer_does_not_clamp_prediction():
    result = srgb_to_linear(torch.tensor([-.5, 0., .04045, 1., 2.]))
    assert result[0] < 0 and result[-1] > 1
    assert result[2].item() == pytest.approx(.04045/12.92)
    assert result[3].item() == 1


def test_lighting_filters_compare_target_and_ignore_artificial_padding():
    x, y, _, _, valid = _inputs()
    valid[:, :, :, :8] = 0
    expected = lighting_target_loss(x, y, valid)
    x[:, :, :, :8] = 200
    y[:, :, :, :8] = -200
    assert lighting_target_loss(x, y, valid).item() == pytest.approx(expected.item())
    linear_difference = srgb_to_linear(torch.tensor(.25))-srgb_to_linear(torch.tensor(.5))
    assert expected.item() == pytest.approx(charbonnier(linear_difference).item(), rel=1e-5)


def test_spatial_lighting_has_no_cross_frame_averaging():
    x, y, _, _, valid = _inputs()
    x2 = torch.cat([x, y], dim=2)
    y2 = torch.cat([y, x], dim=2)
    assert lighting_target_loss(x2, y2, valid.expand(-1, -1, 2, -1, -1)).item() == pytest.approx(
        lighting_target_loss(x, y, valid).item(), rel=1e-5)


def test_combined_loss_has_explicit_weights_and_box_terms_are_independently_normalized():
    loss = ApplicationLoss(ApplicationLosses(perceptual=0))
    x, y, z, zy, valid = _inputs()
    person = valid.clone()
    face = torch.zeros_like(valid)
    face[:, :, :, 4:8, 4:8] = 1
    result = loss(x, y, z, zy, valid, person, face)
    assert result["rgb"].item() == pytest.approx(3*math.sqrt(.25**2+1e-6), rel=1e-6)
    assert result["perceptual"].item() == 0
    assert result["total"].item() == pytest.approx((result["rgb"]+.1*result["latent"]+.2*result["lighting_target"]).item())
    assert result["rgb_person"].item() == pytest.approx(result["rgb_face"].item())


def test_rgb_out_of_range_prediction_is_penalized_without_clamping():
    x, y, z, zy, valid = _inputs()
    loss = ApplicationLoss(ApplicationLosses(perceptual=0, lighting_target=0))
    y.fill_(1)
    x.fill_(2)
    assert loss(x, y, z, zy, valid)["rgb"].item() > .99


def test_missing_perceptual_model_requires_explicit_zero_resolved_weight():
    with pytest.raises(H3CEError, match="explicitly to 0"):
        ApplicationLoss(ApplicationLosses())


@pytest.mark.parametrize("invalid", ["empty", "negative", "nonfinite", "shape"])
def test_invalid_masks_fail_explicitly(invalid):
    x, y, z, zy, valid = _inputs()
    if invalid == "empty": valid.zero_()
    elif invalid == "negative": valid.fill_(-1)
    elif invalid == "nonfinite": valid.fill_(float("nan"))
    else: valid = valid[..., :16, :]
    with pytest.raises(H3CEError):
        ApplicationLoss(ApplicationLosses(perceptual=0))(x, y, z, zy, valid)


def test_nonfinite_prediction_is_rejected_without_optimizer():
    x, y, z, zy, valid = _inputs()
    x.fill_(float("inf"))
    with pytest.raises(H3CEError, match="Nonfinite"):
        ApplicationLoss(ApplicationLosses(perceptual=0))(x, y, z, zy, valid)
