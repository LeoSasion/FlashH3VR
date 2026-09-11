import numpy as np
import pytest
import torch
from h3ce.data.head_bucket import (head_transform, pack_frame, pack_training_pair,
                                   inverse_delta, paste_delta, valid_bucket)


@pytest.mark.parametrize('box', [(160, 80, 208, 144), (320, 120, 608, 504),
                                  (0, 0, 127, 193), (900, 550, 1024, 720)])
def test_moving_zooming_boxes_preserve_original_coordinate_corners(box):
    g = head_transform(box, (720, 1024), frame_index=7, pts=.35)
    f, inv = np.array(g['original_to_bucket']), np.array(g['bucket_to_original'])
    x0, y0, x1, y1 = g['crop_xyxy']
    corners = np.array([[x0-.5, y0-.5, 1], [x1-.5, y1-.5, 1]]).T
    left, _, top, _ = g['pad_lrtb']; rh, rw = g['resized_hw']
    expected = np.array([[left-.5, top-.5, 1], [left+rw-.5, top+rh-.5, 1]]).T
    np.testing.assert_allclose(f @ corners, expected, atol=1e-10)
    np.testing.assert_allclose(inv @ expected, corners, atol=1e-10)
    assert g['frame_index'] == 7 and g['pts'] == .35


def test_zoom_keeps_face_occupancy_and_batch_dimensions():
    boxes = [(300-s*30, 300-s*40, 300+s*30, 300+s*40) for s in (1, 2, 3)]
    gs = [head_transform(b, (720, 1024), frame_index=i, pts=i/60) for i, b in enumerate(boxes)]
    assert len({tuple(g['crop_xyxy']) for g in gs}) == 3
    assert max(g['face_fraction_of_bucket'] for g in gs) - min(g['face_fraction_of_bucket'] for g in gs) < .001
    batch = torch.cat([pack_frame(torch.zeros(1, 3, 720, 1024), g) for g in gs])
    assert batch.shape == (3, 3, 512, 512)


def test_pair_reuses_geometry_zero_delta_is_exact_and_padding_is_ignored():
    torch.manual_seed(7)
    original = torch.rand(1, 3, 90, 150)
    g = head_transform((61, 10, 84, 56), (90, 150), frame_index=0, pts=0)
    x, y, mask = pack_training_pair(original, original.clone(), g)
    assert torch.equal(x, y) and torch.equal(x, pack_frame(original, g))
    assert int(mask.sum()) == np.prod(g['resized_hw'])
    assert torch.equal(original, paste_delta(original, torch.zeros_like(x), g))
    padding_only = (1 - mask).expand_as(x)
    assert torch.equal(original, paste_delta(original, padding_only, g))


def test_composition_changes_only_corresponding_frame_box_and_has_gradient():
    original = torch.zeros(1, 3, 120, 180)
    for i, box in enumerate([(20, 20, 40, 50), (90, 40, 150, 105)]):
        g = head_transform(box, (120, 180), frame_index=i, pts=i/60)
        delta = torch.full((1, 3, 512, 512), .2, requires_grad=True)
        out = paste_delta(original, delta, g)
        x0, y0, x1, y1 = g['crop_xyxy']
        outside = torch.ones(120, 180, dtype=torch.bool); outside[y0:y1, x0:x1] = False
        assert torch.equal(out[..., outside], original[..., outside])
        assert float(out[0, 0, (y0+y1)//2, (x0+x1)//2].detach()) == pytest.approx(.2)
        out.sum().backward()
        assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_empty_or_unrelated_geometry_is_rejected():
    with pytest.raises(ValueError):
        head_transform((20, 30, 10, 40), (90, 150), frame_index=0, pts=0)
    g = head_transform((20, 10, 40, 40), (90, 150), frame_index=0, pts=0)
    with pytest.raises(ValueError):
        pack_frame(torch.zeros(1, 3, 91, 150), g)
