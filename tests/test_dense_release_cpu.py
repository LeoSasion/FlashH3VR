"""CPU contract checks for the standalone Dense inference package."""

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from flashh3vr import BUCKETS, DENSE_WEIGHT_SHA256, FrameMeta, align_half_input
from flashh3vr.backend import H3_VENDOR_SHA256, sha256_file
from flashh3vr.dense import DENSE_SHAPES, DenseInter, load_dense
from flashh3vr.native import native_tiled_dense_inter
from flashh3vr import _vendor


def test_pinned_vendor_and_weight_contract(tmp_path):
    assert sha256_file(_vendor.__file__) == H3_VENDOR_SHA256
    assert len(DENSE_WEIGHT_SHA256) == 64
    assert set(DenseInter().state_dict()) == set(DENSE_SHAPES)
    assert sum(p.numel() for p in DenseInter().parameters()) == 3_152_128
    wrong = tmp_path / "flashh3vr-dense-1837.safetensors"
    save_file({key: torch.zeros(shape, dtype=torch.float32)
               for key, shape in DENSE_SHAPES.items()}, str(wrong))
    with pytest.raises(ValueError, match="SHA256"):
        load_dense(wrong, device="cpu")


def test_dense_exact_single_residual_formula():
    model = DenseInter().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias.fill_(0.125)
    z = torch.arange(24 * 2 * 16 * 16, dtype=torch.float32).reshape(1, 24, 2, 16, 16) / 8192
    assert torch.equal(model(z), z + 0.125)


def test_real_video_pts_and_image_context():
    FrameMeta("image", (0.0,)).validate(1)
    FrameMeta("video", tuple(i / 24 for i in range(22)), True).validate(22)
    with pytest.raises(ValueError, match="22 real"):
        FrameMeta("video", tuple(i / 24 for i in range(5)), True).validate(5)
    with pytest.raises(ValueError, match="increasing"):
        FrameMeta("video", tuple([0.0] * 22), True).validate(22)


@pytest.mark.parametrize("side", BUCKETS)
def test_native_tile_geometry_and_once_per_tile(side):
    with torch.device("meta"):
        native = _vendor.MiniMaxH3VideoVAE()
    class AddConstant(torch.nn.Module):
        spec = DenseInter().spec

        def forward(self, z):
            return z + 0.25

    z = torch.zeros(1, 24, 1, side // 16, side // 16)
    restored, plan = native_tiled_dense_inter(z, AddConstant(), native)
    assert restored.shape == z.shape
    assert torch.allclose(restored, torch.full_like(z, 0.25))
    assert plan["inter_module_calls"] == {256: 1, 448: 4, 640: 9, 832: 16}[side]
    assert min(plan["x_pixels"]["overlaps"] or [64]) >= 64


def test_explicit_half_input_alignment_matches_frozen_algorithms():
    image = torch.linspace(0, 1, 3 * 128 * 128).reshape(1, 3, 1, 128, 128)
    actual = align_half_input(image, kind="image", target_side=256)
    planes = [np.asarray(Image.fromarray(np.ascontiguousarray(image[0, c, 0].numpy()))
                         .resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32)
              for c in range(3)]
    expected = np.clip(np.stack(planes), 0, 1).astype(np.float32)
    assert np.array_equal(actual[0, :, 0].numpy(), expected)
    video = torch.full((1, 3, 22, 128, 128), 0.25)
    actual_video = align_half_input(video, kind="video", target_side=256)
    assert actual_video.shape == (1, 3, 22, 256, 256)
    assert torch.equal(actual_video, torch.full_like(actual_video, 64 / 255))
    with pytest.raises(ValueError, match="Half input"):
        align_half_input(video[:, :, :21], kind="video", target_side=256)
