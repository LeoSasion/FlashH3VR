"""CPU contract checks for the standalone Dense inference package."""

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from flashh3vr import BUCKETS, DENSE_WEIGHT_SHA256, FrameMeta, align_half_input
from flashh3vr.backend import H3_VENDOR_SHA256, sha256_file
from flashh3vr.dense import DENSE_ASSETS, DENSE_SHAPES, DenseInter, load_dense
from flashh3vr.native import native_tiled_dense_inter
from flashh3vr import _vendor
from flashh3vr.__main__ import _read_video_window


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


@pytest.mark.parametrize("filename", list(DENSE_ASSETS))
def test_pinned_model_identity_reaches_loaded_model_and_contract(tmp_path, monkeypatch, filename):
    from flashh3vr import DenseRestorer
    identity = DENSE_ASSETS[filename]
    path = tmp_path / filename
    state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in DENSE_SHAPES.items()}
    save_file(state, str(path), metadata={"step": str(identity["optimizer_step"])})
    digest = sha256_file(path)
    monkeypatch.setitem(DENSE_ASSETS, filename, {**identity, "sha256": digest})
    loaded = load_dense(path, device="cpu")
    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in state.items())
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    restorer = DenseRestorer.__new__(DenseRestorer)
    restorer.dense = loaded
    contract = restorer.contract()
    assert contract["dense_weight_sha256"] == digest
    assert contract["dense_weight_filename"] == filename
    assert contract["dense_optimizer_step"] == identity["optimizer_step"]


def test_wrong_model_bytes_cannot_be_loaded_under_other_pinned_filename(tmp_path, monkeypatch):
    import flashh3vr.dense as dense
    state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in DENSE_SHAPES.items()}
    old = tmp_path / dense.DENSE_WEIGHT_FILENAME
    save_file(state, str(old), metadata={"step": "1837"})
    monkeypatch.setitem(DENSE_ASSETS, old.name, {"sha256": sha256_file(old), "optimizer_step": 1837})
    renamed = tmp_path / dense.DENSE_3139_WEIGHT_FILENAME
    renamed.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="SHA256"):
        load_dense(renamed, device="cpu")
    unknown = tmp_path / "unregistered.safetensors"
    unknown.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="unknown filename"):
        load_dense(unknown, device="cpu")


def test_real_video_pts_and_image_context():
    FrameMeta("image", (0.0,)).validate(1)
    FrameMeta("video", tuple(i / 24 for i in range(22)), True).validate(22)
    FrameMeta("video", tuple(i / 24 for i in range(5)), True).validate(5)
    with pytest.raises(ValueError, match="2–22 real"):
        FrameMeta("video", (0.0,), True).validate(1)
    with pytest.raises(ValueError, match="2–22 real"):
        FrameMeta("video", tuple(i / 24 for i in range(23)), True).validate(23)
    with pytest.raises(ValueError, match="increasing"):
        FrameMeta("video", tuple([0.0] * 22), True).validate(22)


def test_short_head_window_npy_and_pts_length(tmp_path):
    for count in (2, 6, 22):
        path = tmp_path / f"window_{count}.npy"
        np.save(path, np.zeros((count, 256, 256, 3), dtype=np.float32), allow_pickle=False)
        assert _read_video_window(path).shape == (1, 3, count, 256, 256)
        FrameMeta("video", tuple(i / 24 for i in range(count)), True).validate(count)
        with pytest.raises(ValueError, match="matching PTS"):
            FrameMeta("video", tuple(i / 24 for i in range(count - 1)), True).validate(count)
    bad = tmp_path / "one.npy"
    np.save(bad, np.zeros((1, 256, 256, 3), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="2–22"):
        _read_video_window(bad)


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
    assert align_half_input(video[:, :, :21], kind="video", target_side=256).shape[2] == 21
    with pytest.raises(ValueError, match="Half input"):
        align_half_input(video[:, :, :1], kind="video", target_side=256)
