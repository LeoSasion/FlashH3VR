"""CPU source/bridge contract checks; these do not constitute real H3 acceptance."""

import ast
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_portable_vendor_preserves_every_numerical_ast_node():
    original = (ROOT / "h3ce/vae/_upstream/vae.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == "4ca491b1b038acf5df2d5ceb64fa4f37703cc5d01cd744eb64a104b108b1daa5"
    source = ast.parse(original.decode())
    source.body = [node for node in source.body
                   if not (isinstance(node, ast.ImportFrom) and node.module == "toolkit.models.v2._mixin")]
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE":
            assert len(node.bases) == 2
            assert ast.unparse(node.bases[1]) == "OstrisModelMixin"
            node.bases = node.bases[:1]
    actual = ast.parse((ROOT / "h3ce/vae/_vendor.py").read_text(encoding="utf-8"))
    assert ast.dump(source, include_attributes=False) == ast.dump(actual, include_attributes=False)


def test_pinned_weight_header_exact_projection_targets():
    import json
    header = json.loads((ROOT / "h3ce/vae/_upstream/weight_header.json").read_text(encoding="utf-8"))
    for index in range(36):
        assert header[f"decoder.transformer_blocks.{index}.attn.to_out.weight"]["shape"] == [2048, 2048]
    assert header["latents_mean"]["shape"] == header["latents_std"]["shape"] == [24]


@pytest.mark.parametrize("frames,padded", [(2,5), (5,5), (6,22), (22,22), (23,39), (39,39), (40,56)])
def test_reference_cpu_tail_policy(frames, padded):
    from h3ce.vae.bridge import legal_pixel_frames
    assert legal_pixel_frames(frames, kind="video") == padded


@pytest.mark.parametrize("frames,latents", [(1,1), (5,2), (22,7), (39,12)])
def test_native_source_temporal_shape_functions(frames, latents):
    from h3ce.vae._vendor import MiniMaxH3VideoVAE
    assert MiniMaxH3VideoVAE.latent_frames(frames) == latents
    assert MiniMaxH3VideoVAE.pixel_frames(latents) == frames


@pytest.mark.parametrize("pts", [(0.,0.), (1.,0.), (0.,float('nan')), (0.,float('inf'))])
def test_reference_cpu_bad_pts_rejected(pts):
    from h3ce.errors import H3CEError
    from h3ce.vae.bridge import FrameMeta
    with pytest.raises(H3CEError, match="PTS"):
        FrameMeta("video", pts, real_video=True).validate(2)


def test_reference_cpu_image_cannot_claim_real_video():
    from h3ce.errors import H3CEError
    from h3ce.vae.bridge import FrameMeta
    with pytest.raises(H3CEError):
        FrameMeta("image", (0.,), real_video=True).validate(1)


def test_reference_cpu_lora_zero_init_and_frozen_base_gradient():
    import torch
    from h3ce.vae.decoder_adapter import DecoderOutputLoRA
    base = torch.nn.Linear(3, 5)
    layer = DecoderOutputLoRA(base, rank=2, alpha=2)
    x = torch.tensor([[.4, -.2, .3]], requires_grad=True)
    assert torch.equal(layer(x), base(x))
    layer(x).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.B.grad is not None and layer.B.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())


def test_reference_cpu_repeat_padding_mask_has_no_invented_pts():
    import torch
    from h3ce.vae.bridge import FrameMeta, LatentBatch
    pts = (.02, .06, .11)
    batch = LatentBatch(torch.empty(1,24,2,2,2), FrameMeta("video", pts, True), 3, 5, (32,32), "reference_only")
    assert batch.meta.pts == pts
    assert batch.padded_valid_mask().tolist() == [True, True, True, False, False]
