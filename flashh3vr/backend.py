"""Pinned, frozen H3 ConvRot VAE loader and its raw RGB/latent boundaries.

The numerical path is the dequantized-FP16 path used by the adopted Dense
checkpoint. H3 weights are supplied separately by the caller. No downloads,
pickle checkpoint loading, quantization fallback, or replacement VAE occur.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn

from . import _vendor


H3_WEIGHT_SHA256 = "9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410"
H3_VENDOR_SHA256 = "723fc93cf58ad3f3ffe42a770c57046967bdda4288271d0b774ff9ba3a2d2348"
H3_PRECISION = "INT8 ConvRot asset; dequantize decoder linears once to FP16"
_QUANT_CONFIG = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def tf32_disabled():
    """Match the research execution without leaving global precision changed."""
    cuda_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_before = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = cuda_before
        torch.backends.cudnn.allow_tf32 = cudnn_before


def _regular_hadamard(device: torch.device) -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]],
                      device=device, dtype=torch.float32)
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    return h / 16


def _dequantize_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=weight.device.type, enabled=False):
        h = _regular_hadamard(weight.device)
        return ((weight.float() * scale).reshape(-1, 256) @ h.T).reshape(weight.shape)


class PinnedH3Backend:
    """The audited H3 native video implementation with frozen external weights."""

    def __init__(self, model: nn.Module):
        self.model = model.eval().requires_grad_(False)
        self.weight_sha256 = H3_WEIGHT_SHA256

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cuda") -> "PinnedH3Backend":
        path = Path(path)
        if not path.is_file() or sha256_file(path) != H3_WEIGHT_SHA256:
            raise ValueError("H3 asset missing or SHA256 differs from the pinned ConvRot VAE")
        if sha256_file(_vendor.__file__) != H3_VENDOR_SHA256:
            raise ValueError("Pinned native H3 source changed")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The verified H3 dequantized-FP16 inference path requires CUDA")

        with tf32_disabled():
            with safe_open(str(path), framework="pt") as handle:
                metadata = json.loads(handle.metadata()["minimax_h3_video_vae"])
            if metadata["vae_clip_length"] != 17 or metadata["vae_token_drop"] != 3:
                raise ValueError("H3 temporal metadata differs from the pinned asset")
            state = load_file(str(path), device="cpu")
            for name in ("latents_mean", "latents_std"):
                value = state.pop(name)
                if value.dtype != torch.float32 or not torch.equal(
                    value, torch.tensor(metadata[name], dtype=torch.float32)
                ):
                    raise ValueError(f"H3 {name} differs between tensor and metadata")

            with torch.device("meta"):
                model = _vendor.MiniMaxH3VideoVAE()
            expected_buffers = {"latents_mean", "latents_std", "pixel_mean", "pixel_std",
                                "decoder.rope.inv_freq", "decoder.mask_token"}
            actual_buffers = dict(model.named_buffers())
            if set(actual_buffers) != expected_buffers:
                raise ValueError("Unexpected native H3 buffers")
            stats = {"latents_mean": metadata["latents_mean"],
                     "latents_std": metadata["latents_std"],
                     "pixel_mean": _vendor.IMAGENET_MEAN,
                     "pixel_std": _vendor.IMAGENET_STD}
            for name, values in stats.items():
                value = torch.tensor(values, dtype=torch.float32, device=device)
                if name.startswith("pixel"):
                    value = value.view(1, 3, 1, 1, 1)
                setattr(model, name, value)
            model.decoder.rope = _vendor.RotaryEmbedding3d(48, theta=100.0).to(device)

            expected = model.state_dict()
            quant_names = [f"decoder.transformer_blocks.{i}.{suffix}" for i in range(36)
                           for suffix in ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")]
            quant_keys = {name + suffix for name in quant_names
                          for suffix in (".comfy_quant", ".weight_scale")}
            if set(state) != set(expected) | quant_keys:
                raise ValueError("H3 tensor keys differ from the pinned native model")
            for name, tensor in expected.items():
                if state[name].shape != tensor.shape:
                    raise ValueError(f"H3 tensor shape differs: {name}")
            for name in expected:
                if name.removesuffix(".weight") in quant_names and name.endswith(".weight"):
                    continue
                if state[name].dtype != torch.float32 or not torch.isfinite(state[name]).all():
                    raise ValueError(f"Invalid H3 floating tensor: {name}")
                parent, leaf = name.rsplit(".", 1)
                value = state.pop(name).to(device=device, dtype=torch.float16)
                setattr(model.get_submodule(parent), leaf,
                        value if name in actual_buffers else nn.Parameter(value, requires_grad=False))
            for name in quant_names:
                config = json.loads(state.pop(name + ".comfy_quant").numpy().tobytes())
                weight = state.pop(name + ".weight")
                scale = state.pop(name + ".weight_scale")
                if (config != _QUANT_CONFIG or weight.dtype != torch.int8
                        or scale.shape != (weight.shape[0], 1)
                        or not torch.isfinite(scale).all() or not (scale > 0).all()):
                    raise ValueError(f"Invalid H3 ConvRot layer: {name}")
                layer = model.get_submodule(name)
                layer.weight = nn.Parameter(
                    _dequantize_weight(weight.to(device), scale.to(device)).half(),
                    requires_grad=False,
                )
            if state or any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
                raise ValueError("Unconsumed or uninitialized H3 tensors")
            backend = cls(model)
            backend._validate()
            return backend

    def _validate(self) -> None:
        model = self.model
        expected = {"latent_channels": 24, "spatial_compression": 16,
                    "temporal_compression": 4, "clip_length": 17,
                    "token_drop": 3, "tile_size": 256,
                    "tile_overlap_min": 64, "use_tiling": True}
        if any(getattr(model, key, None) != value for key, value in expected.items()):
            raise ValueError("Native H3 geometry or tiling differs")
        if len(model.decoder.transformer_blocks) != 36:
            raise ValueError("H3 decoder transformer block count differs")
        for name in ("latents_mean", "latents_std"):
            values = getattr(model, name)
            if values.shape != (24,) or values.dtype != torch.float32 or not torch.isfinite(values).all():
                raise ValueError(f"Invalid H3 {name}")
        if not (model.latents_std > 0).all():
            raise ValueError("H3 latent standard deviations must be positive")

    def encode_mean_raw(self, external_pixels: torch.Tensor) -> torch.Tensor:
        x = (external_pixels.float() + 1.0) * 0.5
        x = ((x - self.model.pixel_mean) / self.model.pixel_std).to(self.model.dtype)
        moments = (self.model._encode_clip(x)[:, :, -1:] if x.shape[2] == 1
                   else self.model._encode_video(x))
        return moments.float().chunk(2, dim=1)[0]

    def decode_raw(self, raw_latents: torch.Tensor) -> torch.Tensor:
        context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if raw_latents.is_cuda else nullcontext())
        with context:
            if raw_latents.shape[2] == 1:
                decoded = self.model._decode_video(
                    torch.cat([raw_latents, raw_latents], dim=2)
                )[:, :, :1]
            else:
                decoded = self.model._decode_video(raw_latents)
        return decoded.float() * self.model.pixel_std + self.model.pixel_mean
