"""Pinned four-tensor, single-step Dense adapters; 1837 remains the default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn

from .backend import sha256_file


DENSE_WEIGHT_FILENAME = "flashh3vr-dense-1837.safetensors"
DENSE_WEIGHT_SHA256 = "a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804"
DENSE_3139_WEIGHT_FILENAME = "flashh3vr-dense-3139.safetensors"
DENSE_3139_WEIGHT_SHA256 = "1f199f95b3ae146d17bdf2ac8b7dede20262483a569189a4ce1c78a27cf58551"
DENSE_4036_WEIGHT_FILENAME = "flashh3vr-dense-4036.safetensors"
DENSE_4036_WEIGHT_SHA256 = "89d76c2361daa2d543d3628b1bc43b9ec057a485b44dd848af3c825d81bb21fe"
DENSE_ASSETS = {
    DENSE_WEIGHT_FILENAME: {"sha256": DENSE_WEIGHT_SHA256, "optimizer_step": 1837},
    DENSE_3139_WEIGHT_FILENAME: {"sha256": DENSE_3139_WEIGHT_SHA256, "optimizer_step": 3139},
    DENSE_4036_WEIGHT_FILENAME: {"sha256": DENSE_4036_WEIGHT_SHA256, "optimizer_step": 4036},
}
DENSE_PARAMETER_COUNT = 3_152_128
DENSE_SHAPES = {
    "body.1.weight": (256, 6144),
    "body.1.bias": (256,),
    "output.weight": (6144, 256),
    "output.bias": (6144,),
}


@dataclass(frozen=True)
class DenseSpec:
    kind: str = "dense"
    channels: int = 24
    latent_hw: tuple[int, int] = (16, 16)


class DenseInter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = DenseSpec()
        self.body = nn.Sequential(nn.Flatten(1), nn.Linear(6144, 256), nn.GELU())
        self.output = nn.Linear(256, 6144)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5 or z.shape[1] != 24 or z.shape[-2:] != (16, 16) or min(z.shape) < 1:
            raise ValueError("Dense Inter needs [B,24,T,16,16] normalized H3 latent")
        b, c, t, h, w = z.shape
        x = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        residual = self.output(self.body(x)).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return z + residual


def load_dense(path: str | Path, *, device: str | torch.device = "cuda") -> DenseInter:
    path = Path(path)
    identity = DENSE_ASSETS.get(path.name)
    if identity is None or not path.is_file() or sha256_file(path) != identity["sha256"]:
        raise ValueError("Dense asset missing, unknown filename, or SHA256 differs from its pinned identity")
    state = load_file(str(path), device="cpu")
    if set(state) != set(DENSE_SHAPES):
        raise ValueError("Dense asset must contain exactly the pinned four tensor keys")
    for name, shape in DENSE_SHAPES.items():
        value = state[name]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Dense asset tensor differs: {name}")
    model = DenseInter()
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != DENSE_PARAMETER_COUNT:
        raise ValueError("Dense parameter count differs")
    model.weight_filename = path.name
    model.weight_sha256 = identity["sha256"]
    model.optimizer_step = identity["optimizer_step"]
    return model.to(device=device).eval().requires_grad_(False)
