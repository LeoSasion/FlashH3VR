"""The adopted four-tensor, single-step Dense residual Inter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn

from .backend import sha256_file


DENSE_WEIGHT_FILENAME = "flashh3vr-dense-1837.safetensors"
DENSE_WEIGHT_SHA256 = "a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804"
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
    if path.name != DENSE_WEIGHT_FILENAME or not path.is_file() or sha256_file(path) != DENSE_WEIGHT_SHA256:
        raise ValueError("Dense asset missing, wrong filename, or SHA256 differs from adopted step 1837")
    state = load_file(str(path), device="cpu")
    if set(state) != set(DENSE_SHAPES):
        raise ValueError("Dense asset must contain exactly the adopted four tensor keys")
    for name, shape in DENSE_SHAPES.items():
        value = state[name]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Dense asset tensor differs: {name}")
    model = DenseInter()
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != DENSE_PARAMETER_COUNT:
        raise ValueError("Dense parameter count differs")
    return model.to(device=device).eval().requires_grad_(False)
