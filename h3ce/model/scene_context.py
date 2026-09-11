"""Per-frame degraded-scene tokens and crop coordinates, with no time mixing."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from h3ce.errors import H3CEError


def latent_source_indices(source_frames: int, latent_frames: int, *, device=None) -> Tensor:
    """Select round-half-up uniform source indices; one latent uses frame zero.

    This locates scene context only, and is not a claim about the H3 receptive field.
    Integer arithmetic avoids Python's round-to-even behavior at exact half ties.
    """
    if (type(source_frames) is not int or type(latent_frames) is not int
            or source_frames < 1 or latent_frames < 1):
        raise H3CEError("E_MODEL_CONTRACT", "Source and latent frame counts must be positive integers")
    if latent_frames == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    positions = torch.arange(latent_frames, dtype=torch.long, device=device)
    denominator = latent_frames - 1
    return (2 * positions * (source_frames - 1) + denominator) // (2 * denominator)


def scene_token_coordinates(*, device, dtype) -> Tensor:
    """The 8x8 scene grid in normalized original-frame coordinates [0, 1]."""
    axis = (torch.arange(8, device=device, dtype=dtype) + 0.5) / 8
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(64, 2)


def crop_query_coordinates(
    crop_to_original: Tensor, original_hw: Tensor, query_hw: tuple[int, int],
) -> Tensor:
    """Map bottleneck centers through the bucket-to-original affine transform.

    Each bottleneck position covers 4 latent pixels, or 64 input RGB pixels.
    Latent padding remains in the same coordinate system and is cropped at output.
    Values outside [0, 1] deliberately retain padding/context coordinates.
    """
    height, width = query_hw
    ys = (torch.arange(height, device=crop_to_original.device, dtype=torch.float32) + 0.5) * 64
    xs = (torch.arange(width, device=crop_to_original.device, dtype=torch.float32) + 0.5) * 64
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    homogeneous = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(-1, 3)
    # Coordinates use float32 even when the refiner is under BF16 autocast.
    with torch.autocast(device_type=crop_to_original.device.type, enabled=False):
        mapped = homogeneous.unsqueeze(0) @ crop_to_original.float().transpose(-1, -2)
        scale_xy = original_hw.float().flip(-1).unsqueeze(1)
        return mapped[..., :2] / scale_xy


class SceneContext2D(nn.Module):
    """Whole degraded RGB scene -> 64 tokens of width 256, independently per frame.

    Input has shape [N, 3, H, W]. N consists of independent selected (B, Tz)
    entries; the module never merges their tokens. Aspect ratio is preserved
    while reducing the longest edge to at most 256 before the spatial CNN.
    """

    longest_edge = 256
    grid_hw = (8, 8)
    width = 256

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(256)
        self.position = nn.Linear(2, 256)

    def forward(self, scene_rgb: Tensor) -> Tensor:
        if (scene_rgb.ndim != 4 or scene_rgb.shape[1] != 3
                or min(scene_rgb.shape) < 1 or not scene_rgb.is_floating_point()):
            raise H3CEError("E_MODEL_CONTRACT", "SceneContext2D expects floating [N,3,H,W] degraded RGB")
        if not bool(torch.isfinite(scene_rgb).all()):
            raise H3CEError("E_MODEL_CONTRACT", "Scene RGB contains non-finite values")
        if bool((scene_rgb < 0).any()) or bool((scene_rgb > 1).any()):
            raise H3CEError("E_MODEL_CONTRACT", "Scene RGB must use the [0,1] input contract")
        height, width = scene_rgb.shape[-2:]
        if max(height, width) > self.longest_edge:
            factor = self.longest_edge / max(height, width)
            resized = (max(1, int(height * factor + 0.5)), max(1, int(width * factor + 0.5)))
            scene_rgb = F.interpolate(scene_rgb, size=resized, mode="bilinear", align_corners=False, antialias=True)
        features = F.adaptive_avg_pool2d(self.cnn(scene_rgb), self.grid_hw)
        tokens = self.token_norm(features.flatten(2).transpose(1, 2))
        coordinates = scene_token_coordinates(device=tokens.device, dtype=tokens.dtype)
        return tokens + self.position(coordinates).unsqueeze(0)
