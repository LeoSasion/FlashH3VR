"""H3CE SpatialRefinerV2: a frame-independent latent U-Net with scene context.

The network is randomly initialized until bootstrap is explicitly run. Its zero
output projection gives an identity starting point, not a trained restoration base.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from h3ce.errors import H3CEError
from .scene_context import SceneContext2D, crop_query_coordinates, latent_source_indices


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each spatial position; samples stay independent."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialFFN(nn.Module):
    """Per-position Linear projections, retaining the character-LoRA anchors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(channels, 2 * channels)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(2 * channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        y = x.permute(0, 2, 3, 1)
        return self.out_proj(self.activation(self.in_proj(y))).permute(0, 3, 1, 2)


class SpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.ffn = SpatialFFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.depthwise(self.norm(x)))


def _blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(SpatialBlock(channels) for _ in range(count)))


class SpatialRefinerV2(nn.Module):
    """Return delta-z matching [B,24,Tz,Hl,Wl], without learned cross-time mixing.

    scene_rgb is [B,3,Tsource,Hs,Ws], and must be the complete degraded X scene.
    crop_to_original is [B,3,3] or [B,Tsource,3,3], mapping bucket RGB pixel xy
    to original-image pixel xy. original_hw is [B,2] or [B,Tsource,2], in H,W
    order. Shared geometry is repeated only across each sample's selected frames.
    """

    architecture_id = "h3ce_spatial_refiner_v2"
    widths = (128, 192, 256)
    encoder_blocks = (2, 2)
    bottleneck_block_count = 4
    decoder_blocks = (2, 2)
    latent_channels = 24
    spatial_compression = 16

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(24, 128, 3, padding=1)
        self.encoders = nn.ModuleList((_blocks(128, 2), _blocks(192, 2)))
        self.downsamples = nn.ModuleList((nn.Conv2d(128, 192, 2, stride=2), nn.Conv2d(192, 256, 2, stride=2)))
        self.bottleneck = _blocks(256, 4)
        self.scene_context = SceneContext2D()
        self.query_norm = nn.LayerNorm(256)
        self.query_position = nn.Linear(2, 256)
        self.scene_attention = nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
        self.upsamples = nn.ModuleList((nn.Conv2d(256, 192, 3, padding=1), nn.Conv2d(192, 128, 3, padding=1)))
        self.skip_projections = nn.ModuleList((nn.Conv2d(384, 192, 1), nn.Conv2d(256, 128, 1)))
        self.decoders = nn.ModuleList((_blocks(192, 2), _blocks(128, 2)))
        self.output_projection = nn.Conv2d(128, 24, 3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @classmethod
    def from_config(cls, config) -> "SpatialRefinerV2":
        from h3ce.config import ModelConfig
        # Revalidate even model_construct/model_copy inputs; no silent architecture override.
        source = config.model_dump(mode="python") if hasattr(config, "model_dump") else config
        ModelConfig.model_validate(source)
        return cls()

    @staticmethod
    def _selected_geometry(
        z: Tensor, scene_rgb: Tensor, crop_to_original: Tensor, original_hw: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (z.ndim != 5 or z.shape[1] != 24 or min(z.shape) < 1 or not z.is_floating_point()):
            raise H3CEError("E_MODEL_CONTRACT", "Refiner latent must be floating [B,24,Tz,Hl,Wl]")
        if not bool(torch.isfinite(z).all()):
            raise H3CEError("E_MODEL_CONTRACT", "Refiner latent contains non-finite values")
        batch, _, latent_frames, _, _ = z.shape
        if (scene_rgb.ndim != 5 or scene_rgb.shape[:2] != (batch, 3)
                or min(scene_rgb.shape) < 1):
            raise H3CEError("E_MODEL_CONTRACT", "Refiner scene must be [B,3,Tsource,Hs,Ws]")
        if any(tensor.device != z.device for tensor in (scene_rgb, crop_to_original, original_hw)):
            raise H3CEError("E_MODEL_CONTRACT", "Latent, scene and geometry must be on the same device")
        source_frames = scene_rgb.shape[2]
        if crop_to_original.shape == (batch, 3, 3):
            crop_to_original = crop_to_original[:, None].expand(-1, source_frames, -1, -1)
        if crop_to_original.shape != (batch, source_frames, 3, 3):
            raise H3CEError("E_MODEL_CONTRACT", "crop_to_original must be [B,3,3] or [B,Tsource,3,3]")
        if original_hw.shape == (batch, 2):
            original_hw = original_hw[:, None].expand(-1, source_frames, -1)
        if original_hw.shape != (batch, source_frames, 2):
            raise H3CEError("E_MODEL_CONTRACT", "original_hw must be [B,2] or [B,Tsource,2]")
        if (not bool(torch.isfinite(crop_to_original).all()) or not bool(torch.isfinite(original_hw).all())
                or bool((original_hw <= 0).any())):
            raise H3CEError("E_MODEL_CONTRACT", "Geometry must be finite and original dimensions positive")
        affine_row = crop_to_original.new_tensor([0, 0, 1]).expand(batch, source_frames, 3)
        if not torch.allclose(crop_to_original[..., 2, :], affine_row, atol=1e-6, rtol=0):
            raise H3CEError("E_MODEL_CONTRACT", "crop_to_original must be an affine pixel-coordinate transform")
        determinant = torch.linalg.det(crop_to_original[..., :2, :2].float())
        if bool((determinant.abs() < 1e-12).any()):
            raise H3CEError("E_MODEL_CONTRACT", "crop_to_original must have a non-singular spatial transform")
        indices = latent_source_indices(source_frames, latent_frames, device=z.device)
        selected_scene = scene_rgb.index_select(2, indices).permute(0, 2, 1, 3, 4).reshape(
            batch * latent_frames, 3, *scene_rgb.shape[-2:],
        )
        selected_geometry = crop_to_original.index_select(1, indices).reshape(batch * latent_frames, 3, 3)
        selected_hw = original_hw.index_select(1, indices).reshape(batch * latent_frames, 2)
        return selected_scene, selected_geometry, selected_hw

    def forward(
        self, z: Tensor, scene_rgb: Tensor, crop_to_original: Tensor, original_hw: Tensor,
    ) -> Tensor:
        scenes, geometry, dimensions = self._selected_geometry(z, scene_rgb, crop_to_original, original_hw)
        batch, _, latent_frames, height, width = z.shape
        x = z.permute(0, 2, 1, 3, 4).reshape(batch * latent_frames, 24, height, width)
        pad_h, pad_w = (-height) % 4, (-width) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        x = self.input_projection(x)
        skips = []
        for blocks, downsample in zip(self.encoders, self.downsamples):
            x = blocks(x)
            skips.append(x)
            x = downsample(x)
        x = self.bottleneck(x)
        query_hw = x.shape[-2:]
        query = x.flatten(2).transpose(1, 2)
        coordinates = crop_query_coordinates(geometry, dimensions, query_hw)
        tokens = self.scene_context(scenes)
        positioned_query = self.query_norm(query) + self.query_position(coordinates.to(query.dtype))
        attended, _ = self.scene_attention(positioned_query, tokens, tokens, need_weights=False)
        x = (query + attended).transpose(1, 2).reshape(batch * latent_frames, 256, *query_hw)
        for upsample, projection, blocks, skip in zip(self.upsamples, self.skip_projections, self.decoders, reversed(skips)):
            x = upsample(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            x = blocks(projection(torch.cat((x, skip), dim=1)))
        delta = self.output_projection(x)[..., :height, :width]
        return delta.reshape(batch, latent_frames, 24, height, width).permute(0, 2, 1, 3, 4).contiguous()
