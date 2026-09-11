"""One RGB/normalized-latent API, preserving native H3 video computation."""

from dataclasses import asdict, dataclass
import math

import torch

from h3ce.components import canonical_hash, sha256_file
from h3ce.errors import H3CEError
from .decoder_adapter import adapter_fingerprint


@dataclass(frozen=True)
class FrameMeta:
    kind: str
    pts: tuple[float, ...]
    real_video: bool = False
    shot_id: str | None = None

    def validate(self, frame_count):
        if self.kind not in {"image", "video"} or len(self.pts) != frame_count or frame_count < 1:
            raise H3CEError("E_H3_FRAME_CONTRACT", "Media kind, frame count and PTS disagree")
        if self.kind == "image" and (frame_count != 1 or self.real_video):
            raise H3CEError("E_H3_FRAME_CONTRACT", "Images have exactly one frame and cannot be real video")
        if self.kind == "video" and frame_count < 2:
            raise H3CEError("E_H3_FRAME_CONTRACT", "Video requires at least two source frames")
        if not all(math.isfinite(p) for p in self.pts) or any(b <= a for a, b in zip(self.pts, self.pts[1:])):
            raise H3CEError("E_H3_FRAME_CONTRACT", "PTS must be finite and strictly increasing")


@dataclass(frozen=True)
class CodecContract:
    encoder_contract_id: str
    channels: int
    spatial_compression: int
    frame_mappings: dict
    native: dict


@dataclass(frozen=True)
class CodecPack:
    effective_decoder_hash: str
    trainable: bool


@dataclass(frozen=True)
class LatentBatch:
    tensor: torch.Tensor
    meta: FrameMeta
    valid_frames: int
    padded_frames: int
    original_hw: tuple[int, int]
    encoder_contract_id: str

    def with_tensor(self, tensor):
        return LatentBatch(tensor, self.meta, self.valid_frames, self.padded_frames,
                           self.original_hw, self.encoder_contract_id)

    def padded_valid_mask(self, *, device=None):
        """The loss mask excludes repeated padding; PTS never includes invented times."""
        return torch.arange(self.padded_frames, device=device) < self.valid_frames


def legal_pixel_frames(frames, *, kind):
    if frames < 1:
        raise H3CEError("E_H3_FRAME_CONTRACT", "Frame count must be positive")
    if kind == "image":
        if frames != 1:
            raise H3CEError("E_H3_FRAME_CONTRACT", "Images have one frame")
        return 1
    if kind != "video":
        raise H3CEError("E_H3_FRAME_CONTRACT", "Unknown media kind")
    return 5 if frames <= 5 else 5 + 17 * math.ceil((frames - 5) / 17)


class H3VAEBridge:
    def __init__(self, backend):
        self.backend = backend
        self.encoder_id = canonical_hash({"backend": backend.encoder_contract_id(), "bridge_source": sha256_file(__file__)})

    def inspect_contract(self):
        return CodecContract(self.encoder_id, 24, 16, {1: 1, 5: 2, 22: 7, 39: 12},
                             self.backend.numerical_contract())

    def current_codec_pack(self):
        trainable = any(p.requires_grad for p in self.backend.model.decoder.parameters())
        key = canonical_hash({"base": self.backend.weight_sha256, "code": self.backend.source_sha256,
                              "bridge_source": sha256_file(__file__),
                              "backend_adapter_source": sha256_file(__import__(self.backend.__module__, fromlist=["__file__"]).__file__),
                              "adapter_source": sha256_file(__import__(adapter_fingerprint.__module__, fromlist=["__file__"]).__file__),
                              "precision": self.backend.numerical_contract()["precision"],
                              "tiling": self.backend.numerical_contract()["tiling"],
                              "adapter": adapter_fingerprint(self.backend.model), "adapter_scale": 1.0})
        return CodecPack(key, trainable)

    def encode_rgb(self, x, meta):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[0] < 1:
            raise H3CEError("E_H3_FRAME_CONTRACT", "Expected RGB Tensor[B,3,T,H,W]")
        meta.validate(x.shape[2])
        if not x.is_floating_point() or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise H3CEError("E_COLOR_CONTRACT", "RGB inputs must be finite floats in [0,1]")
        if any(side < 32 or side % 32 for side in x.shape[-2:]):
            raise H3CEError("E_H3_FRAME_CONTRACT", "Pixel canvases must be >=32 and aligned to 32")
        model = self.backend.model
        if x.device != model.device:
            raise H3CEError("E_H3_FRAME_CONTRACT", "RGB and VAE must be on the same device")
        if any(p.requires_grad for part in (model.encoder, model.quant_conv, model.post_quant_conv) for p in part.parameters()):
            raise H3CEError("E_H3_MODULE_CONTRACT", "Encoder and quant/post-quant convolutions must remain frozen")
        valid = x.shape[2]
        padded = legal_pixel_frames(valid, kind=meta.kind)
        if padded != valid:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, padded-valid, -1, -1)], dim=2)
        with torch.no_grad():
            raw = self.backend.encode_mean_raw(x.float() * 2.0 - 1.0)
            z = (raw - model.latents_mean.view(1, 24, 1, 1, 1)) / model.latents_std.view(1, 24, 1, 1, 1)
        expected_t = 1 if padded == 1 else 5 * ((padded - 5) // 17) + 2
        expected_shape = (x.shape[0], 24, expected_t, x.shape[-2]//16, x.shape[-1]//16)
        if tuple(z.shape) != expected_shape or not torch.isfinite(z).all():
            raise H3CEError("E_H3_FRAME_CONTRACT", "Native encode returned invalid shape or nonfinite values", {"shape": list(z.shape)})
        return LatentBatch(z, meta, valid, padded, tuple(x.shape[-2:]), self.encoder_id)

    def decode_latent(self, z, *, grad, codec_pack):
        if z.encoder_contract_id != self.encoder_id:
            raise H3CEError("E_H3_FRAME_CONTRACT", "Latent encoder contract differs")
        z.meta.validate(z.valid_frames)
        if z.padded_frames != legal_pixel_frames(z.valid_frames, kind=z.meta.kind):
            raise H3CEError("E_H3_FRAME_CONTRACT", "Latent padding metadata differs from frame contract")
        expected_t = 1 if z.padded_frames == 1 else 5*((z.padded_frames-5)//17)+2
        if z.tensor.ndim != 5 or tuple(z.tensor.shape[1:]) != (24, expected_t, z.original_hw[0]//16, z.original_hw[1]//16):
            raise H3CEError("E_H3_FRAME_CONTRACT", "Latent shape differs from metadata")
        current = self.current_codec_pack()
        if codec_pack != current:
            raise H3CEError("E_CODEC_COMPATIBILITY", "Codec pack changed; refresh effective decoder hash")
        if current.trainable and not grad:
            raise H3CEError("E_H3_GRADIENT_CONTRACT", "Trainable decoder output must retain gradients")
        if not torch.isfinite(z.tensor).all():
            raise H3CEError("E_H3_FRAME_CONTRACT", "Latents contain nonfinite values")
        model = self.backend.model
        with torch.set_grad_enabled(grad):
            raw = z.tensor.float() * model.latents_std.view(1, 24, 1, 1, 1) + model.latents_mean.view(1, 24, 1, 1, 1)
            rgb = self.backend.decode_raw(raw)
        expected = (z.tensor.shape[0], 3, z.padded_frames, *z.original_hw)
        if tuple(rgb.shape) != expected or not torch.isfinite(rgb).all():
            raise H3CEError("E_H3_FRAME_CONTRACT", "Native decode returned invalid shape or nonfinite values", {"shape": list(rgb.shape)})
        return rgb[:, :, :z.valid_frames]
