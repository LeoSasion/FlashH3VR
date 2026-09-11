"""Strict visual-only loading and the audited raw boundaries of AI Toolkit H3.

The vendored numerical implementation is unchanged. Its public latent
normalization and final display clamp are intentionally outside these raw
boundaries; H3VAEBridge owns normalization and training receives unclamped RGB.
"""

from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file

from h3ce.components import canonical_hash, read_component_lock, sha256_file
from h3ce.errors import H3CEError
from . import _vendor
from .acquire import SHA256 as WEIGHT_SHA256


UPSTREAM_REVISION = "7690ea62c87133410ffe1596aa222a6e0e9069f7"
UPSTREAM_SHA256 = "4ca491b1b038acf5df2d5ceb64fa4f37703cc5d01cd744eb64a104b108b1daa5"
VENDOR_SHA256 = "723fc93cf58ad3f3ffe42a770c57046967bdda4288271d0b774ff9ba3a2d2348"


class AIToolkitH3Backend:
    """A real, pretrained, fully frozen H3 VAE loaded from a verified file."""

    def __init__(self, model, *, weight_sha256):
        self.model = model
        self.weight_sha256 = weight_sha256
        self.source_sha256 = sha256_file(Path(_vendor.__file__))
        self._validate_modules()
        self.model.eval().requires_grad_(False)

    @classmethod
    def from_locked(cls, *, project_root, weights, components_lock="components.lock.json", device="cuda"):
        root = Path(project_root).resolve()
        entries = read_component_lock(root / components_lock)
        weight_entry = entries["h3_visual_vae"]
        backend_entry = entries["aitoolkit_h3_backend"]
        requested = (root / weights).resolve()
        for entry in (weight_entry, backend_entry):
            if not all(entry.get(field) for field in ("revision", "sha256", "code_revision", "source_url", "local_path")):
                raise H3CEError("E_COMPONENT_LOCK", f"Unresolved {entry['id']} lock")
        if requested != (root / weight_entry["local_path"]).resolve():
            raise H3CEError("E_COMPONENT_LOCK", "VAE path differs from component lock")
        vendor = Path(_vendor.__file__).resolve()
        if (root / backend_entry["local_path"]).resolve() != vendor or backend_entry["sha256"] != VENDOR_SHA256:
            raise H3CEError("E_COMPONENT_CODE", "Backend lock must identify the audited portable source")
        if backend_entry["revision"] != UPSTREAM_REVISION:
            raise H3CEError("E_COMPONENT_CODE", "Backend source revision is not the audited revision")
        if weight_entry["sha256"] != WEIGHT_SHA256:
            raise H3CEError("E_COMPONENT_HASH", "Unsupported VAE asset; audit the new source before loading")
        return cls.from_verified_file(requested, device=device)

    @classmethod
    def from_verified_file(cls, path, *, device="cuda"):
        """Exact pinned asset only, including acquisition/acceptance before lock update."""
        path = Path(path)
        if not path.is_file() or sha256_file(path) != WEIGHT_SHA256:
            raise H3CEError("E_COMPONENT_HASH", "Visual VAE missing or SHA256 differs from pinned asset")
        if sha256_file(Path(_vendor.__file__)) != VENDOR_SHA256:
            raise H3CEError("E_COMPONENT_CODE", "Portable upstream source has changed; re-audit required")
        state = load_file(str(path), device="cpu")
        # The exact upstream loader preserves stored tensor dtypes, removes only
        # the two known normalization entries and uses strict=True, assign=True.
        try:
            model = _vendor.MiniMaxH3VideoVAE.load_from_state_dict(state)
        except RuntimeError as error:
            raise H3CEError("E_H3_MODULE_CONTRACT", "VAE weight keys/shapes disagree with pinned source", str(error)) from error
        del state
        result = cls(model, weight_sha256=WEIGHT_SHA256)
        model.to(device=device)  # Deliberately no dtype cast or quantization.
        return result

    def _validate_modules(self):
        model = self.model
        attributes = {"latent_channels": 24, "spatial_compression": 16, "temporal_compression": 4,
                      "clip_length": 17, "token_drop": 3, "tile_size": 256, "tile_overlap_min": 64,
                      "use_tiling": True}
        for key, expected in attributes.items():
            if getattr(model, key, None) != expected:
                raise H3CEError("E_H3_MODULE_CONTRACT", f"Unexpected native H3 {key}")
        if len(model.decoder.transformer_blocks) != 36:
            raise H3CEError("E_H3_MODULE_CONTRACT", "Expected 36 native decoder transformer blocks")
        for index, block in enumerate(model.decoder.transformer_blocks):
            if type(block.attn.to_out) is not torch.nn.Linear or (block.attn.to_out.in_features, block.attn.to_out.out_features) != (2048, 2048):
                raise H3CEError("E_H3_MODULE_CONTRACT", f"Unexpected decoder.transformer_blocks.{index}.attn.to_out")
        for name in ("latents_mean", "latents_std"):
            values = getattr(model, name)
            if values.shape != (24,) or values.dtype != torch.float32 or not torch.isfinite(values).all():
                raise H3CEError("E_H3_MODULE_CONTRACT", f"Invalid {name}")
        if not (model.latents_std > 0).all():
            raise H3CEError("E_H3_MODULE_CONTRACT", "Latent standard deviations must be positive")

    def encode_mean_raw(self, external_pixels):
        """Upstream encode L772–784, before sampling/latent normalization."""
        x = (external_pixels.float() + 1.0) * 0.5
        x = ((x - self.model.pixel_mean) / self.model.pixel_std).to(self.model.dtype)
        moments = (self.model._encode_clip(x)[:, :, -1:] if x.shape[2] == 1
                   else self.model._encode_video(x))
        return moments.float().chunk(2, dim=1)[0]

    def decode_raw(self, raw_latents):
        """Upstream decode L810–828, after latent denormalization, before clamp."""
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if raw_latents.is_cuda else nullcontext()
        with ctx:
            if raw_latents.shape[2] == 1:
                decoded = self.model._decode_video(torch.cat([raw_latents, raw_latents], dim=2))[:, :, :1]
            else:
                decoded = self.model._decode_video(raw_latents)
        return decoded.float() * self.model.pixel_std + self.model.pixel_mean

    def numerical_contract(self):
        return {"upstream_revision": UPSTREAM_REVISION, "upstream_sha256": UPSTREAM_SHA256,
                "portable_source_sha256": self.source_sha256, "weights_sha256": self.weight_sha256,
                "storage_dtypes": dict(Counter(str(p.dtype) for name, p in self.model.named_parameters()
                                              if not name.endswith((".A", ".B")))),
                "normalization_source": "locked safetensors latents_mean/latents_std; native FP32 buffers",
                "latents_mean": self.model.latents_mean.detach().cpu().tolist(),
                "latents_std": self.model.latents_std.detach().cpu().tolist(),
                "tiling": {"enabled": True, "tile_size": 256, "overlap_min": 64},
                "precision": "checkpoint storage preserved; CUDA FP16 autocast; native FP32 norms/embed/proj",
                "training_output": "unclamped RGB; export clamp reproduces upstream public output"}

    def encoder_contract_id(self):
        return canonical_hash({"weights": self.weight_sha256, "source": self.source_sha256,
                               "backend_adapter_sha256": sha256_file(__file__),
                               "bridge_version": "h3ce-h3-bridge-v1", "normalization": self.numerical_contract(),
                               "posterior": "mean", "external_range": "zero_one"})
