"""Single-step frozen H3 -> Dense -> frozen H3 head-window inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch

from .backend import H3_PRECISION, H3_WEIGHT_SHA256, PinnedH3Backend, tf32_disabled
from .dense import DENSE_WEIGHT_SHA256, DenseInter, load_dense
from .native import native_tiled_dense_inter


BUCKETS = (256, 448, 640, 832)
REAL_VIDEO_FRAMES = 22
MIN_VIDEO_FRAMES = 2


@dataclass(frozen=True)
class FrameMeta:
    kind: str
    pts: tuple[float, ...]
    real_video: bool = False

    def validate(self, frames: int) -> None:
        if self.kind == "image":
            if frames != 1 or self.real_video or len(self.pts) != 1:
                raise ValueError("Image inference needs exactly one real frame")
        elif self.kind == "video":
            if not MIN_VIDEO_FRAMES <= frames <= REAL_VIDEO_FRAMES or not self.real_video or len(self.pts) != frames:
                raise ValueError("Video windows need 2–22 real frames and matching PTS values")
        else:
            raise ValueError("Media kind must be image or video")
        if not all(math.isfinite(p) for p in self.pts) or any(
            b <= a for a, b in zip(self.pts, self.pts[1:])
        ):
            raise ValueError("PTS must be finite and strictly increasing")


def align_half_input(low: torch.Tensor, *, kind: str, target_side: int) -> torch.Tensor:
    """Explicit interface alignment of half-size input; never an HQ target.

    Images use Pillow float BICUBIC per channel. Real video windows use the
    frozen RGB8 LANCZOS alignment. Both reproduce the training input canvas.
    """
    if (target_side not in BUCKETS or low.ndim != 5 or low.shape[0] != 1
            or low.shape[1] != 3 or low.shape[-2:] != (target_side // 2, target_side // 2)
            or (kind == "image" and low.shape[2] != 1)
            or (kind == "video" and not MIN_VIDEO_FRAMES <= low.shape[2] <= REAL_VIDEO_FRAMES)
            or kind not in ("image", "video") or not low.is_floating_point()
            or not bool(torch.isfinite(low).all()) or bool((low < 0).any()) or bool((low > 1).any())):
        raise ValueError("Half input must be finite RGB [1,3,T,S/2,S/2] in [0,1]")
    source_device = low.device
    frames = low.detach().to(device="cpu", dtype=torch.float32).numpy()[0].transpose(1, 0, 2, 3)
    aligned = []
    for frame in frames:
        if kind == "image":
            planes = [np.asarray(Image.fromarray(np.ascontiguousarray(frame[channel]))
                                 .resize((target_side, target_side), Image.Resampling.BICUBIC),
                                 dtype=np.float32) for channel in range(3)]
            canvas = np.clip(np.stack(planes, axis=0), 0, 1).astype(np.float32)
        else:
            rgb8 = np.uint8(np.clip(frame.transpose(1, 2, 0), 0, 1) * 255 + .5)
            resized = np.asarray(Image.fromarray(rgb8, mode="RGB")
                                 .resize((target_side, target_side), Image.Resampling.LANCZOS),
                                 dtype=np.float32)
            canvas = (resized.transpose(2, 0, 1) / np.float32(255)).astype(np.float32)
        aligned.append(canvas)
    result = torch.from_numpy(np.stack(aligned, axis=1)[None])
    return result.to(device=source_device)


class DenseRestorer:
    """Head-crop tensor inference with the adopted four-tensor Dense baseline.

    Input and output are RGB float [B,3,T,H,W]. B is currently 1, T is 1 for
    images or 2–22 for a real, continuous video window. Accepted canvases are
    square 256/448/640/832. No detection, identity recognition, source crop,
    shot segmentation, face pasting, or long-video stitching is performed.
    """

    def __init__(self, *, h3_weights: str | Path, dense_weights: str | Path,
                 device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The verified release numerical path requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.h3 = PinnedH3Backend.load(h3_weights, device=self.device)
        self.dense: DenseInter = load_dense(dense_weights, device=self.device)
        self.last_plan: dict | None = None

    def restore_tensor(self, rgb: torch.Tensor, *, meta: FrameMeta,
                       clamp_output: bool = False) -> torch.Tensor:
        if (rgb.ndim != 5 or rgb.shape[0] != 1 or rgb.shape[1] != 3
                or rgb.shape[-2] != rgb.shape[-1] or rgb.shape[-1] not in BUCKETS
                or rgb.device != self.device or not rgb.is_floating_point()
                or not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any())
                or bool((rgb > 1).any())):
            raise ValueError("Expected finite RGB [1,3,T,S,S] on the model CUDA device in [0,1]")
        meta.validate(rgb.shape[2])
        model = self.h3.model
        if model.training or self.dense.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("H3 and Dense must remain frozen in evaluation mode")
        valid_frames = rgb.shape[2]
        padded_frames = (1 if meta.kind == "image" else
                         5 if valid_frames <= 5 else REAL_VIDEO_FRAMES)
        expected_latent_frames = 1 if padded_frames == 1 else 2 if padded_frames == 5 else 7
        with tf32_disabled(), torch.no_grad():
            model_input = (torch.cat([rgb, rgb[:, :, -1:].expand(-1, -1, padded_frames - valid_frames,
                                                               -1, -1)], dim=2)
                           if padded_frames != valid_frames else rgb)
            raw = self.h3.encode_mean_raw(model_input.float() * 2.0 - 1.0)
            normalized = (raw - model.latents_mean.view(1, 24, 1, 1, 1)) / model.latents_std.view(1, 24, 1, 1, 1)
            expected = (1, 24, expected_latent_frames, rgb.shape[-2] // 16, rgb.shape[-1] // 16)
            if normalized.shape != expected or not bool(torch.isfinite(normalized).all()):
                raise ValueError("Native H3 encoder returned invalid normalized latent")
            corrected, plan = native_tiled_dense_inter(normalized, self.dense, model)
            raw_corrected = (corrected.float() * model.latents_std.view(1, 24, 1, 1, 1)
                             + model.latents_mean.view(1, 24, 1, 1, 1))
            output = self.h3.decode_raw(raw_corrected)
            if output.shape != model_input.shape or not bool(torch.isfinite(output).all()):
                raise ValueError("Native H3 decoder returned invalid RGB output")
            self.last_plan = {**plan, "valid_frames": valid_frames,
                              "h3_context_frames": padded_frames,
                              "padded_frames": padded_frames,
                              "padding_frames_added": padded_frames - valid_frames,
                              "padded_frames_meaning": "total H3 context frames, including real frames"}
            output = output[:, :, :valid_frames]
            return output.clamp(0, 1) if clamp_output else output

    def restore_image(self, rgb: torch.Tensor, *, clamp_output: bool = False) -> torch.Tensor:
        if rgb.ndim != 4:
            raise ValueError("Image convenience API needs [1,3,S,S]")
        output = self.restore_tensor(rgb.unsqueeze(2), meta=FrameMeta("image", (0.0,)),
                                     clamp_output=clamp_output)
        return output[:, :, 0]

    def restore_video_window(self, rgb: torch.Tensor, *, pts: Sequence[float],
                             clamp_output: bool = False) -> torch.Tensor:
        return self.restore_tensor(rgb, meta=FrameMeta("video", tuple(float(p) for p in pts), True),
                                   clamp_output=clamp_output)

    def contract(self) -> dict:
        return {"h3_weight_sha256": H3_WEIGHT_SHA256,
                "dense_weight_sha256": DENSE_WEIGHT_SHA256,
                "precision": H3_PRECISION,
                "native_tile_pixels": 256,
                "minimum_overlap_pixels": 64,
                "buckets": list(BUCKETS),
                "video_window_real_frames": REAL_VIDEO_FRAMES,
                "video_window_real_frames_min": MIN_VIDEO_FRAMES,
                "video_tail_padding": "repeat_last_only_inside_H3_to_5_or_22; trim_to_real_frames",
                "repair_steps": 1,
                "output_is_unclamped_by_default": True}
