"""Run one image, a real 2–22-frame head window, or a bounded video file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from . import BUCKETS, DenseRestorer, FrameMeta, align_half_input


def _read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Image input must be RGB")
        array = np.asarray(image, dtype=np.float32) / np.float32(255)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None]


def _read_video_window(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if (array.dtype != np.float32 or array.ndim != 4 or not 2 <= array.shape[0] <= 22
            or array.shape[-1] != 3 or array.shape[1] != array.shape[2]
            or not np.isfinite(array).all() or np.any((array < 0) | (array > 1))):
        raise ValueError("Video NPY must be float32 RGB [2–22,S,S,3] in [0,1]")
    return torch.from_numpy(np.ascontiguousarray(array.transpose(3, 0, 1, 2)))[None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen H3 + adopted Dense head restoration")
    parser.add_argument("--h3-weights", type=Path, required=True,
                        help="External pinned H3 INT8 ConvRot safetensors")
    parser.add_argument("--dense-weights", type=Path, required=True,
                        help="flashh3vr-dense-1837.safetensors")
    parser.add_argument("--kind", choices=("image", "video", "full-video"), required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="RGB PNG (image), float32 [2–22,S,S,3] NPY (video), or SDR video file (full-video)")
    parser.add_argument("--output", type=Path, required=True,
                        help="RGB PNG (image), float32 [2–22,S,S,3] NPY (video), or new MP4 (full-video)")
    parser.add_argument("--pts-json", type=Path,
                        help="Required for video: JSON array of 2–22 real increasing PTS seconds")
    parser.add_argument("--face-weights", type=Path,
                        help="Required for full-video: external pinned YOLO11 face .pt")
    parser.add_argument("--max-frames", type=int,
                        help="Required for full-video: finite complete-source limit >=2; excess fails")
    parser.add_argument("--working-long-edge", type=int, default=0,
                        help="Full-video only: 0 preserves source canvas (default); positive value downsizes output to that long edge")
    parser.add_argument("--half-input", action="store_true",
                        help="Explicitly align a half-side source onto the model canvas")
    parser.add_argument("--target-side", type=int, choices=BUCKETS,
                        help="Full-video head bucket; also required with --half-input, 256/448/640/832")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.kind == "full-video":
        if (args.output.suffix.lower() != ".mp4" or args.target_side is None
                or args.face_weights is None or args.max_frames is None
                or args.pts_json or args.half_input or args.overwrite):
            parser.error("full-video needs MP4 output, --face-weights, --target-side and --max-frames; no PTS JSON, half-input or overwrite")
        from .full_video import restore_full_video_file

        result = restore_full_video_file(
            args.input, args.output, h3_weights=args.h3_weights,
            dense_weights=args.dense_weights, face_weights=args.face_weights,
            side=args.target_side, max_frames=args.max_frames,
            working_long_edge=args.working_long_edge,
            device="cuda:0" if args.device == "cuda" else args.device,
        )
        print(json.dumps({"status": "restored", "kind": "full-video", **result}, ensure_ascii=False))
        return
    if args.face_weights is not None or args.max_frames is not None or args.working_long_edge != 0:
        parser.error("--face-weights, --max-frames and --working-long-edge are for full-video only")
    if args.kind == "image":
        if args.input.suffix.lower() != ".png" or args.output.suffix.lower() != ".png" or args.pts_json:
            parser.error("Image mode uses PNG input/output and no --pts-json")
        rgb = _read_image(args.input)
        rgb = rgb.unsqueeze(2)
    else:
        if args.input.suffix.lower() != ".npy" or args.output.suffix.lower() != ".npy" or not args.pts_json:
            parser.error("Video mode uses NPY input/output and requires --pts-json")
        rgb = _read_video_window(args.input)
    if args.half_input != (args.target_side is not None):
        parser.error("--half-input and --target-side must be supplied together")
    if args.half_input:
        rgb = align_half_input(rgb, kind=args.kind, target_side=args.target_side)
    pts_output = args.output.with_suffix(".pts.json") if args.kind == "video" else None
    if not args.overwrite and (args.output.exists() or (pts_output is not None and pts_output.exists())):
        parser.error("Output or video PTS sidecar exists; pass --overwrite to replace it")
    pts = None
    if args.kind == "video":
        pts = json.loads(args.pts_json.read_text(encoding="utf-8"))
        if not isinstance(pts, list):
            parser.error("PTS JSON must be an array")
        try:
            FrameMeta("video", tuple(float(value) for value in pts), True).validate(rgb.shape[2])
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    restorer = DenseRestorer(h3_weights=args.h3_weights, dense_weights=args.dense_weights,
                             device=args.device)
    if args.kind == "image":
        output = restorer.restore_image(rgb[:, :, 0].to(restorer.device), clamp_output=True)
        pixels = (output[0].permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(args.output)
    else:
        output = restorer.restore_video_window(rgb.to(restorer.device), pts=pts)
        np.save(args.output, output[0].permute(1, 2, 3, 0).cpu().numpy().astype(np.float32),
                allow_pickle=False)
        pts_output.write_text(json.dumps(pts, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"status": "restored", "kind": args.kind,
                      "output": str(args.output.resolve()),
                      "pts_output": str(pts_output.resolve()) if pts_output else None,
                      "plan": restorer.last_plan},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
