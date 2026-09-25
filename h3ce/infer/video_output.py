"""Standalone SDR H.264 writer shared by legacy and Dense video entrypoints."""

from fractions import Fraction
from pathlib import Path

import av
import torch


def encode_srgb_h264(path, frames, pts_integer, time_bases, *, rate, reuse_reformatter=False):
    """Actual source PTS survive; explicit sRGB -> BT709 before limited YUV420."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.float32:
        raise ValueError("Expected FP32 TCHW output")
    n, _, h, w = frames.shape
    if min(h, w) < 2 or h % 2 or w % 2 or len(pts_integer) != n or len(time_bases) != n:
        raise ValueError("Even output dimensions and all source timestamps required")
    times = [Fraction(t) * Fraction(*tb) for t, tb in zip(pts_integer, time_bases)]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Output PTS must increase")
    if len(set(tuple(tb) for tb in time_bases)) != 1:
        raise ValueError("Changing source time bases require explicit remux support")
    clock = Fraction(*time_bases[0])
    rate = Fraction(rate)
    if rate <= 0:
        raise ValueError("Positive source rate required")
    with av.open(str(path), "w") as dest:
        stream = dest.add_stream("libx264", rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.time_base = clock
        stream.codec_context.time_base = clock
        stream.options = {"crf": "18", "preset": "fast"}
        stream.codec_context.thread_count = 4
        for key in ("colorspace", "color_primaries", "color_trc", "color_range"):
            setattr(stream.codec_context, key, 1)
        reformatter = av.video.reformatter.VideoReformatter() if reuse_reformatter else None
        for i in range(n):
            s = frames[i].clamp(0, 1)
            linear = torch.where(s <= .04045, s / 12.92, ((s + .055) / 1.055).pow(2.4))
            bt709 = torch.where(linear < .018, 4.5 * linear, 1.099 * linear.pow(.45) - .099)
            rgb = bt709.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            raw = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame = (raw.reformat(format="yuv420p", dst_colorspace="ITU709") if reformatter is None else
                     reformatter.reformat(raw, format="yuv420p", dst_colorspace="ITU709"))
            frame.pts = int(pts_integer[i])
            frame.time_base = clock
            for packet in stream.encode(frame):
                dest.mux(packet)
        for packet in stream.encode():
            dest.mux(packet)
