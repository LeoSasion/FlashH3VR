# Automatic full-frame video inference

Version 0.3 adds a runnable file-to-file path for the published Dense Inter 1837 weights. It reuses the historical SDR decoder, face geometry, shot detection and full-frame compositor, and replaces the old NAF restoration path with the current frozen H3 → Dense → frozen H3 model.

The caller supplies a video file, not pre-cropped faces. The command detects faces, stabilizes head crops, splits eligible segments into real temporal windows, restores them, pastes their corrections into the working full frames, and writes H.264 video plus a JSON report. No private training checkpoint, dataset registry, NAF model or local research receipt is needed.

[中文操作说明](#中文操作说明) · [Agent reproduction procedure](AGENT_REPRODUCTION.md) · [Asset identities](ASSETS.md)

## Install and prepare assets

Use a source checkout and Python 3.12. Install CUDA-enabled PyTorch and matching torchvision for your system first; the recorded environment uses torch 2.10.0 and torchvision 0.25.0. Then, from the repository root:

~~~bash
python -m pip install -e ".[full-video]"
python -m flashh3vr --help
python scripts/download_public_assets.py --asset all --models-dir models
python scripts/download_public_assets.py --asset all --models-dir models --verify-only
~~~

The download command is explicit and uses the pinned URLs, sizes and SHA256 values in [public_assets.json](../configs/public_assets.json). It downloads approximately 3.2 GB: the H3 base file, the small Dense bundle and the face detector. It extracts the complete Dense bundle with its license and notice. Matching files are reused; a mismatching existing file produces an error instead of being overwritten. `--verify-only` uses no network and changes no files. Review the model and dependency terms in [ASSETS.md](ASSETS.md).

The resulting model files are:

~~~text
models/
  minimax_h3_video_vae_int8_convrot.safetensors
  flashh3vr-dense-1837.safetensors
  yolov11m-face.pt
  LICENSE-MINIMAX-H3
  NOTICE
  ...bundle documentation and checksums...
~~~

The person detector is not required by this entry. The face detector identifies face boxes, not a person's identity. The learned adapter remains specific to the subject described in the [model card](../MODEL_CARD.md).

## Run a supported clip

Start with an SDR, silent clip whose complete decoded length is at most 90 frames. The source must have one video stream, increasing timestamps, usable color metadata, and even frame dimensions. Supply your own input; this repository does not distribute portrait examples.

~~~bash
python -m flashh3vr --kind full-video --input input.mp4 --output restored.mp4 --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/flashh3vr-dense-1837.safetensors --face-weights models/yolov11m-face.pt --target-side 448 --max-frames 90 --device cuda:0
~~~

The output and `restored.flashh3vr.json` must be new paths. Existing outputs are not overwritten. `--max-frames` is a resource bound on the entire input, not a request to silently truncate a longer movie: exceeding it is an error. Prepare a shorter clip before running, or explicitly choose a suitable larger finite limit.

`--target-side` chooses the head model canvas: 256, 448, 640 or 832. It does not set the dimensions of the whole output video. Without `--working-long-edge`, the source canvas is retained. Adding `--working-long-edge 768` explicitly downsizes the whole working/output canvas; it is useful for a first integration check but changes the output resolution. Resizing does not create native source detail.

## Per-frame behavior

- A frame is eligible when there is exactly one detected face meeting the 64-pixel minimum in the working canvas. This is a face-selection rule, not identity verification.
- Cuts, missing or ambiguous faces, and PTS gaps split temporal context. Isolated eligible frames are kept unchanged. Skipped frames and reasons remain in the report.
- A continuous eligible segment is processed in windows of at most 22 real frames, with the historical five-frame overlap. Only predictions for the same source frame are combined. The model is not repeatedly applied to its own output.
- A short final window uses the native H3 context convention: 2–5 real frames use a five-frame context; 6–22 use a 22-frame context. Repeated tail context is distinguished from real input frames, and output is trimmed to the original real-frame count. Real PTS are never fabricated for padding.
- The head correction is inverse-mapped into the full frame with the existing feathering. Pixels outside the correction region are unchanged in the pre-encode tensor. Final H.264 re-encoding is lossy, so background pixels in the encoded file are not promised to be bitwise identical.
- A video with no eligible restored segment fails explicitly. An unchanged copy is not reported as successful restoration.

The JSON report records model identities, source PTS, detected regions, geometry, window/padding decisions and skipped frames. Keep it with the output when reporting a reproduction result.

## Current boundaries

This restores faces automatically, but it is a bounded in-memory processor. Host and device memory grow with frame count and canvas size. Start with a short clip; this release does not establish full-film streaming, speed or a minimum-VRAM guarantee.

Audio is rejected explicitly; it is not silently discarded. HDR, missing/unsupported color metadata, multiple video streams, unsupported geometry and non-increasing PTS also require source preparation outside this entry. There is no multi-person identity selector or automatic recognition of the trained subject. Shot detection is a heuristic and is not a guarantee that every edit will be found.

The historical NAF, Comfy Kitchen and Windows cuBLAS integration is not needed. The current restoration core uses the pinned H3 asset through dequantized FP16 execution and native 256-pixel spatial tiles with at least 64-pixel overlap. [Verification scope](VERIFICATION.md).

## 中文操作说明

这次补齐的是**视频文件 → 自动检测人脸 → 稳定头部裁剪 → 分段分窗 → 当前 Dense 修复 → 回贴全画幅 → 导出视频与记录**。用户无需自己逐帧裁脸，也不需要我们的私有训练目录。

1. 克隆源码，使用 Python 3.12，先安装适合显卡的 CUDA PyTorch 与配套 torchvision。
2. 在仓库根目录执行上面的安装和模型下载命令。下载脚本会校验三份模型，并保留权重包的许可文件；H3 约 3.17 GB，Dense 权重约 12 MiB，人脸检测权重约 40.5 MB。
3. 先准备一段短小、无音轨、有正确色彩标记的 SDR 视频。把示例中的 input.mp4 换成自己的文件，运行完整命令。模型只在唯一合格人脸的连续片段上工作；漏检、多人或孤立帧会保留并注明原因。
4. 查看 restored.mp4 和 restored.flashh3vr.json。核对帧数、源时间戳、实际处理帧、跳过原因和回贴画面，再决定是否用于更长片段。

省略 `--working-long-edge` 会保留源画幅尺寸；显式传 768 会缩小整个输出画布。`--target-side 448` 只表示内部头部桶尺寸。`--max-frames 90` 是输入上限，超过会报错，不会只处理前90帧却称整片完成。

当前明确不包含音频回封装、多人身份识别或无限长片流式处理。短段补齐只是 H3 内部上下文，补帧不计为真实输入。完整自动路径已接到当前公开权重，旧版24 FPS或16GB标签仍不能套用。[交给其他 agent 的复现指南](AGENT_REPRODUCTION.md)。
