# FlashH3VR Dense Inter 4036 — optional photo-fit research candidate

The adopted research baseline and default download remain **1837**. Model 3139
remains a separate historical option. This release adds Adam4036 as an explicit
option for the subject-specific Go Youn-jung (고윤정) head restoration path. It is
not a blanket best model or a general portrait restorer. FlashH3VR v0.3.2 or
newer recognizes its pinned filename and checksum.

## Identity and lineage

- Four FP32 tensors; 3,152,128 parameters; 24-channel 16 × 16 latent tiles and
  a 256-unit Dense bottleneck.
- Full Adam3084 plus 952 updates / 6,430.285 effective training seconds,
  ending at Adam4036. The full optimizer checkpoint is not distributed.
- Source checkpoint SHA256: `fe0fe3071f2947602fc5fb508e4898588ccf39bc0fcab66f90100c31459ac79b`.
- Inference safetensors SHA256: `89d76c2361daa2d543d3628b1bc43b9ec057a485b44dd848af3c825d81bb21fe`.
- External H3 checkpoint SHA256:
  `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410`.
  The tested inference path uses the pinned dequantized FP16 H3 mode.
- Training buckets: 256 / 448 / 640 / 832. Native H3 tiles remain 256 pixels.

## Data, measured changes, and limits

The training pool contained 84 photographs and 155 real video windows. This
iteration first admitted nine photos from a newer batch into training; one photo
was held out for development observation, and no new videos passed quality
admission. The 832 bucket included only one new photo and three older video
windows from two video sources, with 373.361 effective training seconds. Derived
sizes and repeated windows do not create independent capture families. HQ targets
were native master crops kept or downsampled, without upscale or sharpening.

On the same fixed conditions, relative to Adam3084, the nine trained photos'
RGB / edge / mouth-edge errors fell 16.899% /
4.569% / 27.320%. **Retained-material tradeoff:** the older 626 training
conditions' RGB error rose 0.485% while edge error fell 0.348%; the reused
12 development conditions' RGB / edge errors rose 2.012% / 0.389%. The single
new held-out development photo's four conditions averaged an 11.777% RGB error
reduction, but each had at least one scored regression. This does not certify
independent-family generalization. Percentages first give each canonical source
equal weight within each populated bucket / photo-or-video / half-or-clean
stratum, then give the populated strata equal weight. The older 448-photo
mouth-edge metric still regresses 2.236%–2.586% against candidate Adam3258.
Among all 684 conditions, 522 had at least
one scored metric regression relative to 3084. Lower numerical error does not
mean an equal gain in perceived detail.

The [public evaluation table](../configs/dense4036.evaluation.json) gives the
aggregate results for each measured pool and comparator, including regressions.

All 3,420 raw model-condition arrays and 40,240 true-frame references were
independently recomputed. A separate actual visual review covered 76 fixed
frames through 176 native-size views plus five curves. Those viewed frames
showed no obvious new severe grid, ghosting, paste boundary, or facial-structure
damage, but most normal-size differences remained small and skin, eyelashes,
fine hair, and lips still looked soft. The review is limited to those frames.
There is no certification of other identities, complete videos, ordinary
playback, long-film streaming, FPS, or 16GB GPU operation from this research
evaluation. Public-entry software checks are reported separately in the release.

## Use and terms

Download or verify this optional bundle with `--dense-model 4036`, extracting
it into its own `models/dense-4036` directory. Keep 1837 available as the
public default and 3139 as the earlier public option. Adam3084 is retained
privately as a research comparison/fallback and has no public downloader option.
The full-video command explicitly
passes `--dense-weights models/dense-4036/flashh3vr-dense-4036.safetensors`.
Only use supported short silent SDR clips within a finite frame limit.

Project source uses AGPL-3.0-only. This H3-dependent adapter retains the
MiniMax H3 Community License and downstream conditions; keep
LICENSE-MINIMAX-H3 and NOTICE with the weight. The bundle contains no H3 base,
face-detector weights, private media, cached outputs, optimizer/scaler, random
state, or full training checkpoint. Do not infer permission to redistribute
training media from its online availability. This project implies no
endorsement by MiniMax, the subject, or publishers.

## Explicit v0.3.2 commands

From a v0.3.2 checkout with its pinned dependencies:

~~~bash
python scripts/download_public_assets.py --asset all --dense-model 4036 --models-dir models
python scripts/download_public_assets.py --asset all --dense-model 4036 --models-dir models --verify-only
python -m flashh3vr --kind full-video --input input.mp4 --output restored-4036.mp4 --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/dense-4036/flashh3vr-dense-4036.safetensors --face-weights models/yolov11m-face.pt --target-side 448 --max-frames 90 --device cuda:0
~~~

The selected filename, SHA256, and Adam step appear in the full-video result.
To use the adopted default, choose `--asset dense` without a model option and
pass `--dense-weights models/flashh3vr-dense-1837.safetensors` with a separate
output path. The 3139 bundle remains an independent historical option.

中文：4036须显式使用 `--dense-model 4036` 下载到独立目录；推理时只替换
Dense权重路径。新照片拟合误差降低，但旧素材和复用验证有回退，不能称所有
素材画质更好。1837默认与3139历史可选保留，实际普通播放和速度认证范围
见本版本的验证记录。
