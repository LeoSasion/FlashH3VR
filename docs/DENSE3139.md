# FlashH3VR Dense Inter 3139 — optional experimental weights

The adopted research baseline and default downloader remain **1837**. This is an
optional single-person head-restoration candidate for Go Youn-jung (고윤정),
compatible with the same frozen H3 encoder → one Dense correction → frozen H3
decoder. Use FlashH3VR **v0.3.1 or newer** to load its pinned identity.

## Identity and lineage

- Four FP32 tensors; 3,152,128 parameters; 24-channel 16 × 16 latent tiles and a 256-unit bottleneck.
- Adam1837 plus 1,302 updates / 9,000.927 effective seconds, ending at Adam3139.
- Source checkpoint SHA256: `04cca1fe931e7aa24961d6c9f64ea66cc86186463e946038db7c42e94d2fd72b`. The full checkpoint is not distributed.
- Safetensors SHA256: `1f199f95b3ae146d17bdf2ac8b7dede20262483a569189a4ce1c78a27cf58551`.
- External H3 SHA256: `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410`, using the pinned dequantized FP16 path.
- Training buckets 256 / 448 / 640 / 832; native H3 256-pixel tiles, minimum overlap 64 pixels.

## Data and evaluation

The final pool contains 75 photographs and 155 real video windows, with 3,307
distinct source-video frames. The increment is nine admitted windows / 100 real
frames: four at 256 and five at 448; no new 640/832 data. All 626 training
conditions received updates. Repeated windows and derived sizes do not add
independent sources. Targets are clear native master crops kept or downsampled;
no upscaled, sharpened or generated HQ targets were admitted.

The comparison uses the same 638 conditions for 1837, 3258 and 3139. Percentages
below are error reduction, first averaged within canonical sources and then
equally across populated strata; positive means lower error. These are training
and repeatedly used development observations, not independent generalization.

| Scope | RGB vs1837 | Edge vs1837 | RGB vs3258 | Edge vs3258 |
|---|---:|---:|---:|---:|
| Old training pool | +4.373% | +2.333% | +0.267% | −0.023% |
| Nine new training windows | +16.166% | +6.543% | +16.258% | +6.532% |
| Reused development windows | +1.041% | +0.304% | +0.899% | +0.243% |
| Old 448 strata | +1.773% | +0.691% | −2.015% | −0.916% |
| Old 832 strata | +1.420% | +0.377% | +3.001% | +0.880% |

448 photograph mouth-edge error improves 1.930%–2.114% against 1837 but regresses
1.653%–2.137% against 3258. Across all 638 conditions, at least one scored metric
regresses in 47 conditions against 1837 and 544 against 3258; repeated same-source
windows strongly affect these counts. No blanket per-condition superiority is
claimed. Video mouth masks were empty and excluded, not scored as zero error.

Independent CPU recomputation checked 1,914 raw arrays / 24,006 true-frame
references and 132 frozen groups. Actual review of 102 fixed normal-size frames
found no obvious new severe grid, ghosting, color fringe or facial structural
damage in that set. Differences remained small and fine detail remained soft.
The 832 bucket still has only three windows / two sources and 282.100 effective
training seconds. No FPS, 16GB, whole-film streaming or new independent
generalization certification applies. Public-entry software/playback checks are
reported separately in [verification](https://github.com/LeoSasion/FlashH3VR/blob/v0.3.1/docs/VERIFICATION.md).

## Use and terms

Prepare the optional bundle separately so its model card/checksums do not replace
1837's. Follow [usage and download instructions](https://github.com/LeoSasion/FlashH3VR/blob/v0.3.1/docs/DENSE3139.md).

### Run the optional model

From a v0.3.1 checkout with the pinned CUDA dependencies installed:

~~~bash
python scripts/download_public_assets.py --asset all --dense-model 3139 --models-dir models
python scripts/download_public_assets.py --asset all --dense-model 3139 --models-dir models --verify-only
python -m flashh3vr --kind full-video --input input.mp4 --output restored-3139.mp4 --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/dense-3139/flashh3vr-dense-3139.safetensors --face-weights models/yolov11m-face.pt --target-side 448 --max-frames 90 --device cuda:0
~~~

Use a supported short, silent SDR clip and a finite frame limit covering the
complete input. Other existing image/head-window commands use the same optional
`--dense-weights` path. The full-video JSON records the selected filename, SHA256
and step 3139. No new repair steps or numerical backend are introduced.

To return to the default, prepare it with `--asset dense` (no model option), pass
`--dense-weights models/flashh3vr-dense-1837.safetensors` and choose a new output
path. Its original card/checksums remain in `models`; 3139's remain in
`models/dense-3139`.

中文：先安装v0.3.1，显式使用 `--dense-model 3139` 下载；程序会放入独立的
`models/dense-3139` 目录。推理时只替换上面命令中的Dense权重路径，默认1837
保留。3139对新增训练窗更好，但旧448照片嘴部仍弱于3258；不代表所有素材都
改善，也未认证通用人像、长片、FPS或16GB。保留各自许可、说明与校验文件。

Project-owned source remains AGPL-3.0-only. The H3-dependent adapter keeps the
MiniMax H3 Community License and downstream conditions, separately from source
licensing. Keep LICENSE-MINIMAX-H3 and NOTICE with the weights. See the retained
license; publication does not change its terms.

No training media, portraits, download/source lists, private paths, H3 base weights,
optimizer/scaler state, RNG or caches are distributed. Public availability of
training media alone does not establish permission to redistribute that media.
The evaluation is subject-specific; use on other people is outside the tested
scope. Output detail is not evidence of a person's actual appearance. This
independent project implies no endorsement by MiniMax, the subject or publishers.
