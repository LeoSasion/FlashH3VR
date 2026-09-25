# Model assets and availability

Source and learned weights have separate distribution terms.

| Asset | SHA256 | Availability |
|---|---|---|
| Dense Inter 1837, four FP32 tensors | a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804 | [Download the GitHub Release bundle](https://github.com/LeoSasion/FlashH3VR/releases/download/v0.2.0/flashh3vr-dense-1837-bundle.zip) (11.18 MiB ZIP). |
| External H3 INT8 ConvRot | 9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410 | Obtain separately from the [pinned upstream file](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/minimax_h3_video_vae_int8_convrot.safetensors), under H3 terms. |
| YOLO11m face, needed only for automatic full-video | 6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661 | Obtain from the [versioned upstream asset](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt), under its applicable terms. |
| Training/evaluation media, crops, raw outputs and caches | Not distributed | Not included. |
| Full optimizer checkpoint and H3 base weights | Not distributed by this project | Not included in the source repository or adapter package. |

The Dense file contains 3,152,128 parameters and is 12,609,192 bytes. [Machine-readable export identity](../configs/dense1837.weights.json). The release loader accepts the exact two model hashes and checks the Dense names, shapes, dtype and finite values.

Extract the complete Dense bundle into models, retaining its license, notice, model card and checksums. Obtain the external H3 file separately under its terms. The two model files should be at:

~~~text
models/
  minimax_h3_video_vae_int8_convrot.safetensors
  flashh3vr-dense-1837.safetensors
~~~

The head-crop Dense API does not need YOLO. The automatic full-video entry additionally uses the pinned YOLO11 face detector; no person detector is needed. Neither current entry needs NAFNet, Comfy Kitchen or the old Windows cuBLAS loader. The external ConvRot tensor file is converted to the same dequantized FP16 execution used by training; original FP16 H3 weights are not an interchangeable substitute.

## Explicit asset preparation

For a fresh source checkout, the standard-library script downloads the pinned external assets and extracts the license-bearing Dense bundle:

~~~bash
python scripts/download_public_assets.py --asset all --models-dir models
python scripts/download_public_assets.py --asset all --models-dir models --verify-only
~~~

Use `--asset dense`, `--asset h3` or `--asset face` to prepare one component. Verification checks exact sizes and SHA256; `--verify-only` never downloads or writes. Existing wrong files are left untouched and cause an error. See [public_assets.json](../configs/public_assets.json) for the complete machine-readable download manifest and [the automatic workflow](FULL_VIDEO.md) for usage.

## Weight distribution

The project-owned inference code is AGPL-3.0-only. The H3-dependent adapter is distributed with the **MiniMax H3 Community License** and its notices, not a claim of unrestricted open-source model licensing. The license includes excluded territories and downstream-use conditions. See the [official license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and [bundled reference copy](../h3ce/vae/_upstream/MiniMax-H3-LICENSE).

The adapter is hosted directly in [GitHub Release v0.2.0](https://github.com/LeoSasion/FlashH3VR/releases/tag/v0.2.0). The ZIP contains the safetensors file, model card, MiniMax H3 license, NOTICE, weights manifest and SHA256SUMS. Its SHA256 is a2199b0bdd4b187feeada476b81526677cf767420248971ec0879fc59e12f236 (11,720,843 bytes).

Download and extract it, then follow [the inference guide](USAGE.md). The adapter requires the external H3 checkpoint above; that base checkpoint is not included in the bundle.

The source license does not grant rights to the training media, source images or a subject's endorsement. See [MODEL_CARD.md](../MODEL_CARD.md). Historical NAF weights remain withheld and are not part of this adapter.
