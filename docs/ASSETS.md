# Model assets and availability

Source and learned weights have separate distribution terms.

| Asset | SHA256 | Availability |
|---|---|---|
| Dense Inter 1837, four FP32 tensors | a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804 | Export prepared and verified locally; public download is not enabled yet. |
| External H3 INT8 ConvRot | 9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410 | Obtain separately from the [pinned upstream file](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/minimax_h3_video_vae_int8_convrot.safetensors), under H3 terms. |
| Training/evaluation media, crops, raw outputs and caches | Not distributed | Not included. |
| Full optimizer checkpoint and H3 base weights | Not distributed by this project | Not included in the source repository or adapter package. |

The Dense file contains 3,152,128 parameters and is 12,609,192 bytes. [Machine-readable export identity](../configs/dense1837.weights.json). The release loader accepts the exact two model hashes and checks the Dense names, shapes, dtype and finite values.

Use independently acquired authorized files at:

~~~text
models/
  minimax_h3_video_vae_int8_convrot.safetensors
  flashh3vr-dense-1837.safetensors
~~~

The current Dense path does not need YOLO, NAFNet, Comfy Kitchen or the old Windows cuBLAS loader. The external ConvRot tensor file is converted to the same dequantized FP16 execution used by training; original FP16 H3 weights are not an interchangeable substitute.

## Weight distribution

The project-owned inference code is AGPL-3.0-only. The H3-dependent adapter is being prepared with the **MiniMax H3 Community License** and its notices, not a claim of unrestricted open-source model licensing. The license includes excluded territories and downstream-use conditions. See the [official license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and [bundled reference copy](../h3ce/vae/_upstream/MiniMax-H3-LICENSE).

A recipient-approved download route is being arranged; this page will carry its exact model/revision link when enabled. There is presently no adapter download URL. The source can be installed and tested, but a new user cannot reproduce the learned restoration without that separately supplied adapter.

The source license does not grant rights to the training media, source images or a subject's endorsement. See [MODEL_CARD.md](../MODEL_CARD.md). Historical NAF weights remain withheld and are not part of this adapter.
