# Licensing and third-party notices

## Project-owned code

FlashH3VR project-owned code is licensed under **GNU AGPL-3.0-only**. The full text is in [LICENSE](LICENSE). This selection aligns the application source with the AGPL option used by the current Ultralytics runtime.

AGPL permits commercial use subject to its conditions. When a covered modified program supports remote network interaction, section 13 requires an opportunity for those users to receive its corresponding source. This does not turn separately licensed models, data or system libraries into AGPL material, and this source release does not assert that a bundled end-user product has cleared all component rights.

Primary references: [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html), [GNU's explanation](https://www.gnu.org/licenses/why-affero-gpl.html), [Ultralytics licensing](https://www.ultralytics.com/license).

## Source included in this repository

| Component | Revision / location | Retained license and changes |
|---|---|---|
| Ostris AI Toolkit H3 VAE | `7690ea62c87133410ffe1596aa222a6e0e9069f7`; `flashh3vr/_vendor.py`, `h3ce/vae/_vendor.py` and `_upstream/vae.py` | MIT, copyright 2024 Ostris, LLC. [License](h3ce/vae/_upstream/LICENSE). The local vendor removes the unrelated loader mixin import/base; the original source is retained for parity checks. The new Dense package carries its own copy in `flashh3vr/licenses/`. |
| NAFNet and contained BasicSR code | `2b4af71ebe098a92a75910c233a3965a3e93ede4`; `third_party/NAFNet-2b4af71ebe098a92a75910c233a3965a3e93ede4/` | [Original combined license](third_party/NAFNet-2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE): NAFNet MIT, copyright 2022 megvii-model; BasicSR Apache-2.0, copyright 2018–2020 BasicSR Authors. Selected upstream files are copied unchanged. The project loader supplies two import bindings locally while retaining upstream class bodies. |
| Ultralytics architecture description and integration | `h3ce/data/detect_yolo11.py`; dependency `ultralytics==8.4.142` | Runtime is external. Its YOLO11 topology is referenced by validation code. [Ultralytics AGPL source license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) applies to upstream material. |

Upstream source: [AI Toolkit VAE](https://github.com/ostris/ai-toolkit/blob/7690ea62c87133410ffe1596aa222a6e0e9069f7/extensions_built_in/diffusion_models/minimax_h3/src/vae.py), [NAFNet](https://github.com/megvii-research/NAFNet/tree/2b4af71ebe098a92a75910c233a3965a3e93ede4).

## External components, not distributed

- **MiniMax H3 / Kijai INT8 ConvRot weights:** governed by the MiniMax H3 Community License, not GPL/AGPL. A [reference copy](h3ce/vae/_upstream/MiniMax-H3-LICENSE) and [pinned upstream source](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE) are provided. It includes territorial and use restrictions, commercial conditions and conditions concerning use of H3 outputs to improve other models. Do not infer authorization for NAF adaptation or checkpoint redistribution from the code license.
- **NAFNet-GoPro weights:** obtained separately from the upstream project. Its source-code license alone is not a blanket clearance of pretrained-weight or GoPro dataset rights.
- **YOLO11m-face weights:** separately sourced from [akanametov/yolo-face](https://github.com/akanametov/yolo-face), whose repository advertises GPL-3.0. This is distinct from the current Ultralytics runtime's AGPL-3.0 terms. No combined or unrestricted weight license is asserted here.
- **Comfy Kitchen and NVIDIA cuBLAS:** external runtime dependencies. Follow their own distribution terms. This repository supplies hashes and origins, not their binaries.
- **PyTorch, torchvision, PyAV/FFmpeg, Ultralytics, safetensors, NumPy, Pillow, LPIPS and other Python dependencies:** installed separately under their respective terms. H.264 availability depends on the selected FFmpeg build.

## Current Dense Inter 1837 adapter

The current path is frozen H3 -> Dense latent residual -> frozen H3. It does not use the historical RGB NAF tail. Project-owned inference code remains AGPL-3.0-only. The separately distributed adapter is subject to the MiniMax H3 Community License and its applicable restrictions; this is not an assertion that the upstream-dependent weights have an unrestricted open-source license. See [the model card](MODEL_CARD.md) and [asset availability](docs/ASSETS.md).

The new package uses dequantized FP16 tensors from the pinned ConvRot asset and requires no Comfy Kitchen binary. It ships no H3 weights. Third-party Python/runtime packages remain external dependencies under their own terms.

## Training and evaluation material

Dense Inter 1837 uses the portrait photo/video data summarized in the model card. The original images, videos, crops and generated output portraits are not included. Public availability and research admission are not a blanket redistribution permission. The current adapter's publication scope must not be inferred from the historical NAF checkpoint discussion below.

No training data, evaluation video, extracted frames or generated demo footage is distributed. ElFuente/Narrator material used by the historical version 0.1 NAF adaptation has a recorded **CC BY-NC-ND 4.0** notice; its research admission was not approval for public checkpoint or product distribution. See [the original source notice](https://media.xiph.org/video/derf/ElFuente/Netflix_Narrator_Copyright.txt).

This paragraph concerns the historical NAF balanced32 checkpoint, not Dense Inter 1837: its legal status was not resolved by version 0.1, and it remains withheld. The project does not assert that every model trained on such material is automatically a derivative work, nor that research admission establishes permission to distribute it.
