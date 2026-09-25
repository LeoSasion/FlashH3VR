# FlashH3VR

**Flash H3 Video Restoration** — a Dense residual adapter inside a frozen H3 video VAE.

[简体中文](README.zh-CN.md) · [Inference guide](docs/USAGE.md) · [Model card](MODEL_CARD.md) · [Assets](docs/ASSETS.md) · [Licensing](THIRD_PARTY_NOTICES.md)

The current research baseline is **Dense Inter 1837**, trained at 256, 448, 640 and 832 pixels. A frozen H3 encoder, one learned latent correction and a frozen decoder restore an already cropped head image or a real head-video window. The current path uses no diffusion transformer, iterative repair or RGB NAF tail.

The new **flashh3vr** package is the current inference entry. Older h3ce/NAF recipes remain historical source. Their full-frame paste-back, 24 FPS measurements and 16GB-VRAM tag do **not** certify this model.

**[Download Dense Inter 1837 weights](https://github.com/LeoSasion/FlashH3VR/releases/download/v0.2.0/flashh3vr-dense-1837-bundle.zip)** — 11.18 MiB ZIP, including the model license and checksums. Extract it into models and obtain the external H3 checkpoint listed in [Assets](docs/ASSETS.md).

## Install

Python 3.12 and PyTorch 2.10.0 are the verification environment. Install a PyTorch build appropriate for your CUDA device, then:

~~~bash
python -m pip install -e ".[inference]"
python -m flashh3vr --help
~~~

Inference needs two separate files: the exact external H3 INT8 ConvRot checkpoint and this project's Dense Inter safetensors. H3 is decoded to the tested FP16 numerical path; this is not the old Comfy Kitchen INT8 execution path. See [asset identities and availability](docs/ASSETS.md), then follow the [image and video-window guide](docs/USAGE.md).

## Current model

| Property | Value |
|---|---|
| Adapter | 3,152,128 parameters, four FP32 tensors |
| Adaptation | Go Youn-jung portrait restoration |
| Export | flashh3vr-dense-1837.safetensors, 12,609,192 bytes |
| Training | 105.005 effective minutes, 971 new updates, Adam step 1837 |
| Spatial execution | Native 256-pixel tiles, at least 64 pixels of overlap |
| Input/output | Prepared head crops; real video frames and source PTS |

Relative to the study's 866-step starting checkpoint, train-partition edge error decreased **8.08%**, and reused development-window edge error decreased **4.10%**. The 448-pixel photograph mouth metric regressed about **2.2%**. Reviewed panels showed small visual differences and no obvious new severe artifacts. These are limited internal observations, not fresh independent generalization or a speed claim. [Definitions and limitations](docs/BASELINE.md).

## Code and weights

Project-owned source remains **AGPL-3.0-only**; third-party source retains its original notices. The upstream-dependent adapter and H3 model terms are separate from the source license. See [distribution status](docs/ASSETS.md). No training media, generated portrait examples, optimizer state or H3 foundation weights are included in Git.

~~~bash
python -m pip install -e ".[dev,video,models,inference]"
python -m pytest -q
~~~

CPU tests use synthetic inputs and do not substitute for real-model verification. Full-frame detection, tracking, audio and paste-back are outside the new inference entry. See [release scope](docs/RELEASE_SCOPE.md).

FlashH3VR is independent and does not imply endorsement by MiniMax or the person represented in the adaptation data.
