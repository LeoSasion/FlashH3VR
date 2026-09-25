# Dense Inter inference

The current entry is the **flashh3vr** package. It restores already cropped head images or exactly 22 real, continuous video frames. Detection, source cropping, shot segmentation, identity recognition, audio, full-frame paste-back and long-video stitching are not part of this entry.

## Installation and assets

Use Python 3.12 and a CUDA-capable PyTorch installation. The recorded validation version is PyTorch 2.10.0.

~~~bash
python -m pip install -e ".[inference]"
python -m flashh3vr --help
~~~

Download [the Dense Inter 1837 bundle](https://github.com/LeoSasion/FlashH3VR/releases/download/v0.2.0/flashh3vr-dense-1837-bundle.zip) and extract it into models, keeping its license and notice alongside the safetensors file. Obtain the exact external H3 safetensors file listed in [ASSETS.md](ASSETS.md), subject to its terms. No automatic download or random-weight fallback is provided. A different H3 or Dense hash is rejected.

## One head image

An RGB PNG with a square side of 256, 448, 640 or 832:

~~~bash
python -m flashh3vr --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/flashh3vr-dense-1837.safetensors --kind image --input head.png --output restored.png
~~~

For an actual half-size source, explicitly align it to the training canvas. This example requires a 224 × 224 input and produces a 448 × 448 output:

~~~bash
python -m flashh3vr --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/flashh3vr-dense-1837.safetensors --kind image --input head224.png --output restored448.png --half-input --target-side 448
~~~

The image path preserves the verified single-image temporal context handling. RGB8 export clips to [0,1]; the tensor API defaults to unclamped float output. No claim is made that canvas enlargement creates real source detail.

## A real video window

Prepare a non-pickled NumPy array with shape **[22,S,S,3]**, RGB float32 in [0,1], with S one of the four bucket sizes. Frames must come from one continuous shot with consistent head geometry. A separate JSON array must contain their 22 real source PTS values in seconds, finite and strictly increasing; retain the original frame IDs/PTS in your own input records. Do not fabricate a clip by repeating independent images.

~~~bash
python -m flashh3vr --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/flashh3vr-dense-1837.safetensors --kind video --input window.npy --pts-json pts.json --output restored.npy
~~~

Video output retains the input order and frame count as float32 **[22,S,S,3]**, unclamped. The CLI also writes a **.pts.json** sidecar with the source time mapping; the output array itself is not an MP4 container. A half-size video array can use the same paired half-input/target-side flags. Image and video alignment reproduce their respective recorded input preparation methods.

Existing output files are rejected unless you pass --overwrite. CUDA is required for the verified model numerical path. CPU tests do not imply CPU inference support.

## Python API

~~~python
import torch
from flashh3vr import DenseRestorer, align_half_input

restorer = DenseRestorer(
    h3_weights="models/minimax_h3_video_vae_int8_convrot.safetensors",
    dense_weights="models/flashh3vr-dense-1837.safetensors",
    device="cuda",
)

# Supply your own RGB head tensor, float32 in [0,1].
# Image: [1,3,S,S]; video: [1,3,22,S,S].
image = image.to(restorer.device)
restored_image = restorer.restore_image(image)
video = video.to(restorer.device)
restored_video = restorer.restore_video_window(video, pts=source_pts_seconds)

# Optional explicit low-resolution input alignment uses [1,3,T,S/2,S/2].
aligned = align_half_input(low_tensor, kind="image", target_side=448)
~~~

The variables above are caller-provided real inputs, not bundled private examples. The model preserves native 256-pixel spatial tiles with at least 64 pixels of overlap, computes one Dense residual, and uses the external H3 checkpoint's dequantized FP16 path. It does not run the legacy Comfy Kitchen INT8 kernels.

The [old NAF API](HISTORICAL_0_1_USAGE.md) describes version 0.1 only. It is not the constructor for this checkpoint.
