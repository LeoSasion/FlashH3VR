# Reproduce automatic Dense video inference

This guide is for an agent working from a fresh public clone. The current automatic entry is `python -m flashh3vr --kind full-video`. Do not use the historical NAF benchmark or try to locate the author's private `runs` directory.

## Inputs to obtain from the user

Use the user's video file, a new output path, their CUDA device, and an explicit finite input-frame limit. Start with a short silent SDR clip. The code does not identify the actor or follow a chosen person through crowded footage; explain this if their input needs identity selection. The model's evaluated subject and limitations are in [MODEL_CARD.md](../MODEL_CARD.md).

## Reproducible setup

~~~bash
git clone https://github.com/LeoSasion/FlashH3VR.git
cd FlashH3VR
git checkout v0.3.0
python -m venv .venv
~~~

Activate the environment using the normal command for the user's shell. Install a CUDA-enabled torch/torchvision pair for their platform; the development verification versions are torch 2.10.0 and torchvision 0.25.0. A CPU-only PyTorch wheel cannot run the pinned restoration path. Do not install the historical Windows DLL runtime or treat `requirements.gpu.lock.txt` as a universal environment lock.

~~~bash
python -m pip install -e ".[full-video,dev]"
python -m flashh3vr --help
python -m pytest -q
python scripts/download_public_assets.py --asset all --models-dir models
python scripts/download_public_assets.py --asset all --models-dir models --verify-only
~~~

The downloader obtains the Dense bundle from GitHub and the two pinned upstream models from their recorded sources. It preserves the Dense license/notice and refuses mismatching files. The manifests and loader hashes are the authority for this release; do not silently substitute another H3, face detector or Dense checkpoint to get past an error. Users who already have exact files can prepare them manually according to [ASSETS.md](ASSETS.md).

## Execute the real entry

Replace input.mp4 with the user's supported clip and choose a finite frame limit that covers all of it:

~~~bash
python -m flashh3vr --kind full-video --input input.mp4 --output restored.mp4 --h3-weights models/minimax_h3_video_vae_int8_convrot.safetensors --dense-weights models/flashh3vr-dense-1837.safetensors --face-weights models/yolov11m-face.pt --target-side 448 --max-frames 90 --device cuda:0
~~~

No manual face crop or NumPy input is required for this entry. Full frame dimensions are retained unless the caller explicitly supplies `--working-long-edge`. Use a new output path and keep the generated `.flashh3vr.json` report.

The sequence is decode with source PTS → detect faces → split cuts/gaps/ineligible frames → stabilize and crop → native H3/Dense windows → merge same-frame overlap → paste correction → encode H.264. Only true decoded input frames count toward the output. Tail context padding is logged separately.

## Verify the result

Check that a video and its report were produced, that the complete source frame count and presentation times were preserved, and that at least one eligible segment was actually restored. Inspect processed/skipped-frame records. View the resulting video at normal size, especially crop edges, cuts, lips and eyes. CPU tests or a zero process exit code alone do not establish perceptual quality.

If FFmpeg's ffprobe is available, inspect both files:

~~~bash
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,nb_read_frames,time_base -of json input.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,nb_read_frames,time_base -of json restored.mp4
ffprobe -v error -select_streams v:0 -show_entries frame=best_effort_timestamp_time -of csv=p=0 restored.mp4
~~~

Container time bases may be represented differently; compare rational presentation times, not only raw PTS integers. The compositor preserves pixels outside the correction region before encoding, while the output codec is lossy.

## Handle a failure precisely

| Failure | Next action |
|---|---|
| Missing or wrong model hash | Obtain the exact pinned file; keep the loader check. |
| CUDA unavailable | Install a matching CUDA PyTorch build and confirm the device; CPU tests are not CPU model support. |
| Input exceeds `--max-frames` | Prepare a shorter clip or deliberately raise the finite bound within available memory. The process must not silently truncate. |
| Audio stream present | This entry has no audio preservation/remux path. Prepare a separate silent input deliberately; do not claim the output preserves sound. |
| Missing color tags or HDR | Determine the real source color contract and prepare a supported SDR file. Relabeling HDR as BT.709 is not a conversion. |
| No eligible head segment | Inspect detections and input face size. An all-skipped copy is not a successful result. |
| Some frames skipped | Read the recorded reasons; there is no hidden identity recognition or cross-cut context. |
| Out of memory | Use a shorter input or explicitly smaller working canvas/head bucket and report the changed scope. Do not silently switch numerical backends. |

The feature is a reproducible bounded automatic pipeline. Do not describe it as verified whole-film streaming, general-person restoration, FPS performance or 16GB compatibility. Current Dense training orchestration and training data are not part of this reproduction procedure.
