# Using the research source

## What works without model assets

Install the project in an isolated Python environment after installing PyTorch and torchvision for your platform:

```bash
python -m pip install -e ".[dev,video,models]"
python -m pytest -q
```

The repository includes CPU tests for detection scheduling, byte preparation, input overlap, continuous head geometry, native frame/chunk contracts, timestamps, color conversion, cache behavior and checkpoints. Model classes are not accepted on the strength of these tests.

The original Windows environment is recorded in `requirements.gpu.lock.txt`. It includes a Windows Python 3.12 CUDA wheel and must not be used as a universal platform lock. The current INT8 runtime loader requires the specific Windows extension and isolated cuBLAS13 recorded in the manifests. Linux CUDA inference and other GPUs have not been accepted for this release.

## Current inference API

The active research file function is:

```python
from h3ce.infer.head_video_file import restore_video_file

result = restore_video_file(
    source, destination, bridge, head, detector,
    max_frames=60,
    side=256,
    long_edge=768,
    reuse_execution=True,
    optimize_input=True,
    overlap_input=True,
)
```

This is an integration example with already-loaded objects, not a complete runnable quick start. The adapted checkpoint is currently unavailable. The required objects and their source constructors are:

| Object | Required setup |
|---|---|
| `bridge` | `H3VAEBridge(Int8KitchenH3Backend.from_locked(...))`; current component lock and external INT8 weight required |
| `detector` | Context-managed `Yolo11BatchExecutor(config, root, profile='dual16', device='cuda:0', byte_preparation='direct_bgr')` |
| `head` | Official `load_model()` from `scripts/research_nafnet_gopro32.py`, then `NAFHead3`; restore the **accepted** tail, freeze all parameters, move to CUDA and wrap with `NAFHead3Inference` |

Use `configs/project.int8.yaml` to supply recorded detection and component settings. Its older VAE factory declaration does not select the accepted Comfy Kitchen path by itself; the explicit backend above is necessary. `CheckpointManager` in `h3ce/train/checkpoint.py` checks a checkpoint receipt and exact training contract before loading with `weights_only=True`. Those private training receipts are not part of this release.

`scripts/benchmark_closure_baseline.py` preserves the complete historical construction, execution and observation recipe. It binds private data, checkpoint and evidence files; running it from a fresh clone fails its preflight. Do not remove those checks to claim reproduction.

## Input and output boundaries

- Explicit finite `max_frames` is required. The implementation retains frame tensors in memory; it is not an unbounded streaming processor.
- Input requires one SDR video stream with usable source color metadata and increasing timestamps. Audio is currently rejected. HDR needs a separately defined transform.
- The current restoration path selects segments with a unique eligible face of at least 64 pixels. Missing, ambiguous, isolated or cut-separated frames do not share H3 context. This is not multi-person identity tracking.
- Output needs a new destination file and even dimensions. H.264 uses CRF 18, preset `fast`, YUV420p, limited BT.709 and source timestamps. Timing includes final encoding and closing.
- Geometry uses limited future context, so the throughput measurement is not a causal streaming-latency guarantee.

## Resource observation and further testing

`scripts/closure_resource_monitor.py` provides `Sampler(root_pid=...)`, `sample()` and `close()` for an external sampler process. It observes whole-machine/per-core CPU, process-tree CPU/RSS and device GPU utilization/memory; unavailable driver fields remain null. It records local process information, so review raw logs before sharing them.

For a new GPU experiment, define the input, checkpoint, configuration and number of full passes first. Start resource sampling before model loading; align samples to loading and inference phase timestamps. Do not automatically rerun failed measurements. CPU tests do not need a GPU, real model assets, or NVML initialization.

The legacy `h3ce` CLI, character LoRA components and older training configurations remain source material for earlier workflows. They do not imply a finished CLI or role-training product for the accepted NAF path.
