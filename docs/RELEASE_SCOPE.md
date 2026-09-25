# Release scope — 0.2.0

This version introduces the current Dense latent-restoration inference path and registers its 1837-step adapter format.

Included:

- A standalone flashh3vr package with strict safetensors loading, the pinned dequantized-FP16 H3 numerical path, native spatial tiling, image and real 22-frame video-window APIs, and a command-line entry.
- Current-model English/Chinese documentation, model card, numerical limitations, dependency identities and source/license notices.
- Synthetic CPU contract tests and a documented bounded comparison against the private research implementation.
- Existing version 0.1 engineering code and historical NAF evidence, explicitly separated from current model claims.

Excluded from Git:

- Dense/H3/NAF/YOLO weight binaries, optimizer checkpoints, media, raw arrays, crops, generated portrait examples and private research data.
- Local environments, CUDA binaries, credential files, internal task instructions and private execution logs.

Adapter safetensors have been exported separately. Their availability is tracked in [ASSETS.md](ASSETS.md); source publication does not imply a public weight download.

No new training was performed for this release. The current API is for prepared head crops, not a complete long-video detector/paste-back application. Old throughput and VRAM tags apply only to the [historical version](HISTORICAL_0_1_BASELINE.md).

The publication inventory records current source origins and the release verification report. Private test inputs are kept outside this repository.
