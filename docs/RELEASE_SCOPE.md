# Release scope — 0.3.0

This version adds automatic full-frame file inference around the published Dense Inter 1837 core. The learned weights are unchanged from version 0.2.0.

Included:

- A standalone flashh3vr package with strict safetensors loading, the pinned dequantized-FP16 H3 numerical path, native spatial tiling, image and real 2–22-frame video-window APIs, and a command-line entry.
- Automatic face detection, stable geometry, cut/gap segmentation, overlapping native windows, full-frame correction paste-back, H.264 encoding and a per-run JSON report.
- A fresh-clone agent guide and explicit pinned-asset download/check script, without private run receipts.
- Current-model English/Chinese documentation, model card, numerical limitations, dependency identities and source/license notices.
- Synthetic CPU contract tests and a documented bounded comparison against the private research implementation.
- Existing version 0.1 engineering code and historical NAF evidence, explicitly separated from current model claims.

Excluded from Git:

- Dense/H3/NAF/YOLO weight binaries, optimizer checkpoints, media, raw arrays, crops, generated portrait examples and private research data.
- Local environments, CUDA binaries, credential files, internal task instructions and private execution logs.

The Dense Inter 1837 safetensors are distributed separately as a GitHub Release bundle, together with the model card, license, notice and checksums. Download links and exact identities are in [ASSETS.md](ASSETS.md). The weight binary is a Release asset and is not committed to Git.

No new training was performed for this release. The head-crop API is retained, and the new full-video entry automatically handles a finite supported SDR clip. It is an in-memory implementation, rejects audio, and does not establish whole-film streaming or multi-person identity selection. Old throughput and VRAM tags apply only to the [historical version](HISTORICAL_0_1_BASELINE.md).

The publication inventory records current source origins and the release verification report. Private test inputs are kept outside this repository.
