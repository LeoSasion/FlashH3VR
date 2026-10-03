# Release scope — 0.3.1

This version retains the automatic full-frame path from 0.3.0 and adds an explicitly
selected, pinned experimental Dense Inter 3139 asset. The adopted 1837 baseline and
default download remain unchanged. Each model has separate weight/license/card
files; runtime reports identify the model actually loaded. The new optional
bundle and its reported 448 tradeoff are documented in [DENSE3139.md](DENSE3139.md).

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

Dense Inter 1837 and optional 3139 safetensors are distributed separately as GitHub
Release bundles with their model card, license, notice and checksums. Download
links and exact identities are in [ASSETS.md](ASSETS.md). Weight binaries are
Release assets and are not committed to Git.

No new training was performed for this release. The head-crop API is retained, and the new full-video entry automatically handles a finite supported SDR clip. It is an in-memory implementation, rejects audio, and does not establish whole-film streaming or multi-person identity selection. Old throughput and VRAM tags apply only to the [historical version](HISTORICAL_0_1_BASELINE.md).

The publication inventory records current source origins and the release verification report. Private test inputs are kept outside this repository.
