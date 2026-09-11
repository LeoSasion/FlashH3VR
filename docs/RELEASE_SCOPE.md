# Source release scope — 0.1.0

The first public FlashH3VR release packages the September 12, 2026 research closure as source code. No new training, model forward pass or GPU speed measurement is performed to publish it.

Included:

- Project Python modules, including the accepted video/head inference path and previous engineering foundations.
- Selected historical training and benchmark recipes with their imported project helpers. They require the original experiment receipts and are not fresh-clone commands.
- Original third-party source notices, selected NAFNet sources, component origins and hashes.
- A selected CPU regression suite and documentation of the accepted model, performance, quality and resource observations.

Excluded:

- All model checkpoints and pretrained model assets.
- Training/evaluation media, derived arrays, crops, cached features and output videos.
- Private experiment logs, host process inventories, local environments, downloaded binary runtimes and agent/task instructions.
- Unrelated historical experiments and internal acceptance registries that would imply independently reproducible public evidence.

## Packaging changes

The Python distribution is named `flashh3vr`; imports retain `h3ce`. The helper `scripts` package is explicitly included because the current video pipeline imports its chunk/overlap functions.

Two copied loaders now resolve NAF provenance and cuBLAS13 runtime manifests relative to the repository. Their numeric operations are unchanged. Component manifests remove local acquisition records and machine-specific paths. Current public manifests therefore have new identities; historical results must not be described as measurements of these changed file bytes.

`publication_origins.json` records the original research hashes for copied files and describes modified copies. The original research workspace remains separate from this public repository.

The historical 32-step training recipe contains temporal objectives because it documents how the accepted model was obtained. Their presence does not restart that research direction. Further temporal optimization and exploratory training remain paused.
