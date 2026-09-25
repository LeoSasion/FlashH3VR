# Working with the public FlashH3VR release

This repository is the standalone public source release. For automatic video
inference, start with [docs/AGENT_REPRODUCTION.md](docs/AGENT_REPRODUCTION.md) and
[docs/FULL_VIDEO.md](docs/FULL_VIDEO.md). The current entry is
`python -m flashh3vr --kind full-video`; the separate image/head-window modes are
documented in [docs/USAGE.md](docs/USAGE.md).

- Use the pinned model identities and `scripts/download_public_assets.py` for
  explicit acquisition. Keep the bundled model license and notice. No private
  dataset, training receipt, NAF weights or author-specific directory is needed.
- Preserve real source frames/PTS, cut boundaries, skipped-frame reasons and
  native H3 temporal/spatial context. Context padding is not real video data.
- The model applies one Dense latent correction. Use no repeated repair, silent
  backend replacement or missing-weight fallback.
- Run relevant CPU tests when changing code. Real-model checks require the
  external assets and CUDA. Report the checks actually performed; a successful
  synthetic test is not a perceptual-quality or performance result.
- Use new output paths and retain the generated report. The automatic path is
  finite and memory-bound; it rejects audio and unsupported color contracts.
- Old `h3ce`/NAF recipes, speed results and 16GB labels are historical. Do not
  present them as current Dense guarantees. No current-model training recipe or
  private media is published by the inference reproduction guide.
