# Contributing

Read [release scope](docs/RELEASE_SCOPE.md), [inference usage](docs/USAGE.md) and [the model card](MODEL_CARD.md).

Describe the changed behavior and run the relevant CPU tests. Preserve source-time ordering, explicit input/output ranges, strict model hashes and native H3 spatial/temporal contracts. Do not turn a failed shape/hash check into a silent fallback.

Synthetic CPU tests do not establish perceptual quality or GPU performance. Reports must state the exact checkpoint, input scope, numerical backend and what was actually measured. The historical 24 FPS and 16GB labels do not apply to Dense Inter 1837.

Do not contribute private media, credentials, optimizer states or third-party assets without the appropriate rights. Project-owned contributions are AGPL-3.0-only; upstream code keeps its original notices. Learned-weight terms are separate.

The published path performs a single latent correction. New training, temporal objectives or model architecture changes require their own documented research protocol.
