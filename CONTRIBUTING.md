# Contributing

Start with [release scope](docs/RELEASE_SCOPE.md) and [usage](docs/USAGE.md). The current release is research source, with adapted checkpoint distribution unresolved.

For code changes, explain the behavior affected and run the relevant CPU tests. Keep source-time geometry, color conventions, component hashes and safe checkpoint loading explicit. A CPU test does not establish model quality or GPU performance.

Performance reports should state the input, GPU, runtime, loading treatment, full-pass count, CPU/GPU sampling and output agreement. Distinguish device-wide memory from framework allocations. Do not claim broad real-time performance from one short clip.

Do not contribute third-party weights, datasets, private media or raw process logs without established rights and review. Project code contributions are under AGPL-3.0-only unless a retained third-party license applies.

Exploratory training and temporal optimization are currently paused. The immediate project direction is usable packaging, distribution-rights clarification and reproducible bounded validation.
