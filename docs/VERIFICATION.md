# Version 0.2.0 verification

- 374 CPU tests and 19 subtests passed; one external-model test was deselected.
- The wheel built successfully. Imports and CLI help passed from an extracted wheel outside the source checkout, without importing the private research package.
- The four exported Dense tensors are bitwise equal to the adopted 1837-step checkpoint.
- One admitted 256-pixel image and one real 22-frame 832-pixel head window were executed with the published numerical implementation and exact pinned assets. The encoded latent and final raw output were bitwise equal to their frozen research references; maximum absolute difference was zero.
- These are bounded implementation-parity checks, not new visual-quality, full-length-video, FPS or minimum-VRAM certification.
- Private test media, paths, frame identifiers and raw outputs are excluded from the publication summary.

[Machine-readable summary](../release_verification.json). H3 remains an externally obtained asset; adapter availability is separate from successful code validation.
