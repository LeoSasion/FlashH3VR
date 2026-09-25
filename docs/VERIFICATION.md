# Version 0.3.0 verification

- **386 CPU tests and 19 subtests passed**, with one external-model test deselected. Coverage includes cut/gap/missing/ambiguous-face segmentation, same-frame overlap, short native context, paste-back, frame limits, rational PTS and explicit asset verification.
- The wheel built successfully. The packaged Python files matched the frozen source. The full-video module and CLI imported from an extracted wheel outside the checkout using isolated Python; imported modules were checked to come from that wheel directory.
- The actual public full-video CLI processed one 44-frame silent SDR file at 768 × 432 with a 448-pixel head canvas: **44 frames restored, none skipped**, with windows of 22, 22 and 10 real frames.
- Two additional public-API checks used the same excerpt's 2- and 6-frame prefixes. Their H3 contexts contained 5 and 22 frames respectively, while the encoded results retained exactly 2 and 6 real frames.
- Independent CPU decoding confirmed the original rational presentation times, complete frame counts and canvas dimensions for all three outputs. These are 52 input-frame executions from one 44-frame source excerpt, not independent sources or a fresh generalization benchmark.
- The input fixtures were private full-frame downsamples of a 4K source, prepared as silent SDR. The check does not certify original-resolution 4K processing or audio preservation. Model weights were unchanged and no training ran.
- Eleven normal-size before/after panels were actually reviewed, including overlap and tail positions. No obvious new paste boundaries, grid, ghosting or facial structural damage appeared in those views; outputs looked slightly softer. This is not a clarity-gain result. Ordinary playback was not completed, so temporal flicker and transition continuity remain unverified.
- The public asset helper completed a real anonymous Dense bundle download, extraction and offline re-verification. Existing H3 and face-model files passed exact SHA checks. Pinned upstream asset URLs returned successful anonymous metadata responses.

[Current machine-readable summary](../release_verification.json). The model's actual quality limitations remain in [MODEL_CARD.md](../MODEL_CARD.md). The historical numerical-parity evidence below remains bounded to its original cases.

# Version 0.2.0 verification

- 374 CPU tests and 19 subtests passed; one external-model test was deselected.
- The wheel built successfully. Imports and CLI help passed from an extracted wheel outside the source checkout, without importing the private research package.
- The four exported Dense tensors are bitwise equal to the adopted 1837-step checkpoint.
- One admitted 256-pixel image and one real 22-frame 832-pixel head window were executed with the published numerical implementation and exact pinned assets. The encoded latent and final raw output were bitwise equal to their frozen research references; maximum absolute difference was zero.
- These are bounded implementation-parity checks, not new visual-quality, full-length-video, FPS or minimum-VRAM certification.
- Private test media, paths, frame identifiers and raw outputs are excluded from the publication summary.

[Version 0.2 machine-readable summary](../release_verification_v0_2.json). H3 remains an externally obtained asset; adapter availability is separate from successful code validation.
