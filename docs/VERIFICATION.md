# Version 0.3.2 verification

- **393 CPU tests and 19 subtests passed**, with one external-model test
  deselected. The loader accepts only the registered 1837, 3139 and 4036
  filenames with their exact hashes; the downloader keeps 1837 as default and
  stores optional bundles separately.
- The 4036 adapter's four FP32 tensors are bitwise equal to the frozen Adam4036
  research endpoint. Its complete seven-member bundle and the two older real
  bundles passed member checksums and offline verification without rewriting
  existing files. No full Adam checkpoint, H3 base or private media is included.
- The 0.3.2 wheel passed ZIP integrity; all 146 packaged Python files matched
  source. Isolated imports and CLI help passed, and all three real pinned Dense
  models loaded through the extracted wheel on CPU with the correct Adam step.
- The actual public CLI restored 44/44 frames at 768×432, using 22/22/10 real
  frame windows. The shared API restored 2/2 and 6/6 frames with five- and
  22-frame internal H3 contexts. Independent decode confirmed rational PTS,
  complete frame counts and the original canvas; no frames were skipped.
- Eleven fixed full-size source/4036 panels were actually reviewed. In those
  frames, there was no obvious new severe grid, ghosting, facial structural
  damage or rectangular paste seam. Skin, lips, brows and fine hair remained
  soft; a broad detail gain was not established.
- Bounded 1× Chromium playback reached the end for the same 44-frame source,
  1837 and 4036 clips, plus 4036's 2/6-frame prefixes. Sampled live views
  showed no obvious new severe artifacts, but this is not exhaustive moving-
  frame inspection or temporal-stability certification.
- The fixture is one reused private 44-frame excerpt: 52 input-frame executions
  with only 44 unique source frames. No new training was run for publication.
  These checks do not certify other identities, long-film streaming, FPS or
  16GB GPU operation. Research improvements and regressions are documented in
  [DENSE4036.md](DENSE4036.md).

[Current machine-readable summary](../release_verification.json). The 0.3.1
summary is preserved in [release_verification_v0_3_1.json](../release_verification_v0_3_1.json).

# Version 0.3.1 verification

- **391 CPU tests and 19 subtests passed**, with one external-model test deselected.
  New checks cover the two pinned Dense identities, rejecting one model renamed
  as the other, preserving default documents during optional downloads, and
  reporting the selected model's identity in the full-video sidecar.
- All four exported FP32 tensors are bitwise equal to the frozen Adam3139
  endpoint. The adopted1837 baseline, asset hash and default download are
  unchanged. Both real bundles passed complete member/checksum and offline
  verification. The optional model is documented in [DENSE3139.md](DENSE3139.md).
- The final wheel passed ZIP integrity,146 packaged Python files matched source,
  and isolated Python imported its CLI/full-video modules from the extracted
  wheel. Both real1837/3139 assets loaded through that wheel's CPU loader with
  their correct hashes and model step in the contract.
- The actual public CLI restored all44 frames at768 ×432 / head448, with three
  windows22/22/10. The same process then checked the two- and six-frame prefixes
  through the public API. Independent CPU decoding preserved complete frame
  counts, rational source PTS and canvas dimensions. The short H3 contexts were
  five and22 frames, while the outputs contained only two and six real frames.
- Eleven fixed native-size input/3139 panels were actually inspected: no obvious
  new severe grid, ghosting, facial damage or rectangular paste seam appeared in
  those views. Faces, lip texture and fine hair remained softer than source.
- Chromium actually played3139's44-frame output, its two/six-frame prefixes and
  the same44-frame input/1837 comparison at1× through the end. Sampled normal-size
  live views showed no obvious new severe artifacts. This is bounded automated
  playback with screenshots, not exhaustive human inspection of every moving
  frame or temporal-stability certification.
- The fixture is one reused private44-frame excerpt,52 input-frame executions
  and44 unique source frames. It does not add an independent test family or
  certify long-video quality, FPS or16GB. No new training ran for publication.

[Current machine-readable summary](../release_verification.json). The historical
0.3.0 summary is retained in [release_verification_v0_3.json](../release_verification_v0_3.json).

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
