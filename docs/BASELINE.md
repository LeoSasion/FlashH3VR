# Current Dense Inter baseline

Selected September 26, 2026: **h3-dense-multires-1837-20260926**.

The research checkpoint SHA256 is 222de15af77cdc9327068ccc9a3f46b3f9bac78b6d63e5cf88da522150717cd1. It contains optimizer/research state and is not the public inference format. The four-tensor safetensors export SHA256 is **a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804**. Every exported tensor is bitwise equal to the adopted checkpoint.

## Training

- Frozen H3 encoder and decoder; only the 3,152,128-parameter Dense Inter was updated.
- 971 new updates, 6,300.325 effective seconds; full optimizer step 1837.
- Bucket updates: 256 × 532, 448 × 233, 640 × 174, 832 × 32. The 105 minutes are shared across buckets.
- 75 photographs and 146 real video windows from 19 canonical video sources; 3,207 unique video source frames.
- All 221 underlying samples received at least four paired half-size/clean exposures. 570/608 admitted training conditions entered the optimizer; the remaining 38 are size variants exposed at other sizes.
- 119 newly added windows share one ELLE master. Window count is not independent-source count. The 832 bucket has only three windows from two video sources and no independent 832 validation.
- HQ targets came directly from clear native-size master crops, kept or downsampled. Half-size inputs were derived from the same clean source; canvas alignment did not add native detail.

## Endpoint comparison

Comparator: the 866-step initialization, SHA256 8a5eb35fa0ce61c7b4ff5bb1edbf7c0ef45f921ca57acd6070e459d24bd3f201. Both models received identical inputs. Positive improvement means lower error.

| Partition | RGB L1 improvement | Edge L1 improvement |
|---|---:|---:|
| Frozen training partition | 19.194% | 8.080% |
| Reused development windows | 11.598% | 4.098% |

Real frames were averaged into windows, then into equal-weight canonical sources within each populated bucket/condition/media stratum; populated strata were weighted equally. The training summary includes the 38 unexposed same-source size variants, not independent validation. Development uses three repeatedly evaluated windows at 448/640 in half-size/clean conditions.

**448 photographs regress:** RGB error rises 1.885%–2.520%, edge error 0.651%–0.840%, and mouth-edge error 2.217%–2.233%. The owner accepted the endpoint as the next research baseline with these regressions recorded.

The calculation was separately recomputed on CPU from 1,240 raw arrays and 15,604 frame references. Thirty fixed output panels and two loss charts were actually reviewed: visual changes were small, with no obvious new severe grid, seam, ghosting or facial-structure damage in the reviewed set. Private media/arrays are not published; this is a disclosed internal evaluation, not a reproducible public benchmark.

## Memory and limitations

Training plus fixed-anchor framework peaks were 12.047 GiB allocated and 13.324 GiB reserved on the research workstation. These overlapping counters must not be added; they are not an inference memory requirement or 16GB-device certification.

There was no equal-time control arm, so longer training, more data and multiresolution effects were not separated. The endpoint has no new ordinary-playback, full-length-video, protected-test or FPS certification. Internal numerical gains do not establish broad perceptual superiority.

The [version 0.1 NAF baseline](HISTORICAL_0_1_BASELINE.md) is a different model and execution path; its performance labels remain historical.
