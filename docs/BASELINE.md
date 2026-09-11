# Accepted baseline and closure observations

Snapshot: 2026-09-12. These are historical project measurements, not a new benchmark performed for the public release. Raw video, arrays, checkpoint and private host process samples are not included. Published hashes identify the internal artifacts; hashes alone do not enable independent reproduction without those artifacts.

## Identity

- Baseline ID: `naf-balanced32-direct-bgr-20260911`.
- Training run: `multihead-balanced-mae32-20260910-v1`, final step 32.
- Checkpoint SHA256: `c6c90106b7206d27878d8099a71b99f3971eff38d4e235a49ab816813bb802bc`.
- Adapted component: 199,747 parameters in the last three existing NAF decoder levels and output layer. No newly added temporal module.
- Accepted execution: dual16, direct BGR preparation, input optimization/overlap and execution reuse.

## Speed and output agreement

The source clip contains 60 frames at 60 fps, 768 × 432. Timing starts with input-file opening after model loading and ends after output-file close. It includes decode, cut analysis, input detection, head geometry, H3 and NAF forward passes, paste-back, color conversion, transfer and H.264 encoding. Preflight, model loading and later evidence saving are excluded.

| Measurement | Historical accepted baseline | Closure with monitoring |
|---|---:|---:|
| Date | 2026-09-11 | 2026-09-12 |
| Full passes | 1 | 1 |
| Processing seconds | 2.4402117 | 2.5396372 |
| Frames per processing second | 24.5880306 | 23.6254218 |
| Loading seconds | 4.9530436 | 6.3140686 |

GPU: NVIDIA RTX PRO 6000 Blackwell, 96 GB class. Recorded environment: Windows, Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0+cu128, PyAV 18.1.0. Driver during closure: 591.86.

The closure time was 4.07% greater than the historical baseline. A single measurement does not establish a stable regression or its cause; the historical baseline remains accepted.

Closure comparison covered 11 floating-point arrays, detections/geometry, all 60 decoded YUV frames and the encoded MP4, with exact identity. MP4 SHA256: `93cf457a3762aa81c88bfdf1cdfdc7fe08e03c295accb3bf0a15dd83b22d791d`.

Independent CPU geometry/color/PTS checks passed. There was no new continuous-playback human review, LPIPS evaluation, temporal-metric evaluation, gradient calculation or training update during closure. Output identity carries the historical visual result; it is not a new human-review claim.

## Historical quality

Johnny evaluation: one degraded 180-frame pass and one clean 180-frame pass with the accepted checkpoint.

| Metric | Accepted model | Interpretation |
|---|---:|---|
| LPIPS | 0.2282778112 | 5.847786% lower than degraded input |
| High-frequency error MSE | 0.0004007139 | 1.210707% lower than degraded input |
| Clean ROI RGB MAE | 0.0002870821 | Clean-input alteration under the original ROI protocol |

The improvement percentages are relative to the degraded input in that evaluation, not relative to another restoration model. The corpus and source diversity are limited. Temporal optimization and the previous numerical temporal threshold were retired by the project owner after accepting visual stability; the historical metric did not meet the old threshold. No temporal improvement is claimed.

## CPU/GPU observation

The closure sampler recorded 325 samples over the whole process, with a median interval of about 100.36 ms. Statistics below intersect the inference interval; 27 sampling intervals overlapped it.

| Resource | Time-weighted mean | Sampled peak |
|---|---:|---:|
| Whole-machine CPU | 13.01% | 43.10% |
| Test process tree, one core = 100% | 231.25% (about 2.31 cores) | See sampling limitations |
| GPU device compute utilization | 52.21% | 99% |
| GPU memory-controller utilization | 27.14% | 48% |
| Device GPU memory, including background | — | 13.2228 GiB |
| Framework allocated GPU memory | — | 6.1899 GiB |
| Framework reserved GPU memory | — | 9.7852 GiB |
| Inference process-tree RSS | — | 3.0556 GiB |

The host had 24 logical CPUs. Whole-machine CPU and device GPU readings include background activity. Framework reserved and allocated memory overlap and must not be added. Summed process RSS can include shared pages. Per-process GPU memory was unavailable from the Windows driver and is not reported as zero.

NVML utilization has driver-defined measurement windows; 100 ms polling does not make each short stage independently measurable. The sampler consumed about 3.56 CPU seconds over 32.43 seconds. Its presence cannot be assumed to explain the full timing difference. One final process-disappearance record occurred after exit; no sampler error occurred in inference.

## Public-copy changes

The publication changes two asset-location lookups to use repository-relative manifests. Numerical inference source is otherwise preserved. These packaging changes have CPU verification only; the historical GPU measurements predate them. See `publication_origins.json` for original source hashes and the listed modifications.
