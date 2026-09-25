# FlashH3VR Dense Inter 1837

## Identity and use

A single-person head-restoration research adapter trained for Go Youn-jung (고윤정). It predicts a residual in the frozen MiniMax H3 video VAE latent space. It is not an identity-recognition model, a full-frame video application, a general face-restoration benchmark winner, or an independently trained replacement for H3.

- Four float32 tensors, 3,152,128 parameters; 24-channel 16 × 16 latent tiles, a 256-unit Dense bottleneck and GELU.
- Actual training buckets: 256, 448, 640 and 832; native H3 256-pixel tiles with at least 64 pixels of overlap.
- Research lineage: 866-step source plus 971 updates, yielding full Adam step 1837.
- Safetensors SHA256: a210d161a00f7089122495c5176118303fd7d6efe9ff1d2d4d112a0c6753b804.
- External H3 checkpoint SHA256: 9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410, executed through dequantized FP16.
- [Inference interface](docs/USAGE.md), [download and asset identities](docs/ASSETS.md), [metrics](docs/BASELINE.md).

## Training data

The final stage used 75 photographs and 146 real video windows, with 19 canonical video sources and 3,207 distinct source-video frames. They are not 146 independent productions: 119 new windows share one ELLE master. All 221 underlying samples were exposed at least four times; 570 of 608 size/input conditions received optimizer updates.

Targets were clear native-size master crops, kept or downsampled. Inputs were independently downsampled from those same clean sources. No upscaled source, generated super-resolution image or sharpened model output was accepted as a new HQ target.

Publicly accessible photographs/video do not by themselves establish redistribution permission. No training media, extracted frames, output portraits or source-download list is distributed here. Dataset intake and technical quality checks are not a blanket license clearance. Publishing adapter parameters and redistributing the underlying media are different acts; this card makes no general legal claim that either automatically authorizes or forbids the other.

## Evaluation and known limitations

Against the same study's 866-step initialization, train-partition edge error decreased 8.080%, and the three repeatedly used development windows decreased 4.098%. The 448 photograph mouth-edge metric regressed 2.217%–2.233%. Thirty fixed panels showed small visual differences and no obvious new severe artifacts in the reviewed set.

The validation windows were reused; source-event independence and broad generalization are not established. The 832 bucket contains only three windows from two sources. There was no equal-time control, new full-length playback test, inference FPS measurement or 16GB deployment certification. Use of a celebrity-specific adapter on other people is outside its evaluated scope and may alter facial detail. Generated detail is not evidence of the person's true appearance.

## Terms and contents

Project-owned source: AGPL-3.0-only, with third-party code notices retained. The H3-dependent adapter is distributed subject to the MiniMax H3 Community License and downstream conditions, separately from the source license. See [licensing](THIRD_PARTY_NOTICES.md).

The adapter format contains only the four learned Dense tensors. It excludes H3 weights, optimizer/scaler state, RNG, training paths, private caches, media, face detections and other model weights. The public inference format uses safetensors; no executable pickle is needed.

This project is independent and does not imply authorization or endorsement by the depicted person, source publishers or MiniMax.
