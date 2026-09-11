# External assets and release availability

This release contains source code and tests. It does not supply a model service, executable model bundle, or downloadable adapted checkpoint.

| Asset | Expected identity | Availability |
|---|---|---|
| Accepted balanced32 tail checkpoint | SHA256 `c6c90106b7206d27878d8099a71b99f3971eff38d4e235a49ab816813bb802bc` | **Not released.** H3-output adaptation conditions and training-source rights need clarification. |
| Kijai H3 INT8 ConvRot | SHA256 `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410` | External, under MiniMax H3 terms; origin in `configs/components.int8.lock.json`. |
| YOLO11m-face | SHA256 `6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661` | External; origin and separate licensing sources in the component lock. |
| NAFNet-GoPro width32 | SHA256 `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c` | External; origin and file identities in `configs/nafnet_gopro32.provenance.json`. |
| Comfy Kitchen 0.2.33 runtime | Per-file SHA256 in `configs/convrot_cuda_runtime.lock.json` | External. Recorded extension is Windows-specific. |
| NVIDIA cuBLAS 13.0.2.14 | Wheel SHA256 `5f7a90456b1392db60e28719547ba6e00696bfa81304b6887cf287f9edf2c1d7` | External; wheel origin and extracted-file hashes in `configs/cublas13_runtime.lock.json`. |
| Training/evaluation media and derived caches | Not distributed | Research admission did not authorize public redistribution. |

Expected repository-relative layout for an independently authorized local research setup:

```text
models/
  minimax_h3_video_vae_int8_convrot.safetensors
  yolov11m-face.pt
  nafnet_gopro32_baseline/NAFNet-GoPro-width32.pth
third_party/
  vosr2_runtime/comfy_kitchen/...
  h3_cublas13_runtime/nvidia/cu13/...
```

The inherited `vosr2_runtime` directory name describes where the research environment stored Comfy Kitchen. It does not mean VOSR2 code or model weights are used by the accepted FlashH3VR inference path.

No script in the quick start downloads these assets. Review original terms before independently acquiring or using them. Hashes identify bytes; they do not grant rights. Do not replace missing learned weights with random parameters and describe the result as the accepted model.

The unadapted NAF model alone does not reproduce the accepted output: the inference wrapper subtracts the official tail contribution from the adapted tail, so an unchanged tail yields zero correction.
