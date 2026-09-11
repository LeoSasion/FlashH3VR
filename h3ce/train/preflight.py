"""Real cached-image/decoder forward inspection, with optimization prohibited."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from h3ce.cache.keys import canonical_json, file_sha256
from h3ce.cache.store import CacheStore, assert_no_links, atomic_write
from h3ce.errors import H3CEError
from .guard import no_training_guard
from .pipeline import move_sample, refine_latent, restore_pixels


def require(condition, message, code="E_TRAINING_CONTRACT"):
    if not condition:
        raise H3CEError(code, message)


def resolve_manifest(config, root, manifest, phase):
    if manifest:
        path = Path(manifest).resolve()
    else:
        pointer = root / "logs/image_training_preparation_latest.json"
        require(pointer.is_file(), "No materialized image TrainingView manifest. Run image preparation first.", "E_TRAINING_CACHE_REQUIRED")
        data = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(data["overfit_manifest" if phase == "overfit" else "training_manifest"]).resolve()
    assert_no_links(path)
    require(path.is_relative_to((root/config.paths.runs).resolve()) and path.is_file(),
            "Training manifest must be an existing file within project runs")
    return path


def open_dataset(config, root, manifest):
    from .data import CachedImageDataset
    original = config.__class__.model_validate(__import__("yaml").safe_load((manifest.parent/"resolved.yaml").read_text(encoding="utf-8")))
    require(config.data.model_dump() == original.data.model_dump()
            and config.paths.raw == original.paths.raw and config.paths.tmp == original.paths.tmp,
            "Prepared data settings differ; prepare the requested data before training")
    store = CacheStore(root/config.paths.tmp, create=False,
        protected=[root/getattr(config.paths, key) for key in ("raw", "runs", "models", "exports")])
    return CachedImageDataset(manifest, store, config=config, raw_root=root/config.paths.raw)


def execution_contract():
    return {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def require_image_bootstrap(config):
    require(config.native_temporal.mode == "frozen" and config.native_temporal.decoder_adapter is None,
            "Image bootstrap requires a fully frozen native H3 VAE", "E_FROZEN_VAE_REQUIRED")
    require(config.training.microbatch == 1, "This image trainer implements microbatch=1; other values are not silently ignored", "E_NOT_IMPLEMENTED")
    require(not config.lora.routing.adapters, "Bootstrap must learn R and scene without character adapters")
    require(torch.cuda.is_available(), "Locked H3 image execution requires CUDA; no automatic precision/device fallback", "E_CUDA_REQUIRED")


def load_bridge(config, root, dataset):
    from h3ce.vae.factory import load_backend
    from h3ce.vae.bridge import H3VAEBridge
    require(dataset.execution_contract == execution_contract(),
            "Cached encoder execution differs from this runtime; rebind latents explicitly", "E_LATENT_EXECUTION_CONTRACT")
    backend = load_backend(config, root, device="cuda")
    bridge = H3VAEBridge(backend)
    require(bridge.encoder_id == dataset.encoder_contract_id, "Cached latent encoder differs from locked bridge")
    require(not any(p.requires_grad for p in backend.model.parameters()), "Every H3 parameter must remain frozen")
    return bridge


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preview(path, sample, prediction, *, output_label="OUTPUT (untrained check)"):
    canvas = Image.new("RGB", (960, 296), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    for col, (label, tensor) in enumerate((("INPUT X", sample["x"]), ("TARGET Y", sample["y"]), (output_label, prediction))):
        array = tensor[0, :, 0].detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        im = Image.fromarray(np.floor(array*255+.5).astype(np.uint8))
        im = ImageOps.contain(im, (312, 256))
        canvas.paste(im, (col*320+(320-im.width)//2, 32+(256-im.height)//2))
        draw.text((col*320+8, 8), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def check_only(config, root, run, manifest, phase="overfit"):
    from h3ce.model import SpatialRefinerV2
    from .perceptual import application_loss
    from .checkpoint import TrainingBudget
    from .engine import implementation_hashes
    started = time.monotonic()
    report = {"status": "incomplete", "scope": "real_H3_and_spatial_network_forward_only",
        "training_started": False, "optimizer_steps": 0, "m2_training_acceptance": "not_run",
        "trained_base": False, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "implementation_sha256": implementation_hashes(root),
        "components_lock_sha256": file_sha256(root/config.paths.components_lock)}
    with no_training_guard() as guard, TrainingBudget(root/config.paths.runs, config.project.budget_seconds, phase="pretraining_check") as budget:
        require_image_bootstrap(config)
        dataset = open_dataset(config, root, manifest)
        report["data_audit"] = dataset.audit(budget_check=budget.check)
        budget.check()
        seed_model(config.project.seed)
        torch.cuda.reset_peak_memory_stats()
        bridge = load_bridge(config, root, dataset)
        budget.check()
        model = SpatialRefinerV2.from_config(config.model).to("cuda").eval()
        versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        vae_versions = {name: (id(p), p._version) for name, p in bridge.backend.model.named_parameters()}
        loss = application_loss(config, root)
        selected = []
        for mode in ("fullbody", "face"):
            selected.extend(next(([i] for i, v in enumerate(dataset.views) if v["mode"] == mode), []))
        require(selected, "No training views are available")
        cases = []
        for i in selected:
            budget.check()
            sample = move_sample(dataset[i], "cuda")
            zp, delta = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
            prediction, pack = restore_pixels(bridge, sample, zp, grad=False, strength=config.model.output.strength)
            terms = loss(prediction, sample["y"], zp, sample["z_target"], sample["valid"],
                         person_mask=sample["person_mask"], face_mask=sample["face_mask"])
            require(torch.count_nonzero(delta).item() == 0 and torch.equal(prediction, sample["x"]),
                    "Zero-initialized R must produce exactly Xbase through both real H3 decode branches")
            require(all(torch.isfinite(value).all().item() for value in terms.values()), "Forward loss is nonfinite")
            output = run/"previews"/f"check_{sample['mode']}.png"
            preview(output, sample, prediction)
            cases.append({"mode": sample["mode"], "view_id": sample["view_id"], "bucket_hw": list(sample["bucket_hw"]),
                "latent_shape": list(zp.shape), "zero_delta": True, "output_equals_input_exactly": True,
                "losses_at_initialization": {k: float(v) for k, v in terms.items()}, "preview": str(output),
                "effective_decoder_hash": pack.effective_decoder_hash})
        require(versions == {n: (id(p), p._version) for n, p in model.named_parameters()}, "R weights changed in check-only")
        require(vae_versions == {n: (id(p), p._version) for n, p in bridge.backend.model.named_parameters()}, "VAE weights changed")
        require(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in bridge.backend.model.parameters()), "Check-only created gradients")
        torch.cuda.synchronize()
        report.update(status="passed_forward_only", cases=cases, execution_guard=dict(guard),
            refiner_parameters=sum(p.numel() for p in model.parameters()), vae_parameters_frozen=True,
            model_parameters_unchanged=True, encoder_contract_id=bridge.encoder_id,
            runtime=execution_contract(), refiner_autocast="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started, cumulative_budget_used_seconds=budget.used,
            limitations=["No backward, optimization, overfit, trained base or training-memory acceptance",
                         "Same-source images do not provide independent quality validation"])
        atomic_write(run/"check_only_report.json", canonical_json(report))
    return {"status": report["status"], "run": str(run), "report": str(run/"check_only_report.json"),
            "training_started": False, "optimizer_steps": 0}
