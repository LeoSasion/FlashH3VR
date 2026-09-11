"""Fixed 19-view measurements; called only between committed mixed-probe steps."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from h3ce.cache.keys import canonical_json, file_sha256
from h3ce.cache.store import atomic_write
from h3ce.train.checkpoint import capture_rng_state, restore_rng_state
from h3ce.train.guard import no_training_guard
from h3ce.train.pipeline import move_sample, refine_latent, restore_pixels
from h3ce.train.preflight import require
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.evaluate_bootstrap_checkpoint import measure, aggregate


def fixed_roi(sample, size=128):
    valid = sample["valid"][0, 0, 0].numpy()
    face = sample["face_mask"][0, 0, 0].numpy()
    yy, xx = np.nonzero(valid)
    fy, fx = np.nonzero(face * valid)
    require(len(fx) > 0 and yy.max() - yy.min() + 1 >= size and xx.max() - xx.min() + 1 >= size,
            "Fixed face ROI needs detected face and sufficient valid canvas")
    cx, cy = (fx.min() + fx.max() + 1) / 2, fy.min() + .35 * (fy.max() - fy.min() + 1)
    x = int(np.clip(round(cx - size / 2), xx.min(), xx.max() + 1 - size))
    y = int(np.clip(round(cy - size / 2), yy.min(), yy.max() + 1 - size))
    require(valid[y:y+size, x:x+size].all(), "ROI crosses padding")
    return [x, y, x+size, y+size]


def edge_mse(pred, target):
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    return float(sum(np.mean((np.diff(pred, axis=a) - np.diff(target, axis=a)) ** 2) for a in (0, 1)) / 2)


def evaluate(model, bridge, dataset, loss, config, budget, run, step, checkpoint, protocol):
    output = run / f"fixed_step_{step:04d}"
    require(not output.exists(), "Existing fixed evaluation must not be overwritten")
    output.mkdir()
    rng = capture_rng_state()
    train_mode = model.training
    versions = [{n: (id(v), v._version) for n, v in m.named_parameters()} for m in (model, bridge.backend.model)]
    cases = []
    try:
        with no_training_guard() as guard:
            model.eval()
            for i, view in enumerate(dataset.views):
                budget.check()
                sample = move_sample(dataset[i], "cuda")
                zp, _ = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
                pred, _ = restore_pixels(bridge, sample, zp, grad=False, strength=config.model.output.strength)
                if step == 0:
                    require(torch.equal(pred, sample["x"]), "Fresh model is not exact identity")
                current = measure(sample, pred, zp, loss)
                baseline = measure(sample, sample["x"], sample["z_input"], loss)
                x0, y0, x1, y1 = protocol["rois"][view["view_id"]]
                require(sample["valid"][0, 0, 0, y0:y1, x0:x1].bool().all(), "Frozen ROI crosses padding")
                def arr(t):
                    return t[0, :, 0].detach().float().cpu().permute(1, 2, 0).numpy()[y0:y1, x0:x1].copy()
                arrays = {"input": arr(sample["x"]), "target": arr(sample["y"]), "model": arr(pred)}
                path = output / f"case_{i:02d}.npz"
                np.savez_compressed(path, **arrays)
                cases.append({"asset_id": sample["asset_id"], "view_id": view["view_id"], "group": sample["supervision_group"],
                    "source": dataset.sources[sample["asset_id"]]["path"], "mode": sample["mode"], "roi_xyxy": [x0,y0,x1,y1],
                    "current": current, "input": baseline, "float_path": str(path), "float_sha256": file_sha256(path),
                    "bands": frequency_metrics(arrays["input"], arrays["target"], arrays["model"]),
                    "edge_mse": {k: edge_mse(arrays[k], arrays["target"]) for k in ("input", "model")}})
                print(canonical_json({"event": "fixed_case_evaluated", "step": step, "case": i, "group": sample["supervision_group"]}).decode(), flush=True)
            require(versions == [{n: (id(v), v._version) for n, v in m.named_parameters()} for m in (model, bridge.backend.model)], "Evaluation changed parameters")
            require(not any(v.grad is not None for m in (model, bridge.backend.model) for v in m.parameters()), "Evaluation accumulated gradients")
            groups = {g: {label: aggregate([c for c in cases if c["group"] == g], label) for label in ("current", "input")}
                      for g in ("original_degraded", "original_clean", "ai_paired")}
            result = {"status": "completed_fixed_mixed_probe_evaluation", "step": step, "run": str(run),
                "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint), "cases": cases, "groups": groups,
                "additional_optimizer_steps": 0, "execution_guard": dict(guard), "parameters_unchanged": True,
                "independent_validation": False, "budget": budget.snapshot()}
            atomic_write(output / "metrics.json", canonical_json(result))
    finally:
        model.train(train_mode)
        restore_rng_state(rng)
    return result
