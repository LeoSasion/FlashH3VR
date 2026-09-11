"""Bounded mixed-target experiment helpers; separate from overfit acceptance."""
from collections import Counter
import json
from pathlib import Path

import torch

from h3ce.cache.keys import file_sha256
from h3ce.train.checkpoint import capture_rng_state, restore_rng_state
from h3ce.train.engine import implementation_hashes, gradient_evidence
from h3ce.train.precision import checked_scaler_step, unscale_and_check
from h3ce.train.preflight import require

ROOT = Path(__file__).resolve().parents[1]
PHASE = "mixed_probe"
STEPS = 76
EVAL_STEPS = (0, 38, 76)
ARMS = ("control", "mixed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def code_hashes():
    result = implementation_hashes(ROOT)
    for folder in ("h3ce/data", "h3ce/cache", "h3ce/vae"):
        result.update({str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p)
                       for p in sorted((ROOT / folder).glob("*.py"))})
    for name in ("gpt_clarity_probe_common.py", "run_gpt_clarity_probe.py", "evaluate_gpt_clarity_probe.py",
                 "plot_training_losses.py", "diagnose_detail_frequency.py", "evaluate_bootstrap_checkpoint.py"):
        result["scripts/" + name] = file_sha256(ROOT / "scripts" / name)
    return result


def check_protocol(protocol):
    require(protocol["status"] == "declared_before_training" and protocol["steps"] == STEPS
            and protocol["evaluation_steps"] == list(EVAL_STEPS) and protocol["maximum_optimizer_updates"] == 152,
            "Unexpected bounded experiment scope")
    require(protocol["code_sha256"] == code_hashes(), "Experiment implementation changed after declaration")
    for item in protocol["evidence"]:
        require(file_sha256(item["path"]) == item["sha256"], "Bound evidence changed: " + item["path"])


def make_slots(dataset):
    """Control repeats the exact parent view for three slots, without forging IDs."""
    result = []
    for i, view in enumerate(dataset.views):
        variant = dataset.variants[view["variant_id"]]
        target = dataset.targets[view["target_id"]]
        group = "ai_paired" if target["target_kind"] == "aligned_pair" else "original_clean" if variant["clean_pair"] else "original_degraded"
        parent = i
        if group == "ai_paired":
            matches = [j for j, v in enumerate(dataset.views) if v["variant_id"] == variant["original_variant_id"]
                       and all(v[k] == view[k] for k in ("crop_xyxy", "bucket_hw", "mode", "crop_to_original"))]
            require(len(matches) == 1, "AI view requires one geometrically identical original parent")
            parent = matches[0]
            other = dataset.views[parent]
            for k in ("x_crop_path", "scene_x_path", "pad_valid_map"):
                require(file_sha256(other[k]) == file_sha256(view[k]), "Control and mixed input differ")
            require(other["z_input_key"] == view["z_input_key"], "Control and mixed latent input differ")
        result.append({"slot": i, "slot_group": group, "control": parent, "mixed": i,
                       "control_view_id": dataset.views[parent]["view_id"], "mixed_view_id": view["view_id"]})
    return result


def update_window(model, optimizer, scheduler, scaler, sampler, get_terms, accumulation, clip, transaction):
    """One real update or overflow rollback; caller persists completed boundaries.

    get_terms returns differentiable terms and serializable consumed-sample info.
    An optimizer failure is marked unsafe to save because it may partially mutate.
    """
    rng = capture_rng_state()
    sampler.begin_window()
    optimizer.zero_grad(set_to_none=True)
    transaction["in_optimizer"] = False
    samples, metrics = [], {}
    try:
        for _ in range(accumulation):
            index = sampler.next_index()
            terms, info = get_terms(index)
            require(all(torch.isfinite(v).all() for v in terms.values()), "Nonfinite application loss")
            scalars = {k: float(v.detach()) for k, v in terms.items()}
            samples.append({**info, "slot": index, "losses": scalars})
            scaler.scale(terms["total"] / accumulation).backward()
            for k, v in scalars.items():
                metrics[k] = metrics.get(k, 0.) + v / accumulation
            del terms
        finite = unscale_and_check(model, optimizer, scaler)
        if not finite:
            transaction["in_optimizer"] = True
            scale = checked_scaler_step(optimizer, scaler, expected_update=False)
            transaction["in_optimizer"] = False
            sampler.rollback_window()
            restore_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            return {"optimizer_updated": False, **scale, "attempted_samples": samples}
        grads = gradient_evidence(model)
        require(any(grads.values()) or all(s.get("clean_pair") is True for s in samples),
                "A non-clean window has no restoration gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True))
        transaction["in_optimizer"] = True
        scale = checked_scaler_step(optimizer, scaler, expected_update=True)
        require(all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite optimizer result")
        scheduler.step()
        sampler.commit_window()
        transaction["in_optimizer"] = False
        optimizer.zero_grad(set_to_none=True)
        return {"losses": metrics, "samples": samples, "gradient_norms": grads,
                "gradient_norm_before_clip": norm, **scale}
    except BaseException:
        if not transaction["in_optimizer"]:
            if sampler.pending_window:
                sampler.rollback_window()
                restore_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
        raise
