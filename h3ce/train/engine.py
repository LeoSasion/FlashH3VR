"""Explicit image bootstrap phases. This module never trains on import/check-only."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.errors import H3CEError
from .checkpoint import CheckpointManager, TrainingBudget, capture_rng_state, restore_rng_state
from .pipeline import move_sample, refine_latent, restore_pixels
from .overfit_data import overfit_probe_passed, summarize_probe, validate_overfit_dataset
from .preflight import (execution_contract, load_bridge, open_dataset, preview, require,
                        require_image_bootstrap, seed_model)
from .sampler import StatefulSampler
from .precision import POLICY, make_scaler, unscale_and_check, checked_scaler_step


def implementation_hashes(root):
    files = [root/"h3ce/cli.py", root/"h3ce/config.py", root/"h3ce/doctor.py", root/"h3ce/components.py"]
    for directory in ("h3ce/model", "h3ce/train"):
        files += sorted((root/directory).glob("*.py"))
    return {str(path.relative_to(root)).replace("\\", "/"): file_sha256(path) for path in files}


def make_contract(config, root, dataset, phase, max_steps):
    return {"schema_version": 1, "phase": phase, "max_steps": max_steps,
        "gradient_precision_policy": POLICY,
        "manifest_sha256": dataset.manifest_sha256,
        "resolved_sha256": digest(config.model_dump(mode="json")),
        "compatibility": {"architecture": config.model.model_dump(mode="json"),
            "encoder_contract_id": dataset.encoder_contract_id,
            "components_sha256": file_sha256(root/config.paths.components_lock),
            "data": config.data.model_dump(mode="json"),
            "implementation_sha256": implementation_hashes(root), "runtime": execution_contract()}}


def read_run_report(root, config, directory):
    path = Path(directory).resolve()
    assert_no_links(path)
    require(path.is_relative_to((root/config.paths.runs).resolve()), "Prior phase must be inside project runs")
    try:
        report = json.loads((path/"training_report.json").read_text(encoding="utf-8"))
        contract = json.loads((path/"training_contract.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise H3CEError("E_OVERFIT_REQUIRED", "Prior training report and contract are missing or unreadable") from exc
    manager = CheckpointManager(root/config.paths.runs, path, contract=contract)
    checkpoint = Path(report["checkpoint"])
    require(checkpoint.resolve().is_relative_to(path), "Prior report checkpoint must belong to that run")
    payload = manager.read(checkpoint)
    require(payload["step"] == report["optimizer_steps"] and payload["stage"] == report["phase"],
            "Prior report disagrees with its checkpoint")
    require(payload["step"] == payload["contract"]["max_steps"] and payload["extra"].get("phase_complete") is True,
            "Prior checkpoint has not completed its phase", "E_INCOMPLETE_PHASE")
    payload["_checkpoint_sha256"] = file_sha256(checkpoint)
    return report, payload


def require_overfit(root, config, directory, contract, dataset):
    require(directory is not None, "Run the 8–16 pair overfit probe before any long bootstrap phase", "E_OVERFIT_REQUIRED")
    report, payload = read_run_report(root, config, directory)
    require(report["phase"] == "overfit" and report["status"] == "passed_overfit_probe"
            and report["optimizer_steps"] >= 2, "The overfit probe has not passed", "E_OVERFIT_REQUIRED")
    require(payload["contract"]["compatibility"] == contract["compatibility"],
            "Overfit code, data recipe, encoder or components differ from this phase", "E_OVERFIT_REQUIRED")
    # Gate using the checkpoint's checksummed extra evidence, not an editable report alone.
    evidence = payload["extra"].get("overfit_evidence", {})
    require(evidence.get("passed") is True and evidence.get("gradients", {}).get("scene", 0) > 0
            and evidence.get("gradients", {}).get("spatial", 0) > 0 and evidence.get("pixel_gradient_max", 0) > 0,
            "Checkpoint lacks successful overfit and gradient evidence", "E_OVERFIT_REQUIRED")
    require(set(evidence["view_ids"]).issubset({v["view_id"] for v in dataset.views}),
            "Overfit views are not part of the requested training manifest", "E_OVERFIT_REQUIRED")
    return payload


def gradient_evidence(model):
    result = {"spatial": 0., "scene": 0., "output": 0.}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        require(torch.isfinite(parameter.grad).all().item(), f"Nonfinite gradients: {name}", "E_NONFINITE_GRADIENT")
        norm = float(parameter.grad.detach().float().norm())
        category = "scene" if name.startswith("scene_context.") else "spatial" if ".ffn." in name else "output"
        result[category] = max(result[category], norm)
    return result


def terms_for(model, bridge, sample, loss, phase, config, *, grad):
    from .losses import latent_loss
    zp, delta = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
    if phase == "latent":
        latent = latent_loss(zp, sample["z_target"], sample["valid"], epsilon=config.training.losses.charbonnier_epsilon)
        return {"total": latent, "latent": latent}, None
    prediction, _ = restore_pixels(bridge, sample, zp, grad=grad, strength=config.model.output.strength)
    terms = loss(prediction, sample["y"], zp, sample["z_target"], sample["valid"],
                 person_mask=sample["person_mask"], face_mask=sample["face_mask"])
    return terms, prediction


def evaluate_probe(model, bridge, dataset, loss, config, budget, run, label):
    totals = []
    rgb = []
    pair_kinds, asset_ids, view_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            budget.check()
            sample = move_sample(dataset[index], "cuda")
            terms, prediction = terms_for(model, bridge, sample, loss, "overfit", config, grad=False)
            require(all(torch.isfinite(t).all().item() for t in terms.values()), "Nonfinite probe loss", "E_NONFINITE_LOSS")
            totals.append(float(terms["total"]))
            rgb.append(float(terms["rgb"]))
            pair_kinds.append("clean" if sample["clean_pair"] else "degraded")
            asset_ids.append(sample["asset_id"])
            view_ids.append(sample["view_id"])
            if index in (0, len(dataset)-1):
                preview(run/"previews"/f"{label}_{index:02d}.png", sample, prediction, output_label=f"OUTPUT ({label} overfit)")
    model.train()
    return summarize_probe(totals, rgb, pair_kinds, asset_ids, view_ids)


def _train_images(config, root, run, manifest, *, phase, max_steps=None, resume="auto", overfit_run=None, init_run=None):
    """Call only on an explicit training invocation; check-only uses another function."""
    from h3ce.model import SpatialRefinerV2
    from .perceptual import application_loss
    require_image_bootstrap(config)
    default = 8 if phase == "overfit" else getattr(config.training.stages, "bootstrap_"+phase).max_steps
    limit = default if max_steps is None else max_steps
    require(type(limit) is int and limit >= (2 if phase == "overfit" else 1), "Invalid optimizer-step limit")
    if phase != "overfit":
        require(limit <= default, "Step override may reduce but cannot silently exceed the configured stage limit")
    started = time.monotonic()
    with TrainingBudget(root/config.paths.runs, config.project.budget_seconds) as budget:
        dataset = open_dataset(config, root, manifest)
        dataset.audit(budget_check=budget.check)
        if phase == "overfit":
            require(init_run is None and overfit_run is None, "The initial overfit probe cannot silently reuse a different training base")
            validate_overfit_dataset(dataset)
        contract = make_contract(config, root, dataset, phase, limit)
        prior = require_overfit(root, config, overfit_run, contract, dataset) if phase != "overfit" else None
        overfit_sha = prior["_checkpoint_sha256"] if prior else None
        if phase == "pixel":
            require(init_run is not None, "Pixel bootstrap must initialize from the completed latent phase", "E_LATENT_STAGE_REQUIRED")
        if init_run:
            initial_report, prior = read_run_report(root, config, init_run)
            expected_phase = "latent" if phase == "pixel" else "overfit"
            require(initial_report["phase"] == expected_phase and initial_report["status"] in {"completed_unvalidated_base", "passed_overfit_probe"},
                    "Initializer has not completed the required previous phase")
            require(prior["contract"]["compatibility"] == contract["compatibility"], "Initializer compatibility differs")
            if phase == "pixel":
                require(prior["contract"]["manifest_sha256"] == dataset.manifest_sha256,
                        "Pixel and latent phases must use the same full training manifest")
            else:
                require(prior["_checkpoint_sha256"] == overfit_sha, "Latent initialization must use the verified overfit checkpoint")
        contract["lineage"] = {"overfit_checkpoint_sha256": overfit_sha,
                               "initializer_checkpoint_sha256": prior["_checkpoint_sha256"] if prior else None}
        atomic_write(run/"training_contract.json", canonical_json(contract))
        seed_model(config.project.seed)
        budget.check()
        bridge = load_bridge(config, root, dataset)
        budget.check()
        model = SpatialRefinerV2.from_config(config.model).to("cuda").train()
        if prior:
            model.load_state_dict(prior["model"], strict=True)
        lr = config.training.stages.bootstrap_pixel.lr if phase in {"overfit", "pixel"} else config.training.stages.bootstrap_latent.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        scaler = make_scaler(phase)
        sampler = StatefulSampler(len(dataset), seed=config.project.seed)
        manager = CheckpointManager(root/config.paths.runs, run, contract=contract)
        checkpoint = manager.find_resume(resume)
        step, extra = 0, {"gradient_maxima": {"spatial": 0., "scene": 0., "output": 0.}, "pixel_gradient_max": 0.,
                         "overflow_attempts": 0, "consecutive_overflows": 0}
        if checkpoint:
            payload = manager.restore(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                      scaler=scaler, sampler=sampler)
            budget.validate_resume_snapshot(payload["budget"])
            step, extra = payload["step"], payload["extra"]
        loss = application_loss(config, root)
        perceptual_versions = {n: (id(p), p._version) for n, p in loss.perceptual.named_parameters()} if loss.perceptual is not None else {}
        vae_versions = {n: (id(p), p._version) for n, p in bridge.backend.model.named_parameters()}

        def save():
            optimizer.zero_grad(set_to_none=True)
            return manager.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                sampler=sampler, stage=phase, step=step, budget=budget.snapshot(),
                resolved_config=config.model_dump(mode="json"), extra=extra)

        checkpoint = save()  # Durable initial/resumed boundary exists before the first graph.
        in_optimizer = False
        window_rng = None
        try:
            if phase == "overfit" and "probe_before" not in extra:
                extra["probe_before"] = evaluate_probe(model, bridge, dataset, loss, config, budget, run, "before")
                checkpoint = save()
            torch.cuda.reset_peak_memory_stats()
            while step < limit:
                budget.check()
                window_rng = capture_rng_state()
                sampler.begin_window()
                optimizer.zero_grad(set_to_none=True)
                metrics = {}
                window_pixel_gradient = 0.
                pixel_gradient_finite = True
                accumulation = config.training.gradient_accumulation
                for _ in range(accumulation):
                    budget.check()
                    sample = move_sample(dataset[sampler.next_index()], "cuda")
                    terms, _ = terms_for(model, bridge, sample, loss, phase, config, grad=True)
                    require(all(torch.isfinite(t).all().item() for t in terms.values()), "Nonfinite loss", "E_NONFINITE_LOSS")
                    if phase == "overfit" and step >= 1:
                        # Isolate the pixel term: latent supervision must not hide a detached decoder path.
                        diagnostic_scale = min(float(scaler.get_scale()), 1024.)
                        pixel_gradient = torch.autograd.grad(terms["rgb"]*diagnostic_scale, model.output_projection.weight,
                                                             retain_graph=True, allow_unused=True)[0]
                        require(pixel_gradient is not None,
                                "RGB loss must reach the refiner through the frozen decoder", "E_GRADIENT_CONTRACT")
                        pixel_gradient = pixel_gradient.float()/diagnostic_scale
                        pixel_gradient_finite &= bool(torch.isfinite(pixel_gradient).all())
                        if pixel_gradient_finite:
                            window_pixel_gradient = max(window_pixel_gradient, float(pixel_gradient.norm()))
                    scaler.scale(terms["total"]/accumulation).backward()
                    for key, value in terms.items():
                        metrics[key] = metrics.get(key, 0.) + float(value.detach())/accumulation
                require(not any(p.grad is not None or p.requires_grad for p in bridge.backend.model.parameters()), "Frozen VAE acquired gradients")
                finite = unscale_and_check(model, optimizer, scaler)
                if not finite:
                    require(scaler.is_enabled(), "Nonfinite gradients without an enabled scaler", "E_NONFINITE_GRADIENT")
                    in_optimizer = True
                    scale_record = checked_scaler_step(optimizer, scaler, expected_update=False)
                    in_optimizer = False
                    sampler.rollback_window()
                    restore_rng_state(window_rng)
                    window_rng = None
                    optimizer.zero_grad(set_to_none=True)
                    extra["overflow_attempts"] += 1
                    extra["consecutive_overflows"] += 1
                    checkpoint = save()  # Updated scaler, unchanged model/optimizer/scheduler/sampler.
                    skipped = {"event":"overflow_retry", "completed_step":step,
                               "attempt":extra["overflow_attempts"], **scale_record, "used_seconds":budget.used}
                    with (run/"skipped_updates.jsonl").open("ab") as stream:
                        stream.write(canonical_json(skipped)+b"\n"); stream.flush()
                    print(json.dumps(skipped), flush=True)
                    require(extra["consecutive_overflows"] < POLICY["max_consecutive_overflows"],
                            "Repeated gradient overflow; inspect the saved checkpoint and scale history", "E_NONFINITE_GRADIENT")
                    continue
                require(pixel_gradient_finite,"Scaled RGB diagnostic gradient is nonfinite", "E_NONFINITE_GRADIENT")
                grads = gradient_evidence(model)
                require(any(grads.values()),
                        "No trainable restoration parameter received a gradient. An all-clean accumulation window at exact identity can legitimately be zero; this strict probe stops without skipping or fabricating an update.",
                        "E_GRADIENT_CONTRACT")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True))
                in_optimizer = True
                scale_record = checked_scaler_step(optimizer, scaler, expected_update=True)
                require(all(torch.isfinite(p).all().item() for p in model.parameters()),
                        "Optimizer produced nonfinite weights; resume the previous durable boundary", "E_NONFINITE_MODEL")
                scheduler.step()
                sampler.commit_window()
                step += 1
                extra["consecutive_overflows"] = 0
                in_optimizer = False
                window_rng = None
                for key, value in grads.items():
                    extra["gradient_maxima"][key] = max(extra["gradient_maxima"][key], value)
                extra["pixel_gradient_max"] = max(extra["pixel_gradient_max"], window_pixel_gradient)
                optimizer.zero_grad(set_to_none=True)
                record = {"step": step, "phase": phase, "losses": metrics, "gradient_norms": grads,
                          "gradient_norm_before_clip":gradient_norm, **scale_record,
                          "used_seconds": budget.used, "learning_rate": scheduler.get_last_lr()[0]}
                with (run/"metrics.jsonl").open("ab") as stream:
                    stream.write(canonical_json(record)+b"\n")
                    stream.flush()
                if step % config.training.preview_every_steps == 0:
                    budget.check()
                    with torch.no_grad():
                        sample = move_sample(dataset[0], "cuda")
                        zp, _ = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
                        prediction, _ = restore_pixels(bridge, sample, zp, grad=False, strength=config.model.output.strength)
                        preview(run/"previews"/f"step_{step:08d}.png", sample, prediction, output_label=f"OUTPUT ({phase} step {step})")
                if step % config.training.checkpoint_every_steps == 0 or step == limit:
                    checkpoint = save()
                print(json.dumps({"event": "optimizer_step", **record}), flush=True)
            require(vae_versions == {n: (id(p), p._version) for n, p in bridge.backend.model.named_parameters()}, "H3 weights changed")
            if loss.perceptual is not None:
                require(perceptual_versions == {n: (id(p), p._version) for n, p in loss.perceptual.named_parameters()}
                        and not any(p.requires_grad or p.grad is not None for p in loss.perceptual.parameters()), "Frozen perceptual weights changed or acquired gradients")
                extra['perceptual_component_sha256'] = loss.perceptual.h3ce_component_sha256
            status = "completed_unvalidated_base"
            if phase == "overfit":
                after = evaluate_probe(model, bridge, dataset, loss, config, budget, run, "after")
                before = extra["probe_before"]
                gradients = extra["gradient_maxima"]
                passed = overfit_probe_passed(step=step, gradients=gradients,
                    pixel_gradient_max=extra["pixel_gradient_max"], before=before, after=after)
                extra["overfit_evidence"] = {"passed": passed, "before": before, "after": after,
                    "gate_pair_kind": "degraded",
                    "gradients": gradients, "pixel_gradient_max": extra["pixel_gradient_max"], "view_ids": [v["view_id"] for v in dataset.views],
                    "scope": "training_subset_numeric_probe_requires_visual_review_before_claiming_useful_base"}
                status = "passed_overfit_probe" if passed else "failed_overfit_probe"
            extra["phase_complete"] = True
            checkpoint = save()
            report = {"status": status, "phase": phase, "optimizer_steps": step, "training_started": True,
                "checkpoint": str(checkpoint), "manifest_sha256": dataset.manifest_sha256,
                "trained_base_accepted": False, "vae_parameters_unchanged": True, "extra": extra,
                "elapsed_seconds": time.monotonic()-started, "budget": budget.snapshot(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "next": "Review fixed previews; independent quality acceptance is still required before character LoRA or export"}
            atomic_write(run/"training_report.json", canonical_json(report))
            return {"status": status, "run": str(run), "report": str(run/"training_report.json"), "optimizer_steps": step}
        except BaseException as exc:
            # A failing optimizer may have partially changed parameters; keep the earlier durable checkpoint.
            if not in_optimizer:
                if sampler.pending_window:
                    sampler.rollback_window()
                    restore_rng_state(window_rng)
                checkpoint = save()
            failure = {"status": "interrupted", "phase": phase, "last_completed_step": step,
                "resume_checkpoint": str(checkpoint), "optimizer_may_have_partially_updated": in_optimizer,
                "error": type(exc).__name__, "message": str(exc), "budget": budget.snapshot(),
                "automatic_fallback": False}
            atomic_write(run/"training_interruption.json", canonical_json(failure))
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise H3CEError("E_CUDA_OOM", "CUDA memory exhausted. The recorded checkpoint can resume under the same contract. Smaller buckets require explicit re-preparation and a new phase run; no silent bucket or precision change.", failure) from exc
            raise


def train_images(config, root, run, manifest, **options):
    """Record initialization failures too, including OOM before a model can be saved."""
    try:
        return _train_images(config, root, run, manifest, **options)
    except BaseException as exc:
        path = run/"training_interruption.json"
        if not path.exists():
            checkpoint = None
            contract_path = run/"training_contract.json"
            if contract_path.is_file():
                try:
                    manager = CheckpointManager(root/config.paths.runs, run,
                        contract=json.loads(contract_path.read_text(encoding="utf-8")))
                    checkpoint = manager.find_resume(options.get("resume", "auto"))
                except (H3CEError, OSError, ValueError):
                    pass  # Never mask the original failure or pretend a broken checkpoint is usable.
            atomic_write(path, canonical_json({"status": "initialization_failed", "phase": options.get("phase"),
                "resume_checkpoint": str(checkpoint) if checkpoint else None, "automatic_fallback": False,
                "error": type(exc).__name__, "message": str(exc), "optimizer_steps_in_this_run": 0}))
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise H3CEError("E_CUDA_OOM", "CUDA initialization exhausted memory; no automatic precision or device fallback. See training_interruption.json.") from exc
        raise
