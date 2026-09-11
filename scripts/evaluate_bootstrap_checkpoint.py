"""Forward-only comparison of a committed bootstrap checkpoint on 8–16 fixed pairs.

Scopes distinguish optimizer originals from unused originals in the same source
group. Neither scope registers a useful restoration base or permits its release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F

from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.config import load_config
from h3ce.model import SpatialRefinerV2
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget
from h3ce.train.engine import implementation_hashes
from h3ce.train.guard import no_training_guard
from h3ce.train.losses import ApplicationLoss, charbonnier, region_error
from h3ce.train.pipeline import move_sample, refine_latent, restore_pixels
from h3ce.train.preflight import execution_contract, load_bridge, open_dataset, require, require_image_bootstrap


def read_committed(root, run, checkpoint=None):
    """Receipt and exact saved contract are mandatory, even for unfinished runs."""
    run = Path(run).resolve()
    assert_no_links(run)
    config = load_config(run / "resolved.yaml")
    runs = (root / config.paths.runs).resolve()
    require(run != runs and run.is_relative_to(runs), "Review run must be inside project runs")
    contract = json.loads((run / "training_contract.json").read_text(encoding="utf-8"))
    report_path = run / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if checkpoint is None:
        require(report is not None, "Unfinished run requires an explicit committed --checkpoint")
        checkpoint = report["checkpoint"]
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint.parent == run / "checkpoints", "Checkpoint must belong to the selected run")
    manager = CheckpointManager(runs, run, contract=contract)
    state = manager.read(checkpoint)
    require(state["resolved_config"] == config.model_dump(mode="json")
            and contract["resolved_sha256"] == digest(state["resolved_config"]),
            "Saved configuration and checkpoint disagree")
    require(state["stage"] == contract["phase"] and state["stage"] in {"overfit", "latent", "pixel"},
            "Expected an image bootstrap checkpoint")
    if report is not None and Path(report["checkpoint"]).resolve() == checkpoint:
        require(report["optimizer_steps"] == state["step"] and report["phase"] == state["stage"],
                "Completed report and checkpoint disagree")
    return config, state, checkpoint


def verify_compatibility(root, config, state, dataset):
    expected = {"architecture": config.model.model_dump(mode="json"),
                "encoder_contract_id": dataset.encoder_contract_id,
                "components_sha256": file_sha256(root / config.paths.components_lock),
                "data": config.data.model_dump(mode="json"),
                "implementation_sha256": implementation_hashes(root), "runtime": execution_contract()}
    require(state["contract"]["compatibility"] == expected,
            "Checkpoint implementation, components, encoder or runtime differ from this evaluation")


def _manifest_index(path):
    identifiers = {"source_asset": "asset_id", "prepared_target": "target_id",
                   "degraded_variant": "variant_id", "training_view": "view_id"}
    result = {kind: {} for kind in identifiers}

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON object key in fixed evaluation manifest binding")
            value[key] = item
        return value

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_object)
        require(isinstance(row, dict) and isinstance(row.get("record_type"), str) and row["record_type"] in identifiers,
                "Unknown record type in fixed evaluation manifest binding")
        kind = row["record_type"]
        key = row.get(identifiers[kind])
        require(isinstance(key, str) and key, "Missing record identifier in fixed evaluation manifest binding")
        require(key not in result[kind], "Duplicate record identifier in fixed evaluation manifest binding")
        result[kind][key] = row
    return result


def verify_subset(run, state, dataset):
    request = json.loads((run / "training_request.json").read_text(encoding="utf-8"))
    manifest = Path(request["manifest"]).resolve()
    assert_no_links(manifest)
    require(file_sha256(manifest) == state["contract"]["manifest_sha256"],
            "Original training manifest no longer matches the checkpoint")
    require(file_sha256(dataset.manifest_path) == dataset.manifest_sha256,
            "Fixed evaluation manifest changed after dataset construction")
    original, fixed = _manifest_index(manifest), _manifest_index(dataset.manifest_path)
    parents = (("source_asset", "asset_id", dataset.sources),
               ("prepared_target", "target_id", dataset.targets),
               ("degraded_variant", "variant_id", dataset.variants))
    for view in dataset.views:
        require(view == fixed["training_view"].get(view["view_id"])
                == original["training_view"].get(view["view_id"]),
                "Fixed evaluation pairs must be unchanged views from the checkpoint training manifest")
        for kind, identifier, records in parents:
            key = view[identifier]
            require(records.get(key) is not None and records[key] == fixed[kind].get(key)
                    == original[kind].get(key),
                    f"Fixed evaluation {kind} must match the checkpoint training manifest")


def view_geometry(view):
    """Bind all non-variant fields, including future geometry extensions."""
    allowed_changes = {"view_id", "variant_id", "x_crop_path", "y_crop_path", "scene_x_path",
                       "pad_valid_map", "z_input_key", "z_input_path"}
    required = {"asset_id", "target_id", "mode", "bucket_hw", "crop_xyxy", "crop_to_original",
                "encoder_contract_id", "latent_execution_contract", "z_target_key", "z_target_path"}
    require(required <= view.keys(), "Companion view is missing bound geometry or target fields")
    return {key: value for key, value in view.items() if key not in allowed_changes}


def verify_source_closure(index, full, label):
    """Every selected record is original and every parent is actually referenced."""
    for kind, records in index.items():
        require(all(record == full[kind].get(key) for key, record in records.items()),
                f"{label} manifest records must be unchanged records from the companion source manifest")
    views = list(index["training_view"].values())
    require(views, f"{label} requires views")
    for kind, identifier in (("source_asset", "asset_id"), ("prepared_target", "target_id"),
                             ("degraded_variant", "variant_id")):
        require(set(index[kind]) == {view.get(identifier) for view in views},
                f"{label} companion manifest has missing or extra unbound {kind} records")
    for view in views:
        variant = index["degraded_variant"][view["variant_id"]]
        target = index["prepared_target"][view["target_id"]]
        source = index["source_asset"][view["asset_id"]]
        require(variant.get("target_id") == view["target_id"]
                and target.get("asset_id") == view["asset_id"]
                and source.get("split") == view.get("split") == "train",
                f"{label} view changes its source, target or split")
    return views


def verify_clean_companions(run, state, dataset, companion_source_manifest):
    """Bind clean diagnostics to 8–16 unique degraded optimizer originals.

    Only cache references derived from the changed variant may differ. All other
    view fields, including target latent keys and future geometry fields, remain
    identical. The normal dataset audit must still verify clean X=Y pixels.
    """
    request = json.loads((run / "training_request.json").read_text(encoding="utf-8"))
    training_manifest = Path(request["manifest"]).resolve()
    full_manifest = Path(companion_source_manifest).resolve()
    for path in (training_manifest, dataset.manifest_path, full_manifest):
        assert_no_links(path)
    training_sha = file_sha256(training_manifest)
    full_sha = file_sha256(full_manifest)
    require(training_sha == state["contract"]["manifest_sha256"],
            "Original training manifest no longer matches the checkpoint")
    require(file_sha256(dataset.manifest_path) == dataset.manifest_sha256,
            "Fixed evaluation manifest changed after dataset construction")
    original, fixed, full = (_manifest_index(path) for path in
                             (training_manifest, dataset.manifest_path, full_manifest))
    require(file_sha256(full_manifest) == full_sha, "Companion source manifest changed while binding")
    training_views = verify_source_closure(original, full, "Training")
    clean_views = verify_source_closure(fixed, full, "Clean evaluation")
    require(8 <= len(training_views) <= 16, "Companion optimizer manifest must contain 8–16 total pairs")
    degraded_views = [view for view in training_views
                      if original["degraded_variant"][view["variant_id"]].get("clean_pair") is False]
    require(8 <= len(degraded_views) <= 16
            and len({view["asset_id"] for view in degraded_views}) == len(degraded_views)
            and {view["mode"] for view in degraded_views} == {"fullbody", "face"},
            "Companion mode requires unique degraded-only training views and clean-only evaluation companions")
    require(8 <= len(clean_views) <= 16
            and len({view["asset_id"] for view in clean_views}) == len(clean_views)
            and len(clean_views) == len(degraded_views)
            and {view["mode"] for view in clean_views} == {"fullbody", "face"},
            "Clean companion binding requires 8–16 distinct originals covering both modes")
    require(all(fixed["degraded_variant"][view["variant_id"]].get("clean_pair") is True for view in clean_views),
            "Companion mode requires degraded-only training views and clean-only evaluation views")
    require({view["view_id"]: view for view in dataset.views} == fixed["training_view"]
            and len(dataset.views) == len(clean_views) and dataset.sources == fixed["source_asset"]
            and dataset.targets == fixed["prepared_target"] and dataset.variants == fixed["degraded_variant"],
            "Constructed dataset must match every clean companion manifest record")
    training_by_asset = {view["asset_id"]: view for view in degraded_views}
    mapping = []
    for clean in dataset.views:
        degraded = training_by_asset.get(clean["asset_id"])
        require(degraded is not None, "Clean companion original was not present in the checkpoint optimizer manifest")
        require(view_geometry(clean) == view_geometry(degraded),
                "Clean companion must preserve every geometry, target and non-variant view field")
        candidates = [candidate for candidate in full["training_view"].values()
                      if candidate.get("asset_id") == degraded["asset_id"]
                      and candidate.get("target_id") == degraded["target_id"]
                      and candidate.get("mode") == degraded["mode"]
                      and full["degraded_variant"].get(candidate.get("variant_id"), {}).get("clean_pair") is True
                      and view_geometry(candidate) == view_geometry(degraded)]
        require(len(candidates) == 1 and candidates[0] == clean,
                "Each optimizer view requires exactly one original clean counterpart with identical geometry")
        mapping.append({"asset_id": clean["asset_id"], "mode": clean["mode"],
                        "training_view_id": degraded["view_id"], "clean_view_id": clean["view_id"]})
    optional_clean = [view for view in training_views if view not in degraded_views]
    require(len({view["asset_id"] for view in optional_clean}) == len(optional_clean)
            and all(view == fixed["training_view"].get(view["view_id"])
                    and original["degraded_variant"][view["variant_id"]].get("clean_pair") is True
                    for view in optional_clean),
            "Optimizer clean views must be unique exact companions of its degraded originals")
    require(file_sha256(training_manifest) == training_sha
            and file_sha256(dataset.manifest_path) == dataset.manifest_sha256
            and file_sha256(full_manifest) == full_sha,
            "A companion binding manifest changed during verification")
    seen = [view["view_id"] for view in optional_clean]
    scope = ("same_training_original_clean_companions_seen_in_optimizer" if len(seen) == len(clean_views) else
             "same_training_original_clean_companions_partly_seen_in_optimizer" if seen else
             "same_training_original_clean_companions_not_seen_in_optimizer")
    return {"scope": scope, "optimizer_seen_clean_view_ids": seen,
            "source_manifest": str(full_manifest), "source_manifest_sha256": full_sha,
            "optimizer_manifest": str(training_manifest), "optimizer_manifest_sha256": training_sha,
            "mapping": mapping, "independent_validation": False,
            "clean_identity_verification": "mandatory_dataset_audit_before_any_model_forward"}


def verify_heldout_originals(run, state, dataset, heldout_source_manifest):
    """Prove original-content exclusion from a fresh optimizer, not independence."""
    request = json.loads((run / "training_request.json").read_text(encoding="utf-8"))
    require(request.get("resume") == "none" and request.get("init_run") is None
            and request.get("overfit_run") is None and request.get("phase") == "overfit"
            and state.get("stage") == "overfit"
            and state["contract"].get("lineage") == {
                "overfit_checkpoint_sha256": None, "initializer_checkpoint_sha256": None},
            "Heldout review requires fresh overfit initialization, resume none and no ancestral checkpoint")
    training_manifest = Path(request["manifest"]).resolve()
    full_manifest = Path(heldout_source_manifest).resolve()
    paths = (training_manifest, dataset.manifest_path, full_manifest)
    for path in paths:
        assert_no_links(path)
    hashes = [file_sha256(path) for path in paths]
    require(hashes[0] == state["contract"]["manifest_sha256"]
            and hashes[1] == dataset.manifest_sha256,
            "Heldout optimizer or evaluation manifest checksum changed")
    original, fixed, full = [_manifest_index(path) for path in paths]
    training = verify_source_closure(original, full, "Training")
    heldout = verify_source_closure(fixed, full, "Heldout evaluation")
    require(8 <= len(training) <= 16 and 8 <= len({view["asset_id"] for view in training}) <= 16,
            "Heldout comparison requires the bounded 8–16 pair optimizer manifest")
    require(len(heldout) == len({view["asset_id"] for view in heldout}) == 16
            and {view["mode"] for view in heldout} == {"fullbody", "face"},
            "Heldout comparison requires exactly 16 distinct originals covering both modes")
    require({view["view_id"]: view for view in dataset.views} == fixed["training_view"]
            and len(dataset.views) == 16 and dataset.sources == fixed["source_asset"]
            and dataset.targets == fixed["prepared_target"] and dataset.variants == fixed["degraded_variant"],
            "Constructed dataset must match every heldout manifest record")
    train_assets, eval_assets = set(original["source_asset"]), set(fixed["source_asset"])
    def source_hashes(index):
        values = [source.get("sha256") for source in index["source_asset"].values()]
        require(all(isinstance(value, str) and len(value) == 64
                    and set(value) <= set("0123456789abcdef") for value in values),
                "Heldout binding requires real original-content SHA256 values")
        require(len(set(values)) == len(values), "Distinct originals must have distinct source content hashes")
        return set(values)
    train_hashes, eval_hashes = source_hashes(original), source_hashes(fixed)
    require(not train_assets & eval_assets and not train_hashes & eval_hashes,
            "Heldout originals overlap optimizer asset IDs or source content hashes")
    train_groups = {source["source_group"] for source in original["source_asset"].values()}
    eval_groups = {source["source_group"] for source in fixed["source_asset"].values()}
    require(eval_groups <= train_groups, "This same-source diagnostic cannot claim a different source-group scope")
    clean_values = {fixed["degraded_variant"][view["variant_id"]].get("clean_pair") for view in heldout}
    require(len(clean_values) == 1 and all(type(value) is bool for value in clean_values),
            "Heldout evaluation must separately select degraded-only or clean-only views")
    require(hashes == [file_sha256(path) for path in paths], "A heldout binding manifest changed during verification")
    return {"scope": "same_source_originals_held_out_of_this_optimizer", "independent_validation": False,
            "source_manifest": str(full_manifest), "source_manifest_sha256": hashes[2],
            "optimizer_manifest": str(training_manifest), "optimizer_manifest_sha256": hashes[0],
            "evaluation_manifest": str(dataset.manifest_path), "evaluation_manifest_sha256": hashes[1],
            "optimizer_asset_ids": sorted(train_assets), "optimizer_source_sha256": sorted(train_hashes),
            "evaluation_asset_ids": sorted(eval_assets), "evaluation_source_sha256": sorted(eval_hashes),
            "asset_overlap": [], "source_sha256_overlap": [], "source_groups": sorted(eval_groups),
            "fresh_initialization_without_ancestral_checkpoint": True, "clean_pair": next(iter(clean_values)),
            "split_unchanged": "train", "near_duplicate_independence_verified": False}


def measure(sample, prediction, z_prediction, loss):
    """All metric values use unclamped tensors; masks count scalar RGB elements."""
    terms = loss(prediction, sample["y"], z_prediction, sample["z_target"], sample["valid"],
                 person_mask=sample["person_mask"], face_mask=sample["face_mask"])
    require(all(torch.isfinite(value).all().item() for value in terms.values()), "Nonfinite evaluation loss")
    values = {key: float(value) for key, value in terms.items()}
    error = (prediction - sample["y"]).float()
    valid = sample["valid"]
    epsilon = loss.weights.get("charbonnier_epsilon", .001)
    counts = {"valid_rgb_elements": float(valid.sum()) * prediction.shape[1]}
    values["rgb_global_mae"] = float(region_error(error.abs(), valid))
    values["rgb_global_charbonnier"] = float(region_error(charbonnier(error, epsilon), valid))
    weights = {name: counts["valid_rgb_elements"] for name in
               ("rgb_global_mae", "rgb_global_charbonnier", "lighting_target")}
    for name in ("person", "face"):
        mask = sample[name + "_mask"] * valid
        count = float(mask.sum()) * prediction.shape[1]
        counts[name + "_rgb_elements"] = count
        values["rgb_" + name + "_mae"] = float(region_error(error.abs(), mask)) if count else None
        if not count:
            values["rgb_" + name] = None
        weights["rgb_" + name + "_mae"] = count
        weights["rgb_" + name] = count
    latent_weight = F.interpolate(valid.float(), size=z_prediction.shape[2:], mode="area")
    values["latent_mae"] = float(region_error((z_prediction - sample["z_target"]).abs(), latent_weight))
    counts["valid_latent_elements"] = float(latent_weight.sum()) * z_prediction.shape[1]
    weights["latent"] = weights["latent_mae"] = counts["valid_latent_elements"]
    values["out_of_range_fraction"] = float(region_error(((prediction < 0) | (prediction > 1)).float(), valid))
    weights["out_of_range_fraction"] = counts["valid_rgb_elements"]
    excursion = F.relu(-prediction) + F.relu(prediction - 1)
    values["out_of_range_mae"] = float(region_error(excursion.float(), valid))
    values["out_of_range_max"] = float(excursion.masked_select(valid.bool().expand_as(excursion)).max())
    weights["out_of_range_mae"] = counts["valid_rgb_elements"]
    return {"metrics": values, "element_weights": weights, "counts": counts}


def aggregate(cases, label):
    """Report source-equal and true element-pooled components separately.

    Each fixed pair must have its own original asset. Composite region-normalized
    training losses have no common pixel denominator, so are only source-equal.
    """
    require(cases and len({case["asset_id"] for case in cases}) == len(cases),
            "Source-equal aggregation requires one fixed view per original asset")
    keys = cases[0][label]["metrics"]
    means, pooled = {}, {}
    for metric in keys:
        values = [case[label]["metrics"][metric] for case in cases if case[label]["metrics"][metric] is not None]
        means[metric] = sum(values) / len(values) if values else None
        entries = [(case[label]["metrics"][metric], case[label]["element_weights"].get(metric, 0)) for case in cases]
        denominator = sum(weight for value, weight in entries if value is not None)
        if denominator > 0:
            pooled[metric] = sum(value * weight for value, weight in entries if value is not None) / denominator
    return {"source_count": len(cases), "source_equal_mean": means, "valid_element_pooled": pooled}


def relative_change(value, baseline):
    return None if baseline == 0 else value / baseline - 1


def compare(cases, baseline):
    summary = {}
    for metric in ("total", "rgb", "rgb_global_mae", "latent", "lighting_target"):
        differences = [case["current"]["metrics"][metric] - case[baseline]["metrics"][metric] for case in cases]
        summary[metric] = {"improved_cases": sum(value < -1e-12 for value in differences),
                           "worsened_cases": sum(value > 1e-12 for value in differences),
                           "unchanged_cases": sum(abs(value) <= 1e-12 for value in differences)}
    current, before = aggregate(cases, "current"), aggregate(cases, baseline)
    for weighting in ("source_equal_mean", "valid_element_pooled"):
        summary[weighting] = {metric: {"absolute_change": value - before[weighting][metric],
                                     "relative_change": relative_change(value, before[weighting][metric])}
                              for metric, value in current[weighting].items()
                              if value is not None and before[weighting].get(metric) is not None}
    return summary


def image_from_tensor(tensor):
    array = tensor[0, :, 0].float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray(np.floor(array * 255 + .5).astype(np.uint8))


def render_pairs(output, cases, images, *, scope_caption="same training original"):
    sheets, paths = [], []
    labels = ("INPUT X", "TARGET Y", "CURRENT CHECKPOINT") + (("OVERFIT REFERENCE",) if "reference" in images[0] else ())
    for index, (case, frames) in enumerate(zip(cases, images)):
        keys = ("input", "target", "current") + (("reference",) if "reference" in frames else ())
        width, height = frames["input"].size
        full = Image.new("RGB", (len(keys) * width, height + 52), (24, 26, 30))
        draw = ImageDraw.Draw(full)
        draw.text((8, 4), f"{index+1:02d} {case['mode']} {case['source']} | {scope_caption}", fill="white")
        for column, (key, label) in enumerate(zip(keys, labels)):
            draw.text((column * width + 8, 27), label, fill="white")
            full.paste(frames[key], (column * width, 52))
        path = output / f"pair_{index+1:02d}_{case['mode']}.png"
        full.save(path)
        case["preview"] = str(path)
        paths.append(str(path))
        if index % 4 == 0:
            sheet = Image.new("RGB", (len(keys) * 320, 4 * 300), (24, 26, 30))
        tile = ImageOps.contain(full, (sheet.width, 296))
        sheet.paste(tile, ((sheet.width - tile.width) // 2, (index % 4) * 300))
        if index % 4 == 3 or index == len(cases) - 1:
            destination = output / f"overview_{index//4+1:02d}.png"
            sheet.save(destination)
            sheets.append(str(destination))
    return paths, sheets


def evaluate(run, manifest, checkpoint=None, reference_run=None, output=None, companion_source_manifest=None,
             heldout_source_manifest=None):
    require(not (companion_source_manifest and reference_run),
            "Companion mode does not permit --reference-run")
    require(not (heldout_source_manifest and (companion_source_manifest or reference_run)),
            "Heldout mode does not permit companion mode or --reference-run")
    root = Path(__file__).resolve().parents[1]
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    config = load_config(run / "resolved.yaml")
    require_image_bootstrap(config)
    with no_training_guard() as guard, TrainingBudget(root / config.paths.runs, config.project.budget_seconds,
                                                    phase="bootstrap_fixed_forward_evaluation") as budget:
        budget.check()
        config, state, checkpoint = read_committed(root, run, checkpoint)
        budget.validate_resume_snapshot(state["budget"])
        checkpoint_sha = file_sha256(checkpoint)
        output = Path(output).resolve() if output else run / f"fixed_evaluation_{state['step']:08d}_{checkpoint_sha[:8]}"
        assert_no_links(output)
        require(output.is_relative_to((root / config.paths.runs).resolve()) and not output.exists(),
                "Evaluation output must be a new directory inside project runs")
        dataset = open_dataset(config, root, manifest)
        require(8 <= len(dataset) <= 16 and len({view["asset_id"] for view in dataset.views}) == len(dataset)
                and {view["mode"] for view in dataset.views} == {"fullbody", "face"},
                "This review requires 8–16 distinct originals covering both modes")
        verify_compatibility(root, config, state, dataset)
        companion_binding = None
        heldout_binding = None
        if companion_source_manifest:
            companion_binding = verify_clean_companions(run, state, dataset, companion_source_manifest)
        elif heldout_source_manifest:
            heldout_binding = verify_heldout_originals(run, state, dataset, heldout_source_manifest)
        else:
            verify_subset(run, state, dataset)
        audit = dataset.audit(budget_check=budget.check)
        output.mkdir(parents=True)
        bridge = load_bridge(config, root, dataset)
        model = SpatialRefinerV2.from_config(config.model).to("cuda").eval().requires_grad_(False)
        from h3ce.train.perceptual import application_loss
        loss = application_loss(config, root)
        vae_versions = {name: (id(p), p._version) for name, p in bridge.backend.model.named_parameters()}
        cases, images, evaluated = [], [], {}
        model_states = [("current", run, config, state, checkpoint)]
        if reference_run:
            reference_run = Path(reference_run).resolve()
            ref_config, ref_state, ref_checkpoint = read_committed(root, reference_run)
            require(ref_state["stage"] == "overfit" and ref_state["extra"].get("phase_complete") is True
                    and ref_state["extra"].get("overfit_evidence", {}).get("passed") is True,
                    "Reference must be a completed passed overfit checkpoint")
            require(ref_config.training.losses == config.training.losses, "Reference loss configuration differs")
            verify_compatibility(root, ref_config, ref_state, dataset)
            verify_subset(reference_run, ref_state, dataset)
            model_states.append(("reference", reference_run, ref_config, ref_state, ref_checkpoint))
        torch.cuda.reset_peak_memory_stats()
        for label, source_run, source_config, source_state, source_checkpoint in model_states:
            budget.check()
            model.load_state_dict(source_state["model"], strict=True)
            versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
            for index in range(len(dataset)):
                budget.check()
                sample = move_sample(dataset[index], "cuda")
                zp, _ = refine_latent(model, sample, autocast_enabled=torch.cuda.is_bf16_supported())
                prediction, pack = restore_pixels(bridge, sample, zp, grad=False, strength=source_config.model.output.strength)
                result = measure(sample, prediction, zp, loss)
                if label == "current":
                    case = {"index": index, "asset_id": sample["asset_id"], "view_id": sample["view_id"],
                            "source": Path(dataset.sources[sample["asset_id"]]["path"]).name, "mode": sample["mode"],
                            "bucket_hw": list(sample["bucket_hw"]), "current": result,
                            "input": measure(sample, sample["x"], sample["z_input"], loss)}
                    cases.append(case)
                    images.append({"input": image_from_tensor(sample["x"]), "target": image_from_tensor(sample["y"]),
                                   "current": image_from_tensor(prediction)})
                else:
                    cases[index][label] = result
                    images[index][label] = image_from_tensor(prediction)
                print(json.dumps({"event": "fixed_pair_evaluated", "model": label, "index": index,
                                  "rgb_global_mae": result["metrics"]["rgb_global_mae"]}), flush=True)
            require(versions == {name: (id(p), p._version) for name, p in model.named_parameters()},
                    "Evaluation changed model parameters")
            evaluated[label] = {"run": str(source_run), "checkpoint": str(source_checkpoint),
                                "checkpoint_sha256": file_sha256(source_checkpoint), "contract_id": source_state["contract_id"],
                                "phase": source_state["stage"], "optimizer_steps_in_checkpoint": source_state["step"],
                                "phase_complete": source_state["extra"].get("phase_complete", False)}
        require(vae_versions == {name: (id(p), p._version) for name, p in bridge.backend.model.named_parameters()},
                "Evaluation changed H3 parameters")
        require(not any(p.grad is not None or p.requires_grad for p in model.parameters())
                and not any(p.grad is not None or p.requires_grad for p in bridge.backend.model.parameters()),
                "Forward review created trainable parameters or gradients")
        caption = ("same source; original held out of this optimizer" if heldout_binding else
                   "training original; clean partly seen by optimizer" if companion_binding
                   and 0 < len(companion_binding["optimizer_seen_clean_view_ids"]) < len(dataset) else
                   "training original; clean seen by optimizer" if companion_binding
                   and companion_binding["optimizer_seen_clean_view_ids"] else
                   "training original; clean not seen by optimizer" if companion_binding else "same training original")
        _, sheets = render_pairs(output, cases, images, scope_caption=caption)
        labels = ["input", *evaluated]
        summaries = {label: aggregate(cases, label) for label in labels}
        comparisons = {label: compare(cases, label) for label in labels if label != "current"}
        by_mode = {mode: {label: aggregate([case for case in cases if case["mode"] == mode], label) for label in labels}
                   for mode in ("fullbody", "face")}
        torch.cuda.synchronize()
        result = {"status": "completed_fixed_training_pair_evaluation", "scope": f"same_{len(dataset)}_training_originals_not_independent_validation",
                  "trained_base_accepted": False, "additional_optimizer_steps": 0, "execution_guard": dict(guard),
                  "models": evaluated, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
                  "data_audit": audit, "cases": cases, "summaries": summaries, "comparisons": comparisons, "by_mode": by_mode,
                  "sheets": sheets, "metric_notes": {"rgb": "global + person + face independently normalized Charbonnier",
                      "lighting_target": "masked linear-RGB lowpass Charbonnier averaged at sigma 4/16/32 scaled to 512 short edge",
                      "rgb_global_mae": "unclamped prediction versus Y over valid RGB scalar elements",
                      "previews": f"all {len(dataset)} cases, clamp and round only for PNG display; metrics remain unclamped",
                      "source_equal_mean": "one equally weighted view per distinct original; not independent source groups",
                      "valid_element_pooled": "each component pooled by its own valid RGB/box/latent scalar count"},
                  "runtime": execution_contract(), "effective_decoder_hash": pack.effective_decoder_hash,
                  "h3_parameters_frozen_and_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "budget": budget.snapshot(), "limitations": ["No independent source-group quality validation", "No motion metric from still images",
                      "LPIPS-Alex active" if config.training.losses.perceptual else "LPIPS explicitly disabled",
                      "This report does not register or release a useful base"]}
        if companion_binding is not None:
            require(file_sha256(Path(companion_binding["source_manifest"]))
                    == companion_binding["source_manifest_sha256"],
                    "Companion source manifest changed during evaluation")
            result.update(status="completed_clean_companion_evaluation", scope=companion_binding["scope"],
                          companion_binding=companion_binding)
        if heldout_binding is not None:
            require(file_sha256(Path(heldout_binding["source_manifest"])) == heldout_binding["source_manifest_sha256"],
                    "Heldout source manifest changed during evaluation")
            result.update(status="completed_same_source_heldout_evaluation", scope=heldout_binding["scope"],
                          heldout_binding=heldout_binding)
        atomic_write(output / "metrics.json", canonical_json(result))
    print(json.dumps({"status": result["status"], "report": str(output / "metrics.json"), "sheets": sheets}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--reference-run")
    parser.add_argument("--output")
    parser.add_argument("--companion-source-manifest",
                        help="Explicit original full preparation manifest for same-original clean companions; no reference model")
    parser.add_argument("--heldout-source-manifest",
                        help="Explicit full preparation manifest for originals excluded from a fresh optimizer; same source groups")
    evaluate(**vars(parser.parse_args()))
