"""Preparation, historical diagnostics and explicit image-bootstrap phases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from h3ce.cache.keys import canonical_json
from h3ce.cache.store import CacheStore, assert_no_links, assert_separate_paths, atomic_write
from h3ce.config import ProjectConfig, load_config, project_root, write_resolved
from h3ce.doctor import diagnose, inventory_or_error, resolve_optional_losses
from h3ce.errors import H3CEError


def parser():
    parser = argparse.ArgumentParser(prog="h3ce", description="H3CE v2: image/video preparation and verified H3 codec engineering.")
    parser.add_argument("--version", action="version", version="h3ce 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    def configured(name, help):
        child = commands.add_parser(name, help=help)
        child.add_argument("--config", default="configs/project.v2.yaml")
        return child

    doctor = configured("doctor", "Report environment and unresolved component contracts")
    doctor.add_argument("--allow-download", action="store_true")
    prepare = configured("prepare", "Prepare trusted HQ images/video with locked local YOLO11 detectors")
    prepare.add_argument("--report", action="store_true")
    prepare.add_argument("--allow-download", action="store_true")
    train = configured("train", "Image bootstrap or forward-only inspection; later stages remain gated")
    train.add_argument("--stage", choices=["bootstrap", "character", "native-codec"], required=True)
    train.add_argument("--resume", default="auto")
    train.add_argument("--mode", choices=["fullbody", "face"])
    train.add_argument("--name")
    train.add_argument("--native-temporal", choices=["frozen", "finetune"])
    train.add_argument("--check-only", action="store_true", help="Audit cached images and forward through real H3; prohibit optimizers/backward")
    train.add_argument("--phase", choices=["overfit", "latent", "pixel"], default="overfit")
    train.add_argument("--manifest", help="Materialized image TrainingView JSONL; default: latest image preparation")
    train.add_argument("--max-steps", type=int, help="Optimizer steps in the selected phase; incompatible with check-only")
    train.add_argument("--overfit-run", help="Successful current-code overfit evidence required for long phases")
    train.add_argument("--init-run", help="Previous completed phase whose R and scene weights initialize this phase")
    cfg = commands.add_parser("config", help="Validate the complete strict v2 configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    cfg_validate = cfg_sub.add_parser("validate")
    cfg_validate.add_argument("--config", default="configs/project.v2.yaml")
    cache = commands.add_parser("cache", help="Inspect cache or prune unpinned indexed entries")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    for name in ("inspect", "prune"):
        child = cache_sub.add_parser(name)
        child.add_argument("--project", default=".")
        child.add_argument("--config")
        if name == "prune":
            mode = child.add_mutually_exclusive_group()
            mode.add_argument("--dry-run", action="store_true")
            mode.add_argument("--apply", action="store_true")
    inference = commands.add_parser("infer", help="Restoration inference awaits the trained M2 base")
    inference.add_argument("--input", required=True)
    inference.add_argument("--codec-pack", default="base")
    inference.add_argument("--lora", action="append", default=[])
    evaluation = commands.add_parser("evaluate", help="Fixed-suite evaluation awaits the real pipeline")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--suite", default="fixed")
    migration = commands.add_parser("migrate-config", help="V1 semantic migration is not yet implemented")
    migration.add_argument("--from-v1", required=True)
    migration.add_argument("--out", required=True)
    return parser


def _new_run(config, root, command):
    protected = [root / getattr(config.paths, field) for field in ("raw", "tmp", "runs", "exports", "models")]
    assert_separate_paths(*protected)
    directory = root / config.paths.runs / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + command + "-" + uuid.uuid4().hex[:8])
    assert_no_links(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_resolved(config, directory / "resolved.yaml")
    return directory


def _execute(args):
    if args.command in {"infer", "evaluate", "migrate-config"}:
        raise H3CEError("E_NOT_IMPLEMENTED", f"{args.command} is not implemented at the current milestone.")
    if args.command == "cache":
        root = Path(args.project).resolve()
        config = load_config(args.config or root / "configs/project.v2.yaml")
        # --project explicitly selects the project, independent of cwd/config placement.
        store = CacheStore(root / config.paths.tmp, create=False,
                           protected=[root / getattr(config.paths, key) for key in ("raw", "runs", "exports", "models")])
        return store.inspect() if args.cache_command == "inspect" else store.prune(dry_run=not args.apply)
    config = load_config(args.config)
    root = project_root(config, args.config)
    if args.command == "train" and args.native_temporal:
        data = config.model_dump(mode="json")
        data["native_temporal"]["mode"] = args.native_temporal
        config = ProjectConfig.model_validate(data)
    if args.command == "train" and args.mode:
        config.lora.train_mode = args.mode
    inventory = inventory_or_error(config, root)
    if args.command != "config":
        config, resolution_notes = resolve_optional_losses(config, inventory, root)
    else:
        resolution_notes = []
    run = _new_run(config, root, args.command)
    atomic_write(run / "resolution_notes.json", canonical_json(resolution_notes))
    try:
        if getattr(args, "allow_download", False):
            from h3ce.acquire import acquire_detectors
            acquisition = acquire_detectors(config, root, allow_download=True)
            atomic_write(run / "acquisition.json", canonical_json(acquisition))
            inventory = inventory_or_error(config, root)
        if args.command == "config":
            return {"status": "valid", "resolved": str(run / "resolved.yaml")}
        if args.command == "doctor":
            report = diagnose(config, root, inventory)
            atomic_write(run / "doctor.json", canonical_json(report))
            return {"status": report["status"], "report": str(run / "doctor.json"),
                    "resolved": str(run / "resolved.yaml"), "training_allowed": False,
                    "message": "Historical M0/M1 checks and implemented M2 image code are separate from pending training acceptance."}
        if args.command == "prepare":
            from h3ce.prepare import prepare_project
            result = prepare_project(config, root, run)
            return {"run": str(run), **result}
        if args.command == "train":
            if args.stage in {"character", "native-codec"}:
                if args.stage == "native-codec":
                    raise H3CEError("E_REAL_VIDEO_REQUIRED", "No verified real-video manifest exists for native Decoder adaptation.")
                raise H3CEError("E_BASE_REQUIRED", "No accepted trained restoration base is registered.")
            diagnostics = diagnose(config, root, inventory)
            atomic_write(run / "doctor.json", canonical_json(diagnostics))
            milestones = diagnostics["milestones"]
            if not all(milestones[key]["acceptance"].startswith("passed") for key in ("M0", "M1")):
                raise H3CEError("E_MILESTONE_REQUIRED", "Verified M0 detection and M1 H3 acceptance must be registered before bootstrap.")
            if args.mode or args.name:
                raise H3CEError("E_TRAINING_CONTRACT", "Bootstrap trains the shared base on both view modes; mode/name belong to character LoRA")
            if args.check_only and any((args.max_steps is not None, args.overfit_run, args.init_run, args.resume != "auto")):
                raise H3CEError("E_TRAINING_CONTRACT", "Check-only inspects initialization; training/resume overrides must be omitted")
            from h3ce.train.preflight import check_only, resolve_manifest
            manifest = resolve_manifest(config, root, args.manifest, args.phase)
            atomic_write(run / "training_request.json", canonical_json({
                "stage": args.stage, "phase": args.phase, "check_only": args.check_only,
                "manifest": str(manifest), "resume": args.resume, "max_steps": args.max_steps,
                "overfit_run": args.overfit_run, "init_run": args.init_run}))
            if args.check_only:
                return check_only(config, root, run, manifest, args.phase)
            from h3ce.train.engine import train_images
            return train_images(config, root, run, manifest, phase=args.phase, max_steps=args.max_steps,
                                resume=args.resume, overfit_run=args.overfit_run, init_run=args.init_run)
    except BaseException as exc:
        error = exc.as_dict() if isinstance(exc, H3CEError) else {"error": type(exc).__name__, "message": str(exc)}
        atomic_write(run / "failure.json", canonical_json(error))
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = _execute(args)
    except H3CEError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "E_INTERRUPTED", "message": "Interrupted; committed cache and run diagnostics are retained."}), file=sys.stderr)
        return 130
    except OSError as exc:
        print(json.dumps({"error": "E_IO", "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") in {"blocked", "failed_overfit_probe"} else 0
