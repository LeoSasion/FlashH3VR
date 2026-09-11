"""Checksummed atomic checkpoints and the shared preparation/training time ledger.

Only complete optimizer boundaries are serialized. A trainer interrupted during an
accumulation window must discard partial gradients and restore its window RNG and
sampler snapshot before calling save. No optimizer update is performed here.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from h3ce.budget import PreparationBudget
from h3ce.cache.keys import canonical_json, digest, file_sha256
from h3ce.cache.locks import FileLock
from h3ce.cache.store import assert_no_links, atomic_write, within
from h3ce.errors import H3CEError

_SCHEMA = 1
_NAME = re.compile(r"checkpoint-[a-z][a-z0-9_-]*-\d{12}-[a-f0-9]{12}\.pt")


def capture_rng_state() -> dict:
    """Encode NumPy state as primitives so weights_only=True needs no allowlist."""
    numpy_state = np.random.get_state()
    cuda_initialized = torch.cuda.is_initialized()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if cuda_initialized else [],
            "cuda_initialized": cuda_initialized}


def restore_rng_state(state: dict):
    try:
        numpy_state = state["numpy"]
        if state["cuda_initialized"]:
            if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
                raise H3CEError("E_CHECKPOINT_RNG", "Checkpoint CUDA device count does not match this runtime.")
        # Validate private copies before mutating any global RNG.
        python_check = random.Random()
        python_check.setstate(state["python"])
        numpy_tuple = (numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                       numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"])
        numpy_check = np.random.RandomState()
        numpy_check.set_state(numpy_tuple)
        torch_check = torch.Generator(device="cpu")
        torch_check.set_state(state["torch_cpu"].cpu())
        for cuda_state in state["torch_cuda"]:
            if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1:
                raise ValueError("Invalid CUDA RNG state")
        random.setstate(state["python"])
        np.random.set_state(numpy_tuple)
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state["cuda_initialized"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
        raise H3CEError("E_CHECKPOINT_RNG", f"Invalid RNG checkpoint: {exc}") from exc


def _safe_tree(value):
    """Reject custom Python objects instead of permitting arbitrary pickle globals."""
    if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, torch.Tensor):
        return
    if isinstance(value, (dict, list, tuple)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            if type(key) not in (str, int):
                raise H3CEError("E_CHECKPOINT_STATE", "Checkpoint keys must be strings or integers.")
            _safe_tree(child)
        return
    raise H3CEError("E_CHECKPOINT_STATE", f"Unsupported checkpoint object: {type(value).__name__}.")


def _validate_budget(snapshot):
    if not isinstance(snapshot, dict):
        raise H3CEError("E_BUDGET_STATE", "Checkpoint requires an explicit budget snapshot.")
    for key in ("used_seconds", "budget_seconds", "remaining_seconds"):
        value = snapshot.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise H3CEError("E_BUDGET_STATE", f"Invalid checkpoint budget field: {key}.")
    expected = max(0, snapshot["budget_seconds"] - snapshot["used_seconds"])
    if abs(snapshot["remaining_seconds"] - expected) > 1e-6:
        raise H3CEError("E_BUDGET_STATE", "Checkpoint remaining budget is inconsistent.")


class TrainingBudget(PreparationBudget):
    """Use exactly the preparation ledger and lock; old checkpoints never reset it."""

    def __init__(self, runs: Path, seconds: int, *, phase: str = "training"):
        super().__init__(runs, seconds)
        self.phase = phase

    def snapshot(self) -> dict:
        used = self.used
        snapshot = {"phase": self.phase, "used_seconds": used, "budget_seconds": self.limit,
                    "remaining_seconds": max(0, self.limit - used)}
        # The durable ledger must be at least as new as a checkpoint's snapshot.
        atomic_write(self.path, canonical_json(snapshot))
        return snapshot

    def save(self):
        self.snapshot()

    def check(self):
        self.save()
        if self.used >= self.limit:
            raise H3CEError("E_BUDGET_EXHAUSTED", "Cumulative preparation/training budget exhausted; save the last committed boundary.")

    def validate_resume_snapshot(self, snapshot: dict):
        _validate_budget(snapshot)
        # Compare with the ledger loaded on entry. Time spent loading a large VAE
        # in this process must not conceal a missing/truncated older ledger.
        if snapshot["used_seconds"] > self.previous["used_seconds"] + 1e-6:
            raise H3CEError("E_BUDGET_STATE", "Checkpoint accounting exceeds the shared ledger; restore the ledger, do not reset its budget.")


class CheckpointManager:
    """Commit .pt first, then its .json receipt; only receipted files can resume.

    ``contract`` is exact, canonical JSON. Callers must include effective resolved
    configuration, manifest SHA256, component lock hashes and implementation ID.
    ``runs_root`` is the configured durable runs directory, never the tmp cache.
    """

    def __init__(self, runs_root: Path, run_dir: Path, *, contract: dict):
        self.runs_root, self.run_dir = Path(runs_root).resolve(), Path(run_dir).resolve()
        assert_no_links(Path(runs_root))
        assert_no_links(Path(run_dir))
        if self.run_dir == self.runs_root or not within(self.run_dir, self.runs_root):
            raise H3CEError("E_CHECKPOINT_PATH", "Checkpoint run must be a child of the durable runs directory.")
        if not isinstance(contract, dict) or not contract:
            raise H3CEError("E_CHECKPOINT_CONTRACT", "An explicit nonempty resume contract is required.")
        self.contract = json.loads(canonical_json(contract))
        self.contract_id = digest(self.contract)
        self.directory = self.run_dir / "checkpoints"
        assert_no_links(self.directory)

    def save(self, *, model, optimizer, scheduler, scaler, sampler, stage: str, step: int,
             budget: dict, resolved_config: dict, accumulation_step: int = 0, extra: dict | None = None) -> Path:
        if type(accumulation_step) is not int or accumulation_step != 0 or sampler.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "Checkpoints require a committed optimizer boundary and no partial gradients.")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", stage) or type(step) is not int or not 0 <= step < 10**12:
            raise H3CEError("E_CHECKPOINT_STATE", "Invalid checkpoint stage or step.")
        _validate_budget(budget)
        payload = {"schema_version": _SCHEMA, "contract": self.contract, "contract_id": self.contract_id,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer is not None else None,
                   "scheduler": scheduler.state_dict() if scheduler is not None else None,
                   "scaler": scaler.state_dict() if scaler is not None else None,
                   "sampler": sampler.state_dict(), "rng": capture_rng_state(),
                   "stage": stage, "step": step, "accumulation_step": 0,
                   "budget": dict(budget), "resolved_config": json.loads(canonical_json(resolved_config)),
                   "extra": extra or {}}
        _safe_tree(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        assert_no_links(self.directory)
        name = f"checkpoint-{stage}-{step:012d}-{uuid.uuid4().hex[:12]}.pt"
        path = self.directory / name
        with FileLock(self.directory / ".checkpoint.lock"):
            fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                checksum = file_sha256(Path(temporary))
                os.replace(temporary, path)
                receipt = {"schema_version": _SCHEMA, "filename": name, "sha256": checksum,
                           "bytes": path.stat().st_size, "contract_id": self.contract_id,
                           "stage": stage, "step": step, "created_ns": time.time_ns()}
                atomic_write(path.with_suffix(".json"), canonical_json(receipt))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return path

    def _receipt(self, path: Path):
        path = Path(os.path.abspath(path))
        assert_no_links(path)
        if not within(path.resolve(), self.runs_root) or not _NAME.fullmatch(path.name) or path.parent.name != "checkpoints":
            raise H3CEError("E_CHECKPOINT_PATH", "Checkpoint must be a recognized file inside durable runs/checkpoints.")
        receipt_path = path.with_suffix(".json")
        assert_no_links(receipt_path)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["schema_version"] != _SCHEMA or receipt["filename"] != path.name:
                raise ValueError("Receipt schema or filename mismatch")
            if not isinstance(receipt["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", receipt["sha256"]):
                raise ValueError("Invalid SHA256")
            if type(receipt["created_ns"]) is not int or receipt["created_ns"] < 0:
                raise ValueError("Invalid checkpoint timestamp")
            if type(receipt["bytes"]) is not int or receipt["bytes"] < 1:
                raise ValueError("Invalid checkpoint size")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise H3CEError("E_CHECKPOINT_RECEIPT", f"Missing or invalid committed checkpoint receipt: {receipt_path.name}.") from exc
        return path, receipt

    def find_resume(self, resume: str | Path = "auto") -> Path | None:
        if str(resume) in ("none", "off"):
            return None
        if str(resume) != "auto":
            path = Path(resume)
            self.read(path)  # Explicit mismatches/corruption fail before model mutation.
            return path.resolve()
        candidates = []
        if not self.runs_root.exists():
            return None
        for receipt_path in self.runs_root.glob("*/checkpoints/checkpoint-*.json"):
            path, receipt = self._receipt(receipt_path.with_suffix(".pt"))
            if receipt.get("contract_id") == self.contract_id:
                candidates.append((receipt["created_ns"], path))
        if not candidates:
            return None
        path = max(candidates, key=lambda item: (item[0], str(item[1])))[1]
        self.read(path)  # Never silently bypass a corrupt latest matching checkpoint.
        return path

    def read(self, path: Path) -> dict:
        path, receipt = self._receipt(Path(path))
        if receipt.get("contract_id") != self.contract_id:
            raise H3CEError("E_CHECKPOINT_CONTRACT", "Checkpoint configuration, data or components do not match this run.")
        if not path.is_file() or path.stat().st_size != receipt["bytes"] or file_sha256(path) != receipt["sha256"]:
            raise H3CEError("E_CHECKPOINT_CHECKSUM", "Checkpoint payload failed its committed size/SHA256 checks.")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            _safe_tree(payload)
            if (payload["schema_version"] != _SCHEMA or payload["contract_id"] != self.contract_id
                    or payload["contract"] != self.contract or digest(payload["contract"]) != self.contract_id):
                raise H3CEError("E_CHECKPOINT_CONTRACT", "Checkpoint payload contract does not match.")
            if payload["accumulation_step"] != 0 or any(payload[k] != receipt[k] for k in ("stage", "step")):
                raise ValueError("Checkpoint boundary/receipt mismatch")
            _validate_budget(payload["budget"])
        except H3CEError:
            raise
        except Exception as exc:
            raise H3CEError("E_CHECKPOINT_STATE", f"Cannot safely load checkpoint: {exc}") from exc
        return payload

    def restore(self, path: Path, *, model, optimizer, scheduler, scaler, sampler,
                restore_rng: bool = True) -> dict:
        payload = self.read(path)
        for name, object_ in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if (payload[name] is None) != (object_ is None):
                raise H3CEError("E_CHECKPOINT_STATE", f"Checkpoint {name} presence does not match this stage.")
        try:
            model.load_state_dict(payload["model"], strict=True)
            sampler.load_state_dict(payload["sampler"])
            for name, object_ in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
                if object_ is not None:
                    object_.load_state_dict(payload[name])
            if restore_rng:
                restore_rng_state(payload["rng"])
        except H3CEError:
            raise
        except Exception as exc:
            raise H3CEError("E_CHECKPOINT_STATE", f"Checkpoint restore failed; stop this run: {exc}") from exc
        return payload
