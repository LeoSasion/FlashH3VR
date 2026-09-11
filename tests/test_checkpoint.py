"""CPU recovery contracts only: fixtures never call backward or optimizer.step."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from h3ce import budget as budget_module
from h3ce.budget import PreparationBudget
from h3ce.errors import H3CEError
from h3ce.train import checkpoint as checkpoint_module
from h3ce.train.checkpoint import (CheckpointManager, TrainingBudget, capture_rng_state,
                                  restore_rng_state)
from h3ce.train.sampler import StatefulSampler


class StateFixture:
    """State serialization fixture; intentionally has no update/step method."""

    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Training/optimizer updates are not authorized in recovery contract tests")
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)


@pytest.fixture
def bundle(tmp_path):
    runs = tmp_path / "runs"
    contract = {"resolved_config": {"native_temporal": "frozen"},
                "manifest_sha256": "1" * 64, "components_sha256": "2" * 64,
                "implementation": "recovery-test-only"}
    manager = CheckpointManager(runs, runs / "test-run", contract=contract)
    sampler = StatefulSampler(5, seed=17)
    model = torch.nn.Linear(3, 2)
    # The optimizer contents are an inert state fixture, not a constructed optimizer.
    optimizer = StateFixture({"state": {0: {"step": torch.tensor(7.),
                                           "exp_avg": torch.ones_like(model.weight)}},
                              "param_groups": [{"params": [0], "lr": 0.0002}]})
    scheduler = StateFixture({"last_epoch": 7, "base_lrs": [0.0002]})
    scaler = StateFixture({"scale": 1.0})
    options = {"model": model, "optimizer": optimizer, "scheduler": scheduler,
               "scaler": scaler, "sampler": sampler}
    save = {**options, "stage": "bootstrap_latent", "step": 7,
            "budget": {"used_seconds": 5.0, "budget_seconds": 100, "remaining_seconds": 95.0},
            "resolved_config": contract["resolved_config"], "extra": {"test_fixture": True}}
    return SimpleNamespace(runs=runs, contract=contract, manager=manager, sampler=sampler,
                           options=options, save=save)


def draws():
    return random.random(), np.random.standard_normal(4), torch.rand(4)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def consume(sampler, count):
    sampler.begin_window()
    values = [sampler.next_index() for _ in range(count)]
    sampler.commit_window()
    return values


def test_atomic_roundtrip_restores_all_state_rng_and_sampler_across_epoch(bundle):
    consume(bundle.sampler, 4)
    path = bundle.manager.save(**bundle.save)
    expected_weights = {name: value.clone() for name, value in bundle.options["model"].state_dict().items()}
    expected_random = draws()
    expected_indices = consume(bundle.sampler, 13)  # Crosses two epoch boundaries.
    with torch.no_grad():
        bundle.options["model"].weight.fill_(123)  # Serialization fixture mutation, no learning.
    bundle.options["optimizer"].state = {}
    bundle.options["scheduler"].state = {}
    bundle.options["scaler"].state = {}
    restored = bundle.manager.restore(path, **bundle.options)
    assert restored["step"] == 7 and restored["budget"]["used_seconds"] == 5.0
    assert restored["extra"]["test_fixture"]
    for name, actual in bundle.options["model"].state_dict().items():
        assert torch.equal(expected_weights[name], actual)
    assert bundle.options["optimizer"].state["param_groups"][0]["lr"] == 0.0002
    assert bundle.options["scheduler"].state["last_epoch"] == 7
    assert bundle.options["scaler"].state["scale"] == 1.0
    assert_draws_equal(expected_random, draws())
    assert consume(bundle.sampler, 13) == expected_indices
    assert bundle.manager.find_resume() == path


def test_weights_only_safe_payload_contains_no_numpy_pickle(bundle):
    path = bundle.manager.save(**bundle.save)
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert isinstance(state["rng"]["numpy"]["keys"], list)
    assert isinstance(state["sampler"]["generator_state"], torch.Tensor)


def test_uncommitted_gradient_window_cannot_be_saved_and_rollback_replays(bundle):
    starting_rng = capture_rng_state()
    bundle.sampler.begin_window()
    first = [bundle.sampler.next_index() for _ in range(8)]
    first_random = draws()
    with pytest.raises(H3CEError, match="E_CHECKPOINT_BOUNDARY"):
        bundle.manager.save(**bundle.save)
    bundle.sampler.rollback_window()
    restore_rng_state(starting_rng)
    assert consume(bundle.sampler, 8) == first
    assert_draws_equal(first_random, draws())
    with pytest.raises(H3CEError, match="E_CHECKPOINT_BOUNDARY"):
        bundle.manager.save(**bundle.save, accumulation_step=1)


def test_resume_auto_matches_exact_contract_and_explicit_mismatch_rejected(bundle):
    original = bundle.manager.save(**bundle.save)
    changed = dict(bundle.contract, manifest_sha256="3" * 64)
    second = CheckpointManager(bundle.runs, bundle.runs / "different-data", contract=changed)
    second_path = second.save(**bundle.save)
    assert bundle.manager.find_resume() == original
    assert second.find_resume() == second_path
    with pytest.raises(H3CEError, match="E_CHECKPOINT_CONTRACT"):
        bundle.manager.restore(second_path, **bundle.options)


def test_tampering_latest_matching_checkpoint_fails_instead_of_fallback(bundle):
    bundle.manager.save(**bundle.save)
    latest = bundle.manager.save(**dict(bundle.save, step=8))
    contents = bytearray(latest.read_bytes())
    contents[len(contents) // 2] ^= 1
    latest.write_bytes(contents)
    with pytest.raises(H3CEError, match="E_CHECKPOINT_CHECKSUM"):
        bundle.manager.find_resume()


def test_interrupted_payload_write_never_commits_and_preserves_previous(bundle, monkeypatch):
    original = bundle.manager.save(**bundle.save)
    def interrupted(payload, stream):
        stream.write(b"partial payload")
        raise OSError("simulated interruption")
    monkeypatch.setattr(checkpoint_module.torch, "save", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        bundle.manager.save(**dict(bundle.save, step=8))
    assert bundle.manager.find_resume() == original
    assert list(bundle.manager.directory.glob(".partial-*")) == []
    assert len(list(bundle.manager.directory.glob("*.pt"))) == 1


def test_interrupted_receipt_leaves_ignored_orphan_payload(bundle, monkeypatch):
    original = bundle.manager.save(**bundle.save)
    def interrupted(*args, **kwargs):
        raise OSError("receipt interruption")
    monkeypatch.setattr(checkpoint_module, "atomic_write", interrupted)
    with pytest.raises(OSError, match="receipt interruption"):
        bundle.manager.save(**dict(bundle.save, step=8))
    assert len(list(bundle.manager.directory.glob("*.pt"))) == 2
    assert bundle.manager.find_resume() == original
    orphan = next(path for path in bundle.manager.directory.glob("*.pt") if path != original)
    with pytest.raises(H3CEError, match="E_CHECKPOINT_RECEIPT"):
        bundle.manager.read(orphan)


def test_checkpoint_restore_component_presence_must_match(bundle):
    path = bundle.manager.save(**bundle.save)
    with pytest.raises(H3CEError, match="E_CHECKPOINT_STATE"):
        bundle.manager.restore(path, **dict(bundle.options, scaler=None))


def test_checkpoint_rejects_custom_pickle_objects(bundle):
    with pytest.raises(H3CEError, match="E_CHECKPOINT_STATE"):
        bundle.manager.save(**dict(bundle.save, extra={"path": Path("untrusted-object")}))


def test_checkpoint_path_must_stay_inside_runs(bundle, tmp_path):
    with pytest.raises(H3CEError, match="E_CHECKPOINT_PATH"):
        CheckpointManager(bundle.runs, tmp_path / "tmp" / "run", contract=bundle.contract)
    with pytest.raises(H3CEError, match="E_CHECKPOINT_PATH"):
        bundle.manager.read(tmp_path / "outside.pt")


@pytest.mark.parametrize("field,value", [("cursor", -1), ("cursor", 6), ("epoch", True),
                                        ("order", [0, 0, 1, 2, 3]), ("seed", 999)])
def test_invalid_sampler_state_rejected_without_changing_cursor(field, value):
    sampler = StatefulSampler(5, seed=17)
    state = sampler.state_dict()
    state[field] = value
    with pytest.raises(H3CEError, match="E_SAMPLER_STATE"):
        sampler.load_state_dict(state)
    assert sampler.cursor == 0 and sampler.epoch == 0


def test_sampler_window_lifecycle_and_unshuffled_resume():
    sampler = StatefulSampler(3, seed=2, shuffle=False)
    with pytest.raises(H3CEError, match="E_CHECKPOINT_BOUNDARY"):
        sampler.next_index()
    with pytest.raises(H3CEError, match="E_CHECKPOINT_BOUNDARY"):
        sampler.rollback_window()
    sampler.begin_window()
    with pytest.raises(H3CEError, match="E_CHECKPOINT_BOUNDARY"):
        sampler.begin_window()
    sampler.rollback_window()
    assert consume(sampler, 7) == [0, 1, 2, 0, 1, 2, 0]


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 100.0}
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: state["now"]))
    return state


def test_shared_budget_prepare_train_resume_accumulates_without_checkpoint_reset(tmp_path, clock):
    runs = tmp_path / "runs"
    with PreparationBudget(runs, 100):
        clock["now"] += 3
    with TrainingBudget(runs, 100) as training:
        assert training.used == 3
        clock["now"] += 5
        checkpoint_budget = training.snapshot()
        assert checkpoint_budget["used_seconds"] == 8
        assert json.loads(training.path.read_text())["used_seconds"] == 8
        clock["now"] += 7
    clock["now"] += 1000  # Idle time between processes is not active work.
    with TrainingBudget(runs, 100) as resumed:
        resumed.validate_resume_snapshot(checkpoint_budget)
        assert resumed.used == 15  # Resuming old checkpoint does not reset ledger to 8.
        clock["now"] += 1
    assert json.loads((runs / "preparation_budget.json").read_text())["used_seconds"] == 16


def test_budget_exhaustion_can_snapshot_before_exit_and_cannot_reset(tmp_path, clock):
    runs = tmp_path / "runs"
    with TrainingBudget(runs, 10) as training:
        clock["now"] += 11
        with pytest.raises(H3CEError, match="E_BUDGET_EXHAUSTED"):
            training.check()
        snapshot = training.snapshot()
        assert snapshot["used_seconds"] == 11 and snapshot["remaining_seconds"] == 0
    with pytest.raises(H3CEError, match="E_BUDGET_EXHAUSTED"):
        with TrainingBudget(runs, 10):
            pytest.fail("Exhausted budget cannot be resumed as zero")


def test_checkpoint_budget_newer_than_ledger_rejected(tmp_path, clock):
    with TrainingBudget(tmp_path / "runs", 100) as training:
        with pytest.raises(H3CEError, match="E_BUDGET_STATE"):
            training.validate_resume_snapshot({"used_seconds": 8., "budget_seconds": 100,
                                               "remaining_seconds": 92.})


@pytest.mark.parametrize("budget", [None, {"used_seconds": -1, "budget_seconds": 100, "remaining_seconds": 101},
                                     {"used_seconds": 3, "budget_seconds": 100, "remaining_seconds": 100}])
def test_invalid_budget_snapshot_cannot_commit(bundle, budget):
    with pytest.raises(H3CEError, match="E_BUDGET_STATE"):
        bundle.manager.save(**dict(bundle.save, budget=budget))
