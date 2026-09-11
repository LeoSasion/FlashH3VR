"""Deterministic sampling with accumulation windows that can be rolled back."""
from __future__ import annotations

import copy

import torch

from h3ce.errors import H3CEError


class StatefulSampler:
    """A single-process sampler; commit only after the corresponding optimizer step.

    The trainer owns global RNG rollback and gradient disposal. This class owns its
    private shuffle generator, order and cursor, including windows crossing epochs.
    Checkpoints cannot serialize an open window and accidentally skip its samples.
    """

    def __init__(self, length: int, *, seed: int = 42, shuffle: bool = True):
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise H3CEError("E_SAMPLER_STATE", "Sampler length must be a positive integer.")
        if type(seed) is not int or not 0 <= seed < 2**63 or type(shuffle) is not bool:
            raise H3CEError("E_SAMPLER_STATE", "Sampler seed must be a nonnegative int64 and shuffle a boolean.")
        self.length, self.seed, self.shuffle = length, seed, shuffle
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = self._new_order()
        self._window = None

    def _new_order(self):
        return (torch.randperm(self.length, generator=self.generator).tolist()
                if self.shuffle else list(range(self.length)))

    @property
    def pending_window(self) -> bool:
        return self._window is not None

    def _snapshot(self):
        return {"schema_version": 1, "length": self.length, "seed": self.seed,
                "shuffle": self.shuffle, "epoch": self.epoch, "cursor": self.cursor,
                "order": list(self.order), "generator_state": self.generator.get_state().clone()}

    def state_dict(self):
        if self.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "Commit or roll back the accumulation window before saving.")
        return self._snapshot()

    def load_state_dict(self, state):
        try:
            if self.pending_window:
                raise ValueError("Cannot restore during an open window")
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                raise ValueError("Unsupported sampler schema")
            for field in ("length", "seed", "shuffle"):
                if state[field] != getattr(self, field) or type(state[field]) is not type(getattr(self, field)):
                    raise ValueError(f"Sampler {field} does not match")
            for field in ("cursor", "epoch"):
                if isinstance(state[field], bool) or not isinstance(state[field], int) or state[field] < 0:
                    raise ValueError(f"Invalid {field}")
            if state["cursor"] > self.length:
                raise ValueError("Cursor exceeds dataset length")
            order = state["order"]
            if not isinstance(order, list) or any(type(i) is not int for i in order) or sorted(order) != list(range(self.length)):
                raise ValueError("Order must be a permutation of the dataset")
            generator = torch.Generator(device="cpu")
            generator.set_state(state["generator_state"].cpu())
            if not self.shuffle and order != list(range(self.length)):
                raise ValueError("Unshuffled order changed")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            raise H3CEError("E_SAMPLER_STATE", f"Invalid sampler checkpoint: {exc}") from exc
        self.generator = generator
        self.order, self.cursor, self.epoch = list(order), state["cursor"], state["epoch"]

    def begin_window(self):
        if self.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "An accumulation window is already open.")
        self._window = self._snapshot()

    def next_index(self) -> int:
        if not self.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "Begin an accumulation window before consuming samples.")
        if self.cursor == self.length:
            self.epoch += 1
            self.cursor = 0
            self.order = self._new_order()
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def commit_window(self):
        if not self.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "No accumulation window to commit.")
        self._window = None

    def rollback_window(self):
        if not self.pending_window:
            raise H3CEError("E_CHECKPOINT_BOUNDARY", "No accumulation window to roll back.")
        state = copy.deepcopy(self._window)
        self._window = None
        self.load_state_dict(state)
