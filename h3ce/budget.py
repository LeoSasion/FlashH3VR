"""Persistent preparation wall-time budget; no training success is implied."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from h3ce.cache.keys import canonical_json
from h3ce.cache.locks import FileLock
from h3ce.cache.store import atomic_write
from h3ce.errors import H3CEError


class PreparationBudget:
    def __init__(self, runs: Path, seconds: int):
        self.path = Path(runs) / "preparation_budget.json"
        self.limit = seconds
        self.lock = FileLock(Path(runs) / ".preparation_budget.lock")

    def __enter__(self):
        self.lock.__enter__()
        try:
            try:
                self.previous = json.loads(self.path.read_text()) if self.path.exists() else {"used_seconds": 0}
                used = self.previous["used_seconds"]
                if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used) or used < 0:
                    raise ValueError("used_seconds must be finite and nonnegative")
            except (ValueError, TypeError, KeyError) as exc:
                raise H3CEError("E_BUDGET_STATE", "Invalid saved preparation budget; accounting was not reset.") from exc
            self.start = time.monotonic()
            self.check()
        except BaseException:
            self.lock.__exit__(None, None, None)
            raise
        return self

    @property
    def used(self):
        return self.previous["used_seconds"] + time.monotonic() - self.start

    def save(self):
        used = self.used
        atomic_write(self.path, canonical_json({"phase": "prepare", "used_seconds": used,
            "budget_seconds": self.limit, "remaining_seconds": max(0, self.limit - used)}))

    def check(self):
        self.save()
        if self.used >= self.limit:
            raise H3CEError("E_BUDGET_EXHAUSTED", "Preparation budget exhausted; committed cache is reusable.")

    def __exit__(self, *exc):
        try:
            self.save()
        finally:
            self.lock.__exit__(*exc)
