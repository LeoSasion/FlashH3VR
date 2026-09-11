"""Character increments on existing spatial FFN projections, never temporal layers."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable, Mapping
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from h3ce.errors import H3CEError


def valid_alias(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
        raise H3CEError("E_LORA_CONTRACT", "Adapter aliases must contain 1..128 ASCII letters, digits, underscores or hyphens")
    return name


def validate_rank_alpha(rank: int, alpha: float) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise H3CEError("E_LORA_CONTRACT", "Character rank must be a positive integer")
    if not isinstance(alpha, Real) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise H3CEError("E_LORA_CONTRACT", "Character alpha must be finite and positive")


def canonical_mix(mix: Mapping[str, float] | Iterable[tuple[str, float]], *, warn: bool = True) -> tuple[tuple[str, float], ...]:
    """Sort aliases to make accumulation order independent of the supplied list."""
    entries = list(mix.items() if isinstance(mix, Mapping) else mix)
    result = {}
    for item in entries:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise H3CEError("E_LORA_CONTRACT", "Each mix entry must be (alias, finite strength)")
        name, strength = item
        valid_alias(name)
        if name in result:
            raise H3CEError("E_LORA_CONTRACT", f"Duplicate adapter alias in mix: {name}")
        if not isinstance(strength, Real) or isinstance(strength, bool) or not math.isfinite(strength):
            raise H3CEError("E_LORA_CONTRACT", "Mix strengths must be arbitrary finite real numbers")
        result[name] = float(strength)
    if warn and any(abs(value) > 1 for value in result.values()):
        warnings.warn("LoRA strengths outside [-1, 1] can amplify artifacts; supplied values are preserved", UserWarning, stacklevel=2)
    return tuple(sorted(result.items()))


class CharacterIncrement(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        validate_rank_alpha(rank, alpha)
        self.rank, self.alpha = rank, float(alpha)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32), requires_grad=False)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * (self.alpha / self.rank)


class MultiLoRALinear(nn.Module):
    """W0(x) + sum strength * (alpha/r) * B(A(x)); all branches use x.

    Dropout is deliberately zero for the initial character implementation. W0
    and all adapters are frozen on construction; selecting trainable parameters
    is separate from executing training and does not accept a restoration base.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if type(base) is not nn.Linear:
            raise H3CEError("E_LORA_MODULE_CONTRACT", "Character adapters require an existing exact nn.Linear")
        self.base = base.requires_grad_(False)
        self.adapters = nn.ModuleDict()
        self.mix: tuple[tuple[str, float], ...] = ()
        self.trainable_alias: str | None = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def add_adapter(self, name: str, *, rank: int = 8, alpha: float = 8.0) -> None:
        valid_alias(name)
        validate_rank_alpha(rank, alpha)
        if name in self.adapters:
            raise H3CEError("E_LORA_CONTRACT", f"Adapter already exists: {name}")
        self.adapters[name] = CharacterIncrement(self.base, rank=rank, alpha=alpha)

    def set_mix(self, mix, *, warn: bool = True) -> None:
        selected = canonical_mix(mix, warn=warn)
        unknown = [name for name, _ in selected if name not in self.adapters]
        if unknown:
            raise H3CEError("E_LORA_CONTRACT", "Unknown character adapters", unknown)
        if self.trainable_alias is not None and selected != ((self.trainable_alias, 1.0),):
            raise H3CEError("E_LORA_TRAINING_MIX", "Character training activates only the selected adapter at strength 1")
        self.mix = selected

    def select_trainable(self, name: str | None) -> None:
        if name is not None and name not in self.adapters:
            raise H3CEError("E_LORA_CONTRACT", f"Unknown character adapter: {name}")
        self.requires_grad_(False)
        self.trainable_alias = name
        if name is not None:
            self.adapters[name].requires_grad_(True)
            self.set_mix(((name, 1.0),), warn=False)
        else:
            self.mix = ()

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        for name, strength in self.mix:
            if strength != 0:
                delta = self.adapters[name](x)
                output = output + delta.to(output.dtype) * strength
        return output
