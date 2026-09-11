"""Resolve exact spatial anchors and route user weights without normalization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from h3ce.errors import H3CEError
from .layers import CharacterIncrement, MultiLoRALinear, canonical_mix, valid_alias, validate_rank_alpha


ARCHITECTURE_ID = "h3ce_spatial_refiner_v2"
TARGET_SUFFIXES = ("ffn.in_proj", "ffn.out_proj")
SHOT_COEFFICIENTS = {
    "close": {"fullbody": 0.0, "face": 1.0},
    "medium": {"fullbody": 0.7, "face": 0.6},
    "far": {"fullbody": 1.0, "face": 0.0},
}


def expected_target_shapes() -> dict[str, tuple[int, int]]:
    """Contract dimensions are (out_features, in_features), checked against the tree."""
    result = {}
    blocks = [(f"encoders.{stage}.{block}", width) for stage, width in enumerate((128, 192)) for block in range(2)]
    blocks += [(f"bottleneck.{block}", 256) for block in range(4)]
    blocks += [(f"decoders.{stage}.{block}", width) for stage, width in enumerate((192, 128)) for block in range(2)]
    for prefix, width in blocks:
        result[f"{prefix}.ffn.in_proj"] = (2 * width, width)
        result[f"{prefix}.ffn.out_proj"] = (width, 2 * width)
    return dict(sorted(result.items()))


def resolve_character_targets(model: nn.Module) -> tuple[str, ...]:
    if getattr(model, "architecture_id", None) != ARCHITECTURE_ID:
        raise H3CEError("E_LORA_MODULE_CONTRACT", "Character adapters require SpatialRefinerV2")
    expected = expected_target_shapes()
    actual = {name: module for name, module in model.named_modules() if name.endswith(TARGET_SUFFIXES)}
    if set(actual) != set(expected):
        raise H3CEError("E_LORA_MODULE_CONTRACT", "Spatial FFN target module tree differs from the architecture contract")
    for name, module in actual.items():
        if type(module) not in (nn.Linear, MultiLoRALinear) or (module.out_features, module.in_features) != expected[name]:
            raise H3CEError("E_LORA_MODULE_CONTRACT", f"Unexpected spatial FFN projection: {name}")
    return tuple(expected)


class CharacterLoRAStack:
    """Adapter manager; it neither trains nor registers a base as accepted.

    A later trainer must require a useful, accepted base before calling its own
    optimizer. This manager freezes the supplied refiner, including scene context;
    it cannot freeze a separately owned VAE and does not claim to do so.
    """

    def __init__(self, model: nn.Module) -> None:
        targets = resolve_character_targets(model)
        existing = [isinstance(model.get_submodule(name), MultiLoRALinear) for name in targets]
        if any(existing) and not all(existing):
            raise H3CEError("E_LORA_MODULE_CONTRACT", "Partially wrapped character target tree is not supported")
        self.model, self.target_modules = model, targets
        if not any(existing):
            wrappers = {name: MultiLoRALinear(model.get_submodule(name)) for name in targets}
            for name, layer in wrappers.items():
                parent, child = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), child, layer)
        self.layers = {name: model.get_submodule(name) for name in targets}
        adapter_sets = {tuple(sorted(layer.adapters)) for layer in self.layers.values()}
        if len(adapter_sets) != 1:
            raise H3CEError("E_LORA_MODULE_CONTRACT", "Inconsistent adapter sets across spatial layers")
        self.model.requires_grad_(False)
        for layer in self.layers.values():
            layer.trainable_alias = None
        self.set_mix(())

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(next(iter(self.layers.values())).adapters))

    def add_adapter(self, name: str, *, rank: int = 8, alpha: float = 8.0) -> None:
        valid_alias(name)
        validate_rank_alpha(rank, alpha)
        if any(name in layer.adapters for layer in self.layers.values()):
            raise H3CEError("E_LORA_CONTRACT", f"Adapter already exists: {name}")
        # Allocate all increments before attaching any, so an allocation failure
        # cannot leave a subset of the refiner carrying the new adapter alias.
        prepared = {target: CharacterIncrement(layer.base, rank=rank, alpha=alpha)
                    for target, layer in self.layers.items()}
        for target, layer in self.layers.items():
            layer.adapters[name] = prepared[target]

    def set_mix(self, mix) -> None:
        selected = canonical_mix(mix)
        if any(name not in layer.adapters for layer in self.layers.values() for name, _ in selected):
            raise H3CEError("E_LORA_CONTRACT", "Mix contains an adapter missing from spatial targets")
        if any(layer.trainable_alias is not None and selected != ((layer.trainable_alias, 1.0),) for layer in self.layers.values()):
            raise H3CEError("E_LORA_TRAINING_MIX", "Character training activates only the selected adapter at strength 1")
        for layer in self.layers.values():
            layer.set_mix(selected, warn=False)

    def select_trainable_adapter(self, name: str | None) -> tuple[nn.Parameter, ...]:
        if name is not None and any(name not in layer.adapters for layer in self.layers.values()):
            raise H3CEError("E_LORA_CONTRACT", f"Unknown character adapter: {name}")
        self.model.requires_grad_(False)
        for layer in self.layers.values():
            layer.select_trainable(name)
        return tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def adapter_state(self, name: str) -> dict[str, torch.Tensor]:
        if name not in self.aliases:
            raise H3CEError("E_LORA_CONTRACT", f"Unknown character adapter: {name}")
        return {f"{target}.{matrix}": getattr(layer.adapters[name], matrix).detach().cpu().contiguous().clone()
                for target, layer in self.layers.items() for matrix in ("A", "B")}

    def load_adapter_state(self, name: str, state: Mapping[str, torch.Tensor]) -> None:
        """Validate every tensor before changing any parameter; metadata is checked by load_adapter."""
        expected = self.adapter_state(name)
        if set(state) != set(expected):
            raise H3CEError("E_LORA_STATE", "Adapter tensor names differ from the exact spatial targets")
        parameters = {f"{target}.{matrix}": getattr(layer.adapters[name], matrix)
                      for target, layer in self.layers.items() for matrix in ("A", "B")}
        converted = {}
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.layout != torch.strided or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise H3CEError("E_LORA_STATE", f"Invalid adapter tensor: {key}")
            parameter = parameters[key]
            try:
                converted[key] = tensor.detach().to(device=parameter.device, dtype=parameter.dtype).contiguous()
            except Exception as exc:
                raise H3CEError("E_LORA_STATE", f"Cannot convert adapter tensor: {key}", str(exc)) from exc
            if not bool(torch.isfinite(converted[key]).all()):
                raise H3CEError("E_LORA_STATE", f"Adapter tensor exceeds the parameter dtype range: {key}")
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(converted[key])


def route_mix(user_mix, adapter_modes: Mapping[str, str], *, shot: str | None = None) -> tuple[tuple[str, float], ...]:
    """Manual default; optional shot coefficients multiply, never replace, user weights.

    This function is stateless. Video callers must call it at chunk boundaries and
    keep the result fixed for that chunk; a video scheduler is not implemented here.
    """
    selected = canonical_mix(user_mix)
    if shot is not None and shot not in SHOT_COEFFICIENTS:
        raise H3CEError("E_LORA_ROUTING", "Shot must be close, medium or far")
    for name, _ in selected:
        if adapter_modes.get(name) not in ("fullbody", "face"):
            raise H3CEError("E_LORA_ROUTING", f"Missing or invalid training mode for {name}")
    return tuple((name, strength * (1.0 if shot is None else SHOT_COEFFICIENTS[shot][adapter_modes[name]])) for name, strength in selected)
