"""Spatial character LoRA primitives; no trainer or accepted base is created here."""

from .layers import CharacterIncrement, MultiLoRALinear, canonical_mix
from .mixer import CharacterLoRAStack, resolve_character_targets, route_mix
from .registry import AdapterMetadata, CompatibilityContext, assert_compatible, load_adapter, read_adapter, save_adapter

__all__ = [
    "AdapterMetadata", "CharacterIncrement", "CharacterLoRAStack", "CompatibilityContext",
    "MultiLoRALinear", "assert_compatible", "canonical_mix", "load_adapter", "read_adapter",
    "resolve_character_targets", "route_mix", "save_adapter",
]
