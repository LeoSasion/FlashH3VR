"""Strict character artifact metadata and atomic, tensor-only persistence.

These helpers do not create a base acceptance record. A matching hash is an
identity check, not evidence that a base restores images or a codec was evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from h3ce.cache.store import assert_no_links, atomic_write
from h3ce.errors import H3CEError
from .layers import validate_rank_alpha
from .mixer import ARCHITECTURE_ID, CharacterLoRAStack, expected_target_shapes


def _sha(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise H3CEError("E_LORA_METADATA", f"{label} must be a lowercase SHA-256")


def _identity(architecture_id: str, encoder_contract_id: str, target_modules: tuple[str, ...]) -> None:
    if architecture_id != ARCHITECTURE_ID:
        raise H3CEError("E_LORA_METADATA", "Only h3ce_spatial_refiner_v2 character adapters are supported")
    if not isinstance(encoder_contract_id, str) or not encoder_contract_id.strip():
        raise H3CEError("E_LORA_METADATA", "encoder_contract_id is required")
    if not isinstance(target_modules, tuple) or target_modules != tuple(expected_target_shapes()):
        raise H3CEError("E_LORA_METADATA", "target_modules must be the sorted, complete spatial FFN module list")


@dataclass(frozen=True)
class AdapterMetadata:
    kind: str
    base_sha: str
    architecture_id: str
    encoder_contract_id: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    training_mode: str
    provenance: Mapping[str, Any]
    validated_decoder_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "character_lora":
            raise H3CEError("E_LORA_METADATA", "Artifact kind must be character_lora, never native H3 DiT or codec LoRA")
        _sha(self.base_sha, "base_sha")
        _identity(self.architecture_id, self.encoder_contract_id, self.target_modules)
        validate_rank_alpha(self.rank, self.alpha)
        if self.training_mode not in ("fullbody", "face"):
            raise H3CEError("E_LORA_METADATA", "training_mode must be fullbody or face")
        if not isinstance(self.provenance, dict) or not self.provenance or any(not isinstance(key, str) for key in self.provenance):
            raise H3CEError("E_LORA_METADATA", "A nonempty provenance object is required")
        try:
            json.dumps(self.provenance, allow_nan=False, sort_keys=True)
        except (ValueError, TypeError) as exc:
            raise H3CEError("E_LORA_METADATA", "provenance must contain finite JSON data") from exc
        if not isinstance(self.validated_decoder_hashes, tuple) or len(set(self.validated_decoder_hashes)) != len(self.validated_decoder_hashes):
            raise H3CEError("E_LORA_METADATA", "validated_decoder_hashes must be a tuple without duplicates")
        for decoder_hash in self.validated_decoder_hashes:
            _sha(decoder_hash, "validated_decoder_hash")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "kind": self.kind, "base_sha": self.base_sha, "architecture_id": self.architecture_id,
            "encoder_contract_id": self.encoder_contract_id, "target_modules": list(self.target_modules),
            "rank": self.rank, "alpha": float(self.alpha), "training_mode": self.training_mode,
            "provenance": json.loads(json.dumps(self.provenance, allow_nan=False)),
            "validated_decoder_hashes": list(self.validated_decoder_hashes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdapterMetadata":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise H3CEError("E_LORA_METADATA", "Character metadata has missing or unknown fields")
        values = dict(value)
        for name in ("target_modules", "validated_decoder_hashes"):
            if not isinstance(values[name], (tuple, list)) or any(not isinstance(item, str) for item in values[name]):
                raise H3CEError("E_LORA_METADATA", f"{name} must be an array of strings")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class CompatibilityContext:
    base_sha: str
    architecture_id: str
    encoder_contract_id: str
    target_modules: tuple[str, ...]
    decoder_hash: str

    def __post_init__(self) -> None:
        _sha(self.base_sha, "base_sha")
        _sha(self.decoder_hash, "decoder_hash")
        _identity(self.architecture_id, self.encoder_contract_id, self.target_modules)


def assert_compatible(metadata: AdapterMetadata, context: CompatibilityContext) -> None:
    metadata.__post_init__()
    context.__post_init__()
    mismatched = [name for name in ("base_sha", "architecture_id", "encoder_contract_id", "target_modules")
                  if getattr(metadata, name) != getattr(context, name)]
    if mismatched:
        raise H3CEError("E_LORA_COMPATIBILITY", "Character adapter identity does not match the active base", mismatched)
    if context.decoder_hash not in metadata.validated_decoder_hashes:
        raise H3CEError("E_LORA_CODEC_VALIDATION", "Character adapter has no recorded validation for the active decoder hash")


def _validate_state(state: Mapping[str, torch.Tensor], metadata: AdapterMetadata) -> None:
    shapes = expected_target_shapes()
    expected = {f"{target}.{matrix}": ((metadata.rank, input_width) if matrix == "A" else (output_width, metadata.rank))
                for target, (output_width, input_width) in shapes.items() for matrix in ("A", "B")}
    if set(state) != set(expected):
        raise H3CEError("E_LORA_STATE", "Artifact must contain exactly A/B tensors for the spatial FFN targets")
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise H3CEError("E_LORA_STATE", f"Invalid float32 character adapter tensor: {name}")


def save_adapter(path: Path, stack: CharacterLoRAStack, alias: str, metadata: AdapterMetadata) -> str:
    """Save an artifact, including an unvalidated candidate, without accepting it.

    Empty validated_decoder_hashes is permitted for a training candidate; the
    compatibility-checked loader will refuse to activate such an artifact.
    """
    from safetensors.torch import save
    metadata.__post_init__()
    if tuple(stack.target_modules) != metadata.target_modules:
        raise H3CEError("E_LORA_METADATA", "Artifact targets differ from the installed spatial stack")
    if alias not in stack.aliases:
        raise H3CEError("E_LORA_CONTRACT", f"Unknown character adapter: {alias}")
    if any(layer.adapters[alias].rank != metadata.rank or layer.adapters[alias].alpha != metadata.alpha for layer in stack.layers.values()):
        raise H3CEError("E_LORA_METADATA", "Artifact rank/alpha differs from the actual adapter")
    state = stack.adapter_state(alias)
    _validate_state(state, metadata)
    path = Path(path)
    if path.suffix != ".safetensors":
        raise H3CEError("E_LORA_STATE", "Character artifacts must use .safetensors")
    assert_no_links(path)
    payload = save(state, metadata={"h3ce_character_metadata": json.dumps(metadata.to_dict(), sort_keys=True, allow_nan=False)})
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def read_adapter(path: Path, context: CompatibilityContext) -> tuple[AdapterMetadata, dict[str, torch.Tensor]]:
    """Read the same bytes for metadata and tensors; never deserialize executable pickle."""
    from safetensors.torch import load
    path = Path(path)
    assert_no_links(path)
    try:
        payload = path.read_bytes()
        if len(payload) < 8:
            raise ValueError("Missing safetensors header")
        header_length = int.from_bytes(payload[:8], "little")
        if header_length < 2 or header_length > min(len(payload) - 8, 1_000_000):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(payload[8:8 + header_length])
        metadata = AdapterMetadata.from_dict(json.loads(header["__metadata__"]["h3ce_character_metadata"]))
        assert_compatible(metadata, context)
        state = load(payload)
        _validate_state(state, metadata)
    except H3CEError:
        raise
    except Exception as exc:
        raise H3CEError("E_LORA_STATE", f"Cannot read character safetensors artifact: {path}", str(exc)) from exc
    return metadata, state


def load_adapter(path: Path, stack: CharacterLoRAStack, alias: str, context: CompatibilityContext) -> AdapterMetadata:
    """Validate identity/state before installing a new frozen, initially inactive adapter."""
    metadata, state = read_adapter(path, context)
    if tuple(stack.target_modules) != metadata.target_modules:
        raise H3CEError("E_LORA_COMPATIBILITY", "Loaded adapter targets differ from installed stack")
    stack.add_adapter(alias, rank=metadata.rank, alpha=metadata.alpha)
    try:
        stack.load_adapter_state(alias, state)
    except BaseException:
        # The alias did not exist before add_adapter. Remove every newly attached
        # increment even if conversion, allocation, copying or interruption fails.
        # Existing adapter weights, selection and mix are never touched here.
        for layer in stack.layers.values():
            if alias in layer.adapters:
                del layer.adapters[alias]
        raise
    return metadata
