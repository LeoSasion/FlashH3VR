"""Explicit stage dependencies; decoder changes never invalidate encoder latents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h3ce.errors import H3CEError


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stage_key(stage: str, **dependencies) -> str:
    return digest({"stage": stage, "dependencies": dependencies})


def latent_key(view_key: str, encoder_contract_id: str) -> str:
    return stage_key("latent", view_key=view_key, encoder_contract_id=encoder_contract_id)


def effective_decoder_hash(*, base_decoder_weights: str, adapter_weights: str | None,
                           adapter_scale: float, code: str, precision: str, tile_contract) -> str:
    return digest(locals())


def decoded_key(latent: str, decoder_hash: str, *, trainable: bool = False) -> str:
    if trainable:
        raise H3CEError("E_CACHE_TRAINABLE_DECODER", "Trainable Decoder outputs cannot be cached.")
    return stage_key("decoded", latent=latent, effective_decoder_hash=decoder_hash)

