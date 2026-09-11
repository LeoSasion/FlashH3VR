"""One shared low-rank increment on exact existing H3 decoder output projections."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from h3ce.errors import H3CEError


class DecoderOutputLoRA(nn.Module):
    def __init__(self, base, *, rank=4, alpha=4.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return self.base(x) + delta.to(x.dtype) * (self.alpha / self.rank)


def attach_decoder_adapter(model, *, last_n_blocks=4, rank=4, alpha=4.0):
    if not isinstance(last_n_blocks, int) or isinstance(last_n_blocks, bool) or not 1 <= last_n_blocks <= 36:
        raise H3CEError("E_H3_MODULE_CONTRACT", "Select 1..36 exact existing decoder blocks")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1 or not math.isfinite(alpha):
        raise H3CEError("E_H3_MODULE_CONTRACT", "Invalid decoder adapter rank/alpha")
    blocks = model.decoder.transformer_blocks
    if len(blocks) != 36:
        raise H3CEError("E_H3_MODULE_CONTRACT", "Expected exactly 36 decoder blocks")
    if any(isinstance(m, DecoderOutputLoRA) for m in model.modules()):
        raise H3CEError("E_H3_MODULE_CONTRACT", "Only one shared native decoder adapter is supported")
    names = [f"decoder.transformer_blocks.{i}.attn.to_out" for i in range(36-last_n_blocks, 36)]
    for name in names:
        layer = model.get_submodule(name)
        if type(layer) is not nn.Linear or (layer.in_features, layer.out_features) != (2048, 2048):
            raise H3CEError("E_H3_MODULE_CONTRACT", f"Unexpected target {name}")
    model.requires_grad_(False)
    for name in names:
        parent, child = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        setattr(module, child, DecoderOutputLoRA(getattr(module, child), rank=rank, alpha=alpha))
    return tuple(names)


def adapter_fingerprint(model):
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, DecoderOutputLoRA):
            digest.update(f"{name}:{module.rank}:{module.alpha}".encode())
            for parameter in (module.A, module.B):
                digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
