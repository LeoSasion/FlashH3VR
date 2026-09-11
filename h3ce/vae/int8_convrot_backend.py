"""Pinned INT8 ConvRot asset on the existing H3 native video implementation.

Two explicit numerical contracts: differentiable dequantized FP16, or frozen
W8A8 inference. No automatic precision fallback. The latter uses real INT8 GEMM
and rejects input gradients; it must not silently disconnect a training graph.
"""
from pathlib import Path
import json
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

from h3ce.components import canonical_hash, read_component_lock, sha256_file
from h3ce.errors import H3CEError
from . import _vendor
from .aitoolkit_h3_backend import AIToolkitH3Backend, VENDOR_SHA256

WEIGHT_SHA256 = '9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410'
REVISION = 'f4cac997f880e93cf6940af61ee8d58ef31ff7f3'
MODES = ('dequant_fp16', 'int8_mm')
QUANT_CONFIG = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}


def regular_hadamard(device, dtype=torch.float32):
    """Regular H4 Kronecker basis used by Comfy Kitchen ConvRot, not Sylvester H2."""
    h4 = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], device=device, dtype=dtype)
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    return h / 16


def dequantize_weight(weight, scale):
    with torch.autocast(device_type=weight.device.type, enabled=False):
        h = regular_hadamard(weight.device)
        return ((weight.float() * scale).reshape(-1,256) @ h.T).reshape(weight.shape)


class ConvRotInt8Linear(nn.Module):
    """Frozen W8A8. Buffer storage and GEMM remain INT8; accumulation is INT32."""
    def __init__(self, weight, scale, bias):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        packed = weight.T.contiguous()
        self.register_buffer('weight', packed.T)
        self.register_buffer('weight_t', packed, persistent=False)
        self.register_buffer('scale', scale.float().reshape(1,-1))
        self.register_buffer('bias', bias.half())
        self.register_buffer('rotation', regular_hadamard(weight.device, torch.float16), persistent=False)
        self.calls = 0

    def forward(self, x):
        if torch.is_grad_enabled() and x.requires_grad:
            raise H3CEError('E_H3_GRADIENT_CONTRACT', 'W8A8 has no surrogate backward; select dequant_fp16 explicitly for decoder gradients')
        if not x.is_cuda:
            raise H3CEError('E_CUDA_REQUIRED', 'INT8 GEMM mode requires CUDA; no floating fallback')
        self.calls += 1
        shape = x.shape
        with torch.autocast(device_type='cuda', enabled=False):
            x = (x.half().reshape(-1,256) @ self.rotation).reshape(-1,self.in_features)
            scale = (x.abs().amax(-1,keepdim=True).float()/127).clamp_min(1e-30)
            divisor = scale.half()
            divisor = torch.where(divisor == 0, torch.full_like(divisor,torch.finfo(torch.float16).tiny),divisor)
            q = (x/divisor).round().clamp(-128,127).to(torch.int8)
            rows = len(q)
            if rows % 32:
                q = torch.nn.functional.pad(q,(0,0,0,32-rows%32))
            accum = torch._int_mm(q, self.weight_t)[:rows]
            out = (accum.float() * (scale * self.scale)).half() + self.bias
        return out.reshape(*shape[:-1],self.out_features)


class Int8ConvRotH3Backend(AIToolkitH3Backend):
    @classmethod
    def from_locked(cls, *, project_root, weights, components_lock, device='cuda', compute_mode='dequant_fp16'):
        root=Path(project_root).resolve(); entries=read_component_lock(root/components_lock)
        entry=entries['h3_visual_vae']; backend=entries['aitoolkit_h3_backend']
        requested=(root/weights).resolve()
        if (requested != (root/entry['local_path']).resolve() or entry['sha256'] != WEIGHT_SHA256
                or entry['revision'] != REVISION or backend['sha256'] != VENDOR_SHA256):
            raise H3CEError('E_COMPONENT_LOCK','INT8 asset or native source differs from the pinned lock')
        return cls.from_verified_file(requested,device=device,compute_mode=compute_mode)

    @classmethod
    def from_verified_file(cls, path, *, device='cuda', compute_mode='dequant_fp16'):
        if compute_mode not in MODES:
            raise H3CEError('E_CONFIG','Unknown INT8 compute mode')
        if sha256_file(path) != WEIGHT_SHA256 or sha256_file(_vendor.__file__) != VENDOR_SHA256:
            raise H3CEError('E_COMPONENT_HASH','INT8 asset or native source hash mismatch')
        with safe_open(str(path),framework='pt') as handle:
            metadata=json.loads(handle.metadata()['minimax_h3_video_vae'])
        if metadata['vae_clip_length'] != 17 or metadata['vae_token_drop'] != 3:
            raise H3CEError('E_H3_MODULE_CONTRACT','INT8 time metadata differs')
        state=load_file(str(path),device='cpu')
        for name in ('latents_mean','latents_std'):
            value=state.pop(name)
            if value.dtype != torch.float32 or not torch.equal(value,torch.tensor(metadata[name],dtype=torch.float32)):
                raise H3CEError('E_H3_MODULE_CONTRACT','Tensor and metadata normalization disagree')
        with torch.device('meta'):
            model=_vendor.MiniMaxH3VideoVAE()
        # Native constructor creates only small nonpersistent buffers; recreate them
        # from the same expressions by constructing the module tree on meta then
        # generating each known buffer explicitly (strictly verify all names).
        expected_buffers={'latents_mean','latents_std','pixel_mean','pixel_std','decoder.rope.inv_freq','decoder.mask_token'}
        actual_buffers=dict(model.named_buffers())
        if set(actual_buffers) != expected_buffers:
            raise H3CEError('E_H3_MODULE_CONTRACT',f'Unexpected native buffers: {list(actual_buffers)}')
        stats={'latents_mean':metadata['latents_mean'],'latents_std':metadata['latents_std'],
               'pixel_mean':_vendor.IMAGENET_MEAN,'pixel_std':_vendor.IMAGENET_STD}
        for name,values in stats.items():
            value=torch.tensor(values,dtype=torch.float32,device=device)
            if name.startswith('pixel'):value=value.view(1,3,1,1,1)
            setattr(model,name,value)
        model.decoder.rope = _vendor.RotaryEmbedding3d(48,theta=100.0).to(device)
        expected=model.state_dict()
        quant_names=[f'decoder.transformer_blocks.{i}.{suffix}' for i in range(36)
                     for suffix in ('attn.to_qkv','attn.to_out','ff.w1','ff.w2')]
        quant_keys={n+s for n in quant_names for s in ('.comfy_quant','.weight_scale')}
        if set(state) != set(expected)|quant_keys:
            raise H3CEError('E_H3_MODULE_CONTRACT','INT8 tensor keys differ from exact native model')
        for name,tensor in expected.items():
            if state[name].shape != tensor.shape:
                raise H3CEError('E_H3_MODULE_CONTRACT',f'INT8 shape differs: {name}')
        # Set each original parameter explicitly; no missing/random parameters survive.
        for name,tensor in expected.items():
            if name.removesuffix('.weight') in quant_names and name.endswith('.weight'):
                continue
            if state[name].dtype != torch.float32 or not torch.isfinite(state[name]).all():
                raise H3CEError('E_H3_MODULE_CONTRACT',f'Invalid floating tensor: {name}')
            parent,leaf=name.rsplit('.',1)
            value=state.pop(name).to(device=device,dtype=torch.float16)
            setattr(model.get_submodule(parent),leaf,value if name in actual_buffers else nn.Parameter(value,requires_grad=False))
        for name in quant_names:
            conf=json.loads(state.pop(name+'.comfy_quant').numpy().tobytes())
            weight=state.pop(name+'.weight'); scale=state.pop(name+'.weight_scale')
            if conf != QUANT_CONFIG or weight.dtype != torch.int8 or scale.shape != (weight.shape[0],1) or not torch.isfinite(scale).all() or not (scale>0).all():
                raise H3CEError('E_H3_MODULE_CONTRACT',f'Invalid ConvRot layer: {name}')
            layer=model.get_submodule(name); weight=weight.to(device); scale=scale.to(device)
            if compute_mode=='dequant_fp16':
                layer.weight=nn.Parameter(dequantize_weight(weight,scale).half(),requires_grad=False)
            else:
                parent,leaf=name.rsplit('.',1)
                setattr(model.get_submodule(parent),leaf,ConvRotInt8Linear(weight,scale,layer.bias.detach()))
        if state:raise H3CEError('E_H3_MODULE_CONTRACT','Unconsumed INT8 weights')
        result=cls.__new__(cls); result.compute_mode=compute_mode
        result.model=model.eval().requires_grad_(False);result.weight_sha256=WEIGHT_SHA256
        result.source_sha256=sha256_file(_vendor.__file__)
        if compute_mode=='dequant_fp16':result._validate_modules()
        if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
            raise H3CEError('E_H3_MODULE_CONTRACT','Uninitialized native tensors remain')
        return result

    def numerical_contract(self):
        result=super().numerical_contract()
        result['precision']={'asset':'INT8 ConvRot; 144 decoder linears', 'compute_mode':self.compute_mode,
            'floating_parameters':'FP16 from supplied F32 tensors; native FP32 norm/embed arithmetic preserved',
            'gradients':'ordinary autograd' if self.compute_mode=='dequant_fp16' else 'frozen conditioning only; decoder input gradients rejected',
            'weight_runtime':'dequantized once to FP16' if self.compute_mode=='dequant_fp16' else 'INT8 with INT32 GEMM; rowwise INT8 activations'}
        result['int8_backend_sha256']=sha256_file(__file__)
        return result

    def encoder_contract_id(self):
        return canonical_hash({'native':super().encoder_contract_id(),'int8_backend':sha256_file(__file__),
                               'compute_mode':self.compute_mode})
