# Publication change 2026-09-12: move machine-local receipt to relative runtime lock.
"""Explicit frozen upstream Comfy Kitchen INT8 linear path with isolated cuBLAS13."""
from functools import lru_cache
from pathlib import Path
import os,json,ctypes
import torch
from torch import nn
from h3ce.components import sha256_file
from h3ce.errors import H3CEError
from .convrot_cuda import runtime as quantizer_runtime
from .int8_convrot_backend import Int8ConvRotH3Backend,ConvRotInt8Linear

@lru_cache(maxsize=1)
def runtime():
    root=Path(__file__).resolve().parents[2]
    receipt=root/'configs/cublas13_runtime.lock.json'
    lock=json.loads(receipt.read_text(encoding='utf-8'));base=root/lock['destination']
    if lock['status']!='downloaded_sha256_verified_and_isolated_extracted' or base!=root/'third_party/h3_cublas13_runtime':
        raise H3CEError('E_COMPONENT_LOCK','Unexpected cuBLAS13 acquisition')
    for path,expected in lock['files'].items():
        if sha256_file(base/path)!=expected:raise H3CEError('E_COMPONENT_CODE','Isolated cuBLAS13 runtime changed')
    libraries=list(base.rglob('cublasLt64_13.dll'))
    if len(libraries)!=1:raise H3CEError('E_CUDA_REQUIRED','Expected exactly one locked cuBLASLt13 library')
    directory_handle=os.add_dll_directory(str(libraries[0].parent))
    # The extension uses LoadLibraryA rather than LoadLibraryEx search flags.
    # Preload the verified absolute DLL so its basename lookup sees that module.
    handle=(directory_handle,ctypes.WinDLL(str(libraries[0])))
    cuda,extension_hash=quantizer_runtime()
    if not cuda._CUBLASLT_AVAILABLE:
        raise H3CEError('E_CUDA_REQUIRED','Full Comfy Kitchen CUDA INT8 backend unavailable; use a fresh process with the isolated library')
    return cuda,handle,{'acquisition_sha256':sha256_file(receipt),'extension_lock_sha256':extension_hash}

class KitchenInt8Linear(nn.Module):
    def __init__(self,original):
        super().__init__();self.in_features=original.in_features;self.out_features=original.out_features;self.calls=0
        self.register_buffer('weight',original.weight.contiguous())
        self.register_buffer('scale',original.scale.reshape(-1).contiguous())
        self.register_buffer('bias',original.bias)
    def forward(self,x):
        if torch.is_grad_enabled() and x.requires_grad:
            raise H3CEError('E_H3_GRADIENT_CONTRACT','Upstream INT8 is frozen inference/conditioning only; no surrogate backward')
        if not x.is_cuda or x.dtype!=torch.float16:
            raise H3CEError('E_H3_MODULE_CONTRACT','Current upstream INT8 experiment requires CUDA FP16 activations')
        self.calls+=1;cuda,_,_=runtime()
        return cuda.int8_linear(x,self.weight,self.scale,self.bias,out_dtype=torch.float16,convrot=True,convrot_groupsize=256)

class Int8KitchenH3Backend(Int8ConvRotH3Backend):
    @classmethod
    def from_verified_file(cls,*a,**kw):
        raise H3CEError('E_COMPONENT_LOCK','Use from_locked for both upstream runtime and source weights')
    @classmethod
    def from_locked(cls,**kwargs):
        if kwargs.pop('compute_mode','int8_mm')!='int8_mm':raise H3CEError('E_CONFIG','Upstream runtime uses explicit INT8 compute')
        _,_,identity=runtime()
        base=Int8ConvRotH3Backend.from_locked(**kwargs,compute_mode='int8_mm')
        result=cls.__new__(cls);result.__dict__.update(base.__dict__);result.runtime_identity=identity
        names=[n for n,m in result.model.named_modules() if isinstance(m,ConvRotInt8Linear)]
        if len(names)!=144:raise H3CEError('E_H3_MODULE_CONTRACT','Expected 144 pretrained quantized decoder layers')
        for name in names:
            parent,leaf=name.rsplit('.',1);setattr(result.model.get_submodule(parent),leaf,KitchenInt8Linear(result.model.get_submodule(name)))
        return result
    def numerical_contract(self):
        result=super().numerical_contract()
        result['precision']=dict(result['precision'],compute_mode='comfy_kitchen_cuda_int8')
        result['upstream_int8']={'runtime':self.runtime_identity,'backend_sha256':sha256_file(__file__),
            'arithmetic':'Pinned upstream int8_linear: fused ConvRot quantization + CUTLASS INT8/dequant epilogue or cuBLAS INT8/separate CUDA dequant; no eager fallback',
            'scope':'Frozen conditioning/inference only; no claim universal zero/tiny-input support before validation'}
        return result
