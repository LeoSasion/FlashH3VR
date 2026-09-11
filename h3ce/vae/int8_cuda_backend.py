"""Frozen H3 W8A8 with fused ConvRot, original INT8 GEMM and native video path."""
import torch
from torch import nn
from h3ce.components import sha256_file
from h3ce.errors import H3CEError
from .int8_convrot_backend import Int8ConvRotH3Backend,ConvRotInt8Linear
from .convrot_cuda import quantize_fused,runtime


class FusedConvRotInt8Linear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_features,self.out_features=original.in_features,original.out_features
        for name in ('weight','weight_t','scale','bias'):
            self.register_buffer(name,getattr(original,name),persistent=name!='weight_t')
        self.calls=0

    def forward(self,x):
        if torch.is_grad_enabled() and x.requires_grad:
            raise H3CEError('E_H3_GRADIENT_CONTRACT','Fused W8A8 is frozen conditioning/inference; use dequant_fp16 for Decoder gradients')
        self.calls+=1;shape=x.shape
        with torch.autocast(device_type='cuda',enabled=False):
            q,scale=quantize_fused(x.half().reshape(-1,self.in_features).contiguous())
            rows=len(q)
            if rows%32:q=torch.nn.functional.pad(q,(0,0,0,32-rows%32))
            accum=torch._int_mm(q,self.weight_t)[:rows]
            out=(accum.float()*(scale*self.scale)).half()+self.bias
        return out.reshape(*shape[:-1],self.out_features)


class Int8CudaH3Backend(Int8ConvRotH3Backend):
    @classmethod
    def from_verified_file(cls,*args,**kwargs):
        raise H3CEError('E_COMPONENT_LOCK','Use from_locked for the fused runtime and weight contract')

    @classmethod
    def from_locked(cls,**kwargs):
        if kwargs.pop('compute_mode','int8_mm')!='int8_mm':
            raise H3CEError('E_CONFIG','Fused backend is an explicit W8A8 contract')
        _,runtime_hash=runtime()
        base=Int8ConvRotH3Backend.from_locked(**kwargs,compute_mode='int8_mm')
        result=cls.__new__(cls);result.__dict__.update(base.__dict__);result.runtime_hash=runtime_hash
        names=[n for n,m in result.model.named_modules() if isinstance(m,ConvRotInt8Linear)]
        if len(names)!=144:raise H3CEError('E_H3_MODULE_CONTRACT','Expected exactly 144 INT8 layers')
        for name in names:
            parent,leaf=name.rsplit('.',1)
            setattr(result.model.get_submodule(parent),leaf,FusedConvRotInt8Linear(result.model.get_submodule(name)))
        return result

    def numerical_contract(self):
        result=super().numerical_contract()
        result['precision']=dict(result['precision'],compute_mode='int8_mm_cuda_fused_safezero')
        result['fused_quantizer']={'runtime_lock_sha256':self.runtime_hash,
            'source_sha256':sha256_file(__import__(quantize_fused.__module__,fromlist=['__file__']).__file__),
            'backend_sha256':sha256_file(__file__),
            'arithmetic':'FP32 regular-H4 staged rotation and absmax; FP16 numerator/divisor/division before quantization; half-underflow rows forced to zero'}
        return result
