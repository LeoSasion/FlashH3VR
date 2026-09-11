"""Pinned optional CUDA ConvRot quantizer with explicit half-underflow handling."""
from functools import lru_cache
from pathlib import Path
import importlib,json,sys
import torch
from h3ce.components import sha256_file
from h3ce.errors import H3CEError


@lru_cache(maxsize=1)
def runtime():
    root=Path(__file__).resolve().parents[2]
    lock_path=root/'configs/convrot_cuda_runtime.lock.json'
    lock=json.loads(lock_path.read_text(encoding='utf-8'));base=root/lock['root']
    for path,expected in lock['files'].items():
        if sha256_file(base/path)!=expected:raise H3CEError('E_COMPONENT_CODE','ConvRot CUDA runtime changed: '+path)
    existing=sys.modules.get('comfy_kitchen')
    if existing is not None and Path(existing.__file__).resolve()!=(base/'comfy_kitchen/__init__.py').resolve():
        raise H3CEError('E_COMPONENT_CODE','Different Comfy Kitchen already imported')
    sys.path.insert(0,str(base))
    cuda=importlib.import_module('comfy_kitchen.backends.cuda')
    if not cuda._EXT_AVAILABLE or not hasattr(cuda._C,'quantize_int8_rowwise_convrot64'):
        raise H3CEError('E_CUDA_REQUIRED','Pinned fused ConvRot extension unavailable; no automatic fallback')
    return cuda,sha256_file(lock_path)


def quantize_fused(x):
    if x.ndim!=2 or x.dtype!=torch.float16 or not x.is_cuda or not x.is_contiguous() or x.shape[1] not in (2048,8192):
        raise H3CEError('E_H3_MODULE_CONTRACT','Fused H3 quantizer requires contiguous CUDA FP16 rows with K=2048/8192')
    cuda,_=runtime()
    if not cuda._convrot_fused_shared_memory_fits(x,x.shape[1],256):
        raise H3CEError('E_CUDA_REQUIRED','Fused ConvRot shared memory exceeds this GPU; no automatic fallback')
    q,scale=cuda.quantize_int8_rowwise_convrot64(x,256)
    # The upstream kernel casts scale to FP16 before division. If it underflows
    # to zero, the old eager path uses the minimum *normal* half divisor; all
    # such tiny rows then quantize to zero. Apply that same result explicitly.
    q.masked_fill_(scale.to(torch.float16)==0,0)
    return q,scale
