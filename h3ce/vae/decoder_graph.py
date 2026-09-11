"""Explicitly prepared frozen decoder CUDA graph; encoder remains native serial.

Preparation has one real warmup forward and one graph recording. New shape,
weights, device or precision requires a new graph. Replays copy fresh inputs and
clone the result so a later call cannot overwrite an earlier returned image.
"""
import torch
from h3ce.components import sha256_file
from h3ce.errors import H3CEError
from .aitoolkit_h3_backend import AIToolkitH3Backend


class FrozenDecoderGraphBackend(AIToolkitH3Backend):
    def __init__(self,eager):
        self.eager=eager;self.model=eager.model
        self.weight_sha256=eager.weight_sha256;self.source_sha256=eager.source_sha256
        self.graph=None;self.replays=0;self._shape=None

    def _versions(self):
        return tuple((name,id(t),t._version,str(t.device),str(t.dtype),bool(t.requires_grad))
                     for name,t in list(self.model.named_parameters())+list(self.model.named_buffers()))

    @torch.no_grad()
    def prepare(self,raw):
        if self.graph is not None:raise H3CEError('E_H3_MODULE_CONTRACT','Graph already prepared; create a new wrapper explicitly')
        if any(t.requires_grad for t in self.model.parameters()):raise H3CEError('E_H3_GRADIENT_CONTRACT','CUDA decoder graph requires frozen parameters')
        if not raw.is_cuda or raw.requires_grad or raw.dtype!=torch.float32:
            raise H3CEError('E_H3_FRAME_CONTRACT','Prepare from detached CUDA FP32 raw latents')
        self._shape=(tuple(raw.shape),raw.device,raw.dtype)
        self._input=raw.detach().clone();self._state=self._versions()
        stream=torch.cuda.Stream(device=raw.device);stream.wait_stream(torch.cuda.current_stream(raw.device))
        with torch.cuda.stream(stream):
            warmup=self.eager.decode_raw(self._input)
        stream.synchronize();del warmup
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            output=self.eager.decode_raw(self._input)
        torch.cuda.current_stream(raw.device).wait_stream(stream)
        self.graph=graph;self._output=output

    def decode_raw(self,raw):
        if torch.is_grad_enabled():raise H3CEError('E_H3_GRADIENT_CONTRACT','CUDA decoder graph is frozen inference/conditioning only')
        if self.graph is None:raise H3CEError('E_H3_MODULE_CONTRACT','Prepare the decoder graph explicitly before replay')
        if (tuple(raw.shape),raw.device,raw.dtype)!=self._shape:
            raise H3CEError('E_H3_FRAME_CONTRACT','Decoder graph shape/device/dtype differs; no automatic recapture')
        if self._versions()!=self._state:
            raise H3CEError('E_CODEC_COMPATIBILITY','Decoder graph tensor identity/version/precision changed')
        self._input.copy_(raw);self.graph.replay();self.replays+=1
        return self._output.clone()

    def numerical_contract(self):
        result=self.eager.numerical_contract()
        result['precision']=dict(result['precision'],execution='frozen_cuda_graph')
        result['decoder_graph_source_sha256']=sha256_file(__file__)
        return result

    def encoder_contract_id(self):
        # Decoder execution alone does not alter encoded latents.
        return self.eager.encoder_contract_id()
