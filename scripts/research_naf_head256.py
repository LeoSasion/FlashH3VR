"""Pretrained two-level NAFNet tail with an exact same-shape frozen reference."""
import copy
import torch
from scripts.research_head_tail2 import strict_spatial

class Tail(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        assert len(base.decoders)==4 and len(base.decoders[-2])==len(base.decoders[-1])==1
        assert base.ending.in_channels==32 and base.ending.out_channels==3
        self.coarse=copy.deepcopy(base.decoders[-2]);self.up=copy.deepcopy(base.ups[-1])
        self.fine=copy.deepcopy(base.decoders[-1]);self.output=copy.deepcopy(base.ending)
    def forward(self,features,skip):
        return self.output(self.fine(self.up(self.coarse(features))+skip))

class NAFHead(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.backbone=base.eval().requires_grad_(False)
        self.tail=Tail(base).float().requires_grad_(True)
        self.reference=copy.deepcopy(self.tail).eval().requires_grad_(False)
        assert sum(v.numel() for v in self.tail.parameters())==48067
    def features(self,condition):
        feat=[];skip=[];baseline=[]
        hs=[self.backbone.decoders[-2].register_forward_pre_hook(lambda m,a:feat.append(a[0].detach())),
            self.backbone.encoders[0].register_forward_hook(lambda m,a,y:skip.append(y.detach())),
            self.backbone.ending.register_forward_hook(lambda m,a,y:baseline.append(y.detach()))]
        try:
            with torch.no_grad(),strict_spatial():self.backbone(condition)
        finally:
            for h in hs:h.remove()
        assert len(feat)==len(skip)==len(baseline)==1
        return {'features':feat[0],'skip':skip[0],'official_ending':baseline[0]}
    def delta(self,features,skip,parameters=None):
        with torch.no_grad():negative=self.reference(features,skip)
        positive=(self.tail(features,skip) if parameters is None else
                  torch.func.functional_call(self.tail,parameters,(features,skip)))
        return positive-negative
