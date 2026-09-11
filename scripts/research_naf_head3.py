"""Three existing pretrained NAF decoder levels; no new temporal modules."""
import copy
import torch
from scripts.research_head_tail2 import strict_spatial


class Tail3(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        if len(base.decoders)!=4 or len(base.ups)!=4 or any(len(x)!=1 for x in base.decoders[-3:]):
            raise ValueError('Expected official GoPro32 decoder structure')
        self.blocks=torch.nn.ModuleList(copy.deepcopy(list(base.decoders[-3:])))
        self.ups=torch.nn.ModuleList(copy.deepcopy(list(base.ups[-2:])))
        self.output=copy.deepcopy(base.ending)
    def forward(self,features,skip_mid,skip_full):
        x=self.blocks[0](features)
        x=self.blocks[1](self.ups[0](x)+skip_mid)
        return self.output(self.blocks[2](self.ups[1](x)+skip_full))


class NAFHead3(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.backbone=base.eval().requires_grad_(False)
        self.tail=Tail3(base).float().requires_grad_(True)
        self.reference=copy.deepcopy(self.tail).eval().requires_grad_(False)
    def features(self,condition):
        features=[];middle=[];full=[];ending=[]
        handles=[self.backbone.decoders[-3].register_forward_pre_hook(lambda m,a:features.append(a[0].detach())),
            self.backbone.encoders[1].register_forward_hook(lambda m,a,y:middle.append(y.detach())),
            self.backbone.encoders[0].register_forward_hook(lambda m,a,y:full.append(y.detach())),
            self.backbone.ending.register_forward_hook(lambda m,a,y:ending.append(y.detach()))]
        try:
            with torch.no_grad(),strict_spatial():self.backbone(condition)
        finally:
            for h in handles:h.remove()
        if not all(len(x)==1 for x in (features,middle,full,ending)):raise ValueError('Unexpected native feature flow')
        return {'features':features[0],'skip_mid':middle[0],'skip_full':full[0],'official_ending':ending[0]}
    def delta(self,features,skip_mid,skip_full,parameters=None):
        # Same batch and shape on both tails are intentional for exact initial identity.
        with torch.no_grad():negative=self.reference(features,skip_mid,skip_full)
        positive=(self.tail(features,skip_mid,skip_full) if parameters is None else
            torch.func.functional_call(self.tail,parameters,(features,skip_mid,skip_full)))
        return positive-negative


class NAFHead3Inference(torch.nn.Module):
    def __init__(self,head):
        super().__init__()
        if any(p.requires_grad for p in head.parameters()):raise ValueError('Freeze inference head first')
        official=Tail3(head.backbone).state_dict();reference=head.reference.state_dict()
        if official.keys()!=reference.keys() or any(not torch.equal(v,reference[k]) for k,v in official.items()):raise ValueError('Reference must equal official tail')
        self.head=head
    @torch.no_grad()
    def forward(self,condition):
        with strict_spatial():
            c=self.head.features(condition)
            return self.head.tail(c['features'],c['skip_mid'],c['skip_full'])-c['official_ending']
