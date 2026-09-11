"""Inference adapter reusing the official ending already evaluated by features()."""
import torch
from scripts.research_naf_head256 import Tail
from scripts.research_head_tail2 import strict_spatial


class NAFHeadInference(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Freeze the inference model before wrapping')
        # A trained or different negative branch cannot be replaced by the base.
        official = Tail(model.backbone).state_dict()
        reference = model.reference.state_dict()
        if official.keys() != reference.keys() or any(
                not torch.equal(value, reference[key]) for key, value in official.items()):
            raise ValueError('Reference must equal the official frozen backbone tail')
        self.model = model

    @torch.no_grad()
    def forward(self, condition):
        with strict_spatial():
            cache = self.model.features(condition)
            return self.model.tail(cache['features'], cache['skip']) - cache['official_ending']
