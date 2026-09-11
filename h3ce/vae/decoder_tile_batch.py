"""Frozen decoder-only tile batching; original encoder execution is retained."""
from types import MethodType
from h3ce.cache.keys import file_sha256
from .tile_batch import TileBatchH3Backend


class DecoderTileBatchH3Backend(TileBatchH3Backend):
    def __init__(self, model, *, weight_sha256, tile_batch_size=4):
        super().__init__(model,weight_sha256=weight_sha256,tile_batch_size=tile_batch_size)
        model._encode_clip = MethodType(type(model)._encode_clip,model)

    def numerical_contract(self):
        result = super().numerical_contract()
        result['tiling']['encoder_batch_size'] = 1
        result['tiling']['decoder_batch_size'] = self.tile_batch_size
        result['tiling']['batch_scope'] = 'decoder only; native encoder method unchanged'
        result['decoder_tile_batch_source_sha256'] = file_sha256(__file__)
        return result
