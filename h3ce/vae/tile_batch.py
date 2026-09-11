"""Opt-in frozen H3 inference: batch spatial tiles, retain native time/stitching."""
from types import MethodType

import torch

from h3ce.cache.keys import file_sha256
from .aitoolkit_h3_backend import AIToolkitH3Backend


def spatial_clip(model, tensor, *, decode, tile_batch_size):
    """Batch tiles along B only; split them back before the original stitch."""
    if type(tile_batch_size) is not int or tile_batch_size < 1:
        raise ValueError("tile_batch_size must be a positive integer")
    if not model.use_tiling:
        raise ValueError("Batched tiles require native spatial tiling")
    ratio = model.spatial_compression
    scale = ratio if decode else 1
    ys, hs, yo = model._split_tiles(tensor.shape[-2] * scale)
    xs, ws, xo = model._split_tiles(tensor.shape[-1] * scale)
    tiles = [tensor[..., y//scale:(y+h)//scale, x//scale:(x+w)//scale]
             for y, h in zip(ys, hs) for x, w in zip(xs, ws)]
    # Group by shape without changing the eventual row-major blending order.
    groups = {}
    for index, tile in enumerate(tiles):
        groups.setdefault(tuple(tile.shape[1:]), []).append(index)
    results = [None] * len(tiles)
    batch = tensor.shape[0]
    for indices in groups.values():
        for start in range(0, len(indices), tile_batch_size):
            selected = indices[start:start+tile_batch_size]
            packed = torch.cat([tiles[i] for i in selected], dim=0)
            output = (model.decoder(model.post_quant_conv(packed)) if decode
                      else model.quant_conv(model.encoder(packed)))
            if output.shape[0] != batch * len(selected):
                raise RuntimeError("Native tile batch dimension changed")
            for i, value in zip(selected, output.split(batch, dim=0)):
                results[i] = value
    rows = [results[i:i+len(xs)] for i in range(0, len(results), len(xs))]
    return model._stitch_tiles(rows, yo if decode else [v//ratio for v in yo],
                               xo if decode else [v//ratio for v in xo])


class TileBatchH3Backend(AIToolkitH3Backend):
    """Separate numerical/cache contract; production serial backend is untouched."""
    def __init__(self, model, *, weight_sha256, tile_batch_size=4):
        super().__init__(model, weight_sha256=weight_sha256)
        if type(tile_batch_size) is not int or tile_batch_size < 1:
            raise ValueError("tile_batch_size must be a positive integer")
        self.tile_batch_size = tile_batch_size
        self.model._encode_clip = MethodType(
            lambda m, x: spatial_clip(m, x, decode=False, tile_batch_size=self.tile_batch_size), model)
        self.model._decode_clip = MethodType(
            lambda m, z: spatial_clip(m, z, decode=True, tile_batch_size=self.tile_batch_size), model)

    def decode_raw(self, raw_latents):
        if torch.is_grad_enabled():
            raise RuntimeError("TileBatchH3Backend is frozen inference only")
        return super().decode_raw(raw_latents)

    def numerical_contract(self):
        result = super().numerical_contract()
        result['tiling']['batch_size'] = self.tile_batch_size
        result['tiling']['batch_axis'] = 'independent spatial tiles folded into B; native T unchanged'
        result['tile_batch_source_sha256'] = file_sha256(__file__)
        return result
