"""Apply the same Dense Inter once per native H3 spatial latent window.

The H3 object only supplies its existing geometry and differentiable stitch
operation. It is never registered inside the Inter or changed here.
"""
from __future__ import annotations

import torch


def _axis_plan(native_h3, latent_length):
    pixels = latent_length * 16
    values = native_h3._split_tiles(pixels)
    if len(values) != 3:
        raise ValueError('Native tile plan must contain starts, sizes and overlaps')
    starts, sizes, overlaps = [list(v) for v in values]
    if (not starts or len(sizes) != len(starts) or len(overlaps) != len(starts)-1
            or any(type(v) is not int for part in [starts, sizes, overlaps] for v in part)
            or starts[0] != 0 or starts[-1]+sizes[-1] != pixels
            or any(size != 256 for size in sizes)
            or any(start < 0 or start % 16 for start in starts)
            or any(not 64 <= overlap < 256 or overlap % 16 for overlap in overlaps)):
        raise ValueError('Invalid native256 tile geometry or latent alignment')
    for i, overlap in enumerate(overlaps):
        if starts[i+1] <= starts[i] or starts[i]+sizes[i]-starts[i+1] != overlap:
            raise ValueError('Native overlap differs from tile coordinates')
    return dict(starts=starts, sizes=sizes, overlaps=overlaps)


def native_tiled_dense_inter(z, inter, native_h3):
    """Return corrected normalized latent and actual window-processing metadata.

    Each window is read from the original z. Only residual corrections are
    stitched; no window consumes a previous window's restored result. The
    underlying Dense keeps its existing independent processing of B*T rows.
    """
    if (z.ndim != 5 or min(z.shape) < 1 or z.shape[1] != 24
            or min(z.shape[-2:]) < 16 or any(n % 2 for n in z.shape[-2:])
            or not z.is_floating_point() or not bool(torch.isfinite(z).all())):
        raise ValueError('Expected finite H3 [B,24,T,H,W] latent from a >=256, 32-aligned canvas')
    if (inter.spec.kind != 'dense' or inter.spec.channels != 24
            or tuple(inter.spec.latent_hw) != (16, 16)):
        raise ValueError('Native tiling requires the configured 24-channel,16x16 Dense Inter')
    if (not native_h3.use_tiling or native_h3.tile_size != 256
            or native_h3.spatial_compression != 16 or native_h3.tile_overlap_min < 64):
        raise ValueError('H3 native256 tiling and minimum64 overlap must remain enabled')
    yp = _axis_plan(native_h3, z.shape[-2])
    xp = _axis_plan(native_h3, z.shape[-1])
    count = len(yp['starts']) * len(xp['starts'])
    info = dict(y_pixels=yp, x_pixels=xp, inter_module_calls=count,
                latent_time_positions=z.shape[2], batch_size=z.shape[0],
                dense_independent_time_rows=count*z.shape[0]*z.shape[2],
                new_cross_time_mixing=False, repair_steps=1)
    if count == 1:
        out = inter(z)
    else:
        rows = []
        for sy in yp['starts']:
            row = []
            for sx in xp['starts']:
                tile = z[..., sy//16:sy//16+16, sx//16:sx//16+16]
                # Keep the existing Dense implementation/checkpoint contract.
                corrected = inter(tile)
                if corrected.shape != tile.shape or corrected.dtype != z.dtype:
                    raise ValueError('Dense returned a different tile shape or dtype')
                row.append(corrected-tile)
            rows.append(row)
        residual = native_h3._stitch_tiles(rows,
            [v//16 for v in yp['overlaps']], [v//16 for v in xp['overlaps']])
        if residual.shape != z.shape:
            raise ValueError('Native residual stitching returned a different latent shape')
        out = z + residual
    if out.shape != z.shape or out.dtype != z.dtype or not bool(torch.isfinite(out).all()):
        raise ValueError('Tiled Dense output must preserve the finite latent shape and dtype')
    return out, info
