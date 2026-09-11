"""CPU numerical and ownership tests for optional input execution changes."""
from types import SimpleNamespace

import numpy as np
import pytest

from h3ce.data.video import _decode_rgb,detect_shots
from h3ce.data.decode import resize_rgb


@pytest.mark.parametrize('transfer', ['project_declared_sdr_srgb','srgb_identity','bt709_to_srgb'])
@pytest.mark.parametrize('strided', [False,True])
def test_float_buffer_reuse_all_uint8_codes_exact_and_source_unchanged(transfer,strided):
    raw=np.tile(np.arange(256,dtype=np.uint8), (3,7,1)).transpose(1,2,0)
    raw=raw[:,::-1] if strided else np.ascontiguousarray(raw)
    saved=raw.copy()
    frame=SimpleNamespace(to_ndarray=lambda **kwargs:raw)
    contract=dict(matrix='native_rgb',transfer=transfer)
    expected=_decode_rgb(frame,contract)
    actual=_decode_rgb(frame,contract,reuse_buffers=True)
    assert actual.dtype==np.float32 and actual.shape==expected.shape and actual.tobytes()==expected.tobytes()
    assert np.array_equal(raw,saved) and not np.shares_memory(actual,raw)
    actual[:]=0
    assert np.array_equal(raw,saved)


@pytest.mark.parametrize('shape', [(48,48),(12,31),(96,129)])
@pytest.mark.parametrize('strides', ['contiguous','negative','subsampled'])
def test_packed_float_planes_preserve_pillow_bicubic(shape,strides):
    x=np.random.default_rng(741).random((67,99,3),dtype=np.float32)
    if strides=='negative':x=x[::-1,::-1]
    if strides=='subsampled':x=x[::2,::2]
    saved=x.copy()
    expected=resize_rgb(x,shape)
    actual=resize_rgb(x,shape,packed_channels=True)
    assert np.array_equal(actual.view(np.uint8),expected.view(np.uint8)) and np.array_equal(x,saved)


def test_shot_measurements_and_cuts_exact_on_flash_and_color_changes():
    rng=np.random.default_rng(71)
    dark=rng.random((63,97,3),dtype=np.float32)*.03
    light=1-dark
    red=np.zeros_like(dark);red[...,0]=1
    frames=[dark,dark.copy(),light,light*.97,red,dark]
    expected=detect_shots(frames)
    actual=detect_shots(frames,packed_channels=True)
    assert actual==expected and any(t['cut'] for t in actual[1]['transitions'])
