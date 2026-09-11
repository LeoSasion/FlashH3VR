"""CPU rounding, byte identity, storage lifetime and logical hash checks."""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from h3ce.data.face_preparation import FaceBytePreparer, frame_sha256


def original_bgr(frame):
    return np.ascontiguousarray(np.uint8(frame.clip(0,1)*255+.5)[:,:,::-1])


@pytest.mark.parametrize('mode', ['legacy','inplace_rgb','direct_bgr'])
def test_all_quantization_boundary_neighbors_match_original(mode):
    centers = ((np.arange(256, dtype=np.float64)+.5)/255).astype(np.float32)
    values = np.concatenate([np.nextafter(centers, np.float32(-np.inf)),centers,
        np.nextafter(centers,np.float32(np.inf)),np.array([-100,-1,-0.,0.,1.,2.,100],np.float32)])
    frame = np.stack([values,np.roll(values,1),np.roll(values,7)],axis=-1)[None]
    source = frame.tobytes()
    result = FaceBytePreparer(mode).prepare_bgr([frame])[0]
    assert result.dtype == np.uint8 and result.flags.c_contiguous and result.flags.owndata
    assert result.tobytes() == original_bgr(frame).tobytes() and frame.tobytes() == source


@pytest.mark.parametrize('mode', ['inplace_rgb','direct_bgr'])
@pytest.mark.parametrize('layout', ['contiguous','fortran','transpose','reverse','channels','readonly'])
def test_layout_source_and_previous_byte_frames_remain_independent(mode, layout):
    rng = np.random.default_rng(240911)
    frame = rng.uniform(-.2,1.2,(13,17,3)).astype(np.float32)
    if layout == 'fortran':frame = np.asfortranarray(frame)
    if layout == 'transpose':frame = frame.transpose(1,0,2)
    if layout == 'reverse':frame = frame[::-1,::-1]
    if layout == 'channels':frame = frame[:,:,::-1]
    if layout == 'readonly':frame.flags.writeable = False
    source, expected = frame.tobytes(), original_bgr(frame)
    preparer = FaceBytePreparer(mode)
    first, second = preparer.prepare_bgr([frame,frame])
    scratch = preparer.scratch
    assert first.tobytes() == second.tobytes() == expected.tobytes()
    assert not np.shares_memory(first,second) and not np.shares_memory(first,scratch)
    later = preparer.prepare_bgr([np.zeros_like(frame)])
    assert preparer.scratch is scratch and not later[0].any()
    assert first.tobytes() == second.tobytes() == expected.tobytes() and frame.tobytes() == source
    first.fill(17)
    assert second.tobytes() == expected.tobytes()
    resized = preparer.prepare_bgr([np.full((2,4,3),.5,np.float32)])[0]
    assert preparer.scratch.shape == (2,4,3) and np.all(resized==128)


@pytest.mark.parametrize('layout', ['contiguous','fortran','transpose','reverse','readonly'])
def test_hash_matches_logical_c_order_bytes_including_float_bit_patterns(layout):
    # Hashing preserves bit patterns, including sign of zero and NaN payloads;
    # it does not sanitize, quantize or compare floating point values.
    patterns = np.array([0x00000000,0x80000000,0x7fc00001,0x7fc00002,0x3f800000,0xbf000000],np.uint32)
    frame = np.tile(patterns,12).view(np.float32).reshape(3,8,3)
    if layout == 'fortran':frame=np.asfortranarray(frame)
    if layout == 'transpose':frame=frame.transpose(1,0,2)
    if layout == 'reverse':frame=frame[::-1,::-1,::-1]
    if layout == 'readonly':frame.flags.writeable=False
    before = frame.tobytes(order='C')
    assert frame_sha256(frame) == hashlib.sha256(before).hexdigest()
    assert frame.tobytes(order='C') == before


def test_private_scratch_cannot_cross_worker_threads():
    preparer = FaceBytePreparer('inplace_rgb')
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(preparer.prepare_bgr,[np.zeros((1,1,3),np.float32)])
        with pytest.raises(RuntimeError, match='another thread'):future.result()


@pytest.mark.parametrize('frames', [[],[np.zeros((2,2,3),np.uint8)],
    [np.zeros((2,2),np.float32)],[np.zeros((1,2,3),np.float32),np.zeros((2,2,3),np.float32)]])
def test_invalid_source_rejected(frames):
    with pytest.raises(ValueError):FaceBytePreparer('inplace_rgb').prepare_bgr(frames)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):FaceBytePreparer('automatic').prepare_bgr([np.zeros((1,1,3),np.float32)])
