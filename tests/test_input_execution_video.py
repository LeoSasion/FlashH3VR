"""Actual CPU codec/VFR ownership coverage for optional input buffer reuse."""
from fractions import Fraction

import av
import numpy as np
import pytest

from h3ce.data.video import iter_video_frames
from h3ce.errors import H3CEError


@pytest.mark.parametrize('long_edge', [96,48])
def test_vfr_owned_frames_and_real_resize_match_original(tmp_path,long_edge):
    path=tmp_path/'cpu_vfr.mkv'
    pts=[0,40,125]
    with av.open(str(path),'w') as destination:
        stream=destination.add_stream('ffv1',rate=25)
        stream.width,stream.height,stream.pix_fmt=96,64,'bgr0'
        stream.time_base=stream.codec_context.time_base=Fraction(1,1000)
        stream.codec_context.color_primaries=1
        stream.codec_context.color_trc=13
        stream.codec_context.color_range=2
        for index,timestamp in enumerate(pts):
            codes=(np.arange(64*96*3,dtype=np.uint32).reshape(64,96,3)+index*23).astype(np.uint8)
            frame=av.VideoFrame.from_ndarray(codes,format='rgb24')
            frame.pts,frame.time_base=timestamp,Fraction(1,1000)
            for packet in stream.encode(frame):destination.mux(packet)
        for packet in stream.encode():destination.mux(packet)
    reference=list(iter_video_frames(path,working_long_edge_max=long_edge,max_frames=3,reuse_reformatter=True))
    actual=list(iter_video_frames(path,working_long_edge_max=long_edge,max_frames=3,reuse_reformatter=True,reuse_buffers=True))
    assert len(actual)==3 and [f.pts for f in actual]==[0.,.04,.125]
    for old,new in zip(reference,actual,strict=True):
        assert old.rgb.shape==new.rgb.shape and old.rgb.tobytes()==new.rgb.tobytes()
        assert (old.pts,old.pts_integer,old.time_base,old.color_contract)==(new.pts,new.pts_integer,new.time_base,new.color_contract)
    assert not any(np.shares_memory(a.rgb,b.rgb) for i,a in enumerate(actual) for b in actual[i+1:])
    second=actual[1].rgb.copy();actual[0].rgb[:]=0
    assert np.array_equal(second,actual[1].rgb)
    with pytest.raises(H3CEError,match='E_VIDEO_LIMIT'):
        list(iter_video_frames(path,max_frames=2,reuse_reformatter=True,reuse_buffers=True))
