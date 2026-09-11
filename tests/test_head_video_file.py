"""Real CPU codec/PTS roundtrips, no H3 or face-model inference."""
from fractions import Fraction
import av
import pytest
import torch
import numpy as np
from h3ce.infer.head_video_file import encode_srgb_h264
from h3ce.infer import head_video_file
from h3ce.data.video import iter_video_frames


@pytest.mark.parametrize('reuse', [False, True])
def test_writer_preserves_nonuniform_pts_and_color(tmp_path, reuse):
    path=tmp_path/'nonuniform.mp4'
    frames=torch.tensor([.2,.4,.6])[None,:,None,None].expand(6,3,64,96).contiguous()
    pts=[0,2,5,7,11,17];tb=[[1,120]]*6
    encode_srgb_h264(path,frames,pts,tb,rate=Fraction(60),reuse_reformatter=reuse)
    with av.open(str(path)) as source:
        actual=[(f.width,f.height,f.pts*f.time_base) for f in source.decode(video=0)]
    assert actual==[(96,64,Fraction(i,120)) for i in pts]
    decoded=list(iter_video_frames(path,max_frames=6,reuse_reformatter=reuse))
    for frame in decoded:
        assert frame.color_contract['transfer']=='bt709_to_srgb'
        assert abs(frame.rgb-[.2,.4,.6]).max()<.015


def test_writer_refuses_overwrite_and_invalid_time(tmp_path):
    path=tmp_path/'existing.mp4';path.write_bytes(b'keep original')
    x=torch.zeros(2,3,64,96)
    with pytest.raises(FileExistsError):encode_srgb_h264(path,x,[0,1],[[1,60]]*2,rate=60)
    assert path.read_bytes()==b'keep original'
    other=tmp_path/'bad.mp4'
    with pytest.raises(ValueError):encode_srgb_h264(other,x,[1,1],[[1,60]]*2,rate=60)
    assert not other.exists()


def test_owned_detector_receives_whole_video_once():
    detector=object.__new__(head_video_file.Yolo11BatchExecutor)
    frames=np.zeros((33,2,2,3),dtype=np.float32)
    calls=[]
    def detect(value):
        assert value is frames
        calls.append(len(value))
        return [[i] for i in range(len(value))]
    detector.detect_frames=detect
    assert list(head_video_file._detect_video_faces(detector,frames,reuse_execution=True))==[[i] for i in range(33)]
    assert calls==[33]


@pytest.mark.parametrize('reuse', [False,True])
def test_legacy_detector_keeps_batching_and_rgb_quantization(monkeypatch,reuse):
    frames=np.full((17,2,2,3),.5,dtype=np.float32)
    calls=[]
    def detect(detector,byte):
        assert all(x.dtype==np.uint8 and (x==128).all() for x in byte)
        start=sum(calls);calls.append(len(byte))
        return [[i] for i in range(start,start+len(byte))]
    monkeypatch.setattr(head_video_file,'detect_face_batch_reuse' if reuse else 'detect_face_batch',detect)
    assert list(head_video_file._detect_video_faces(object(),frames,reuse_execution=reuse))==[[i] for i in range(17)]
    assert calls==[16,1]
