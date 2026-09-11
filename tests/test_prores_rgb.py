import av
import numpy as np
import pytest
from scripts.research_prores_rgb import frame_to_code_rgb,frame_to_srgb,bt709_code_to_srgb


def constant_frame(y,u=512,v=512):
    frame=av.VideoFrame(16,16,'yuv422p10le')
    for plane,value in zip(frame.planes,(y,u,v)):
        plane.update(np.full((plane.height,plane.line_size//2),value,dtype='<u2').tobytes())
    frame.color_range=frame.colorspace=frame.color_trc=frame.color_primaries=1
    return frame


def test_limited_neutral_levels_against_10bit_reference():
    for y in (0,64,128,256,512,940,1023):
        expected=np.clip((y-64)/876,0,1)
        np.testing.assert_allclose(frame_to_code_rgb(constant_frame(y),8,8),expected,atol=1e-4,rtol=0)


def test_chroma_against_independent_bt709_matrix():
    for y,u,v in ((512,512,700),(512,700,512),(512,300,300),(800,480,570)):
        yy=(y-64)/876;cb=(u-512)/896;cr=(v-512)/896
        expected=np.clip([yy+1.5748*cr,yy-.1873242729306488*cb-.46812427293064884*cr,yy+1.8556*cb],0,1)
        actual=frame_to_code_rgb(constant_frame(y,u,v),8,8)
        np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-4,rtol=0)


def test_adjacent_10bit_codes_survive_before_float_conversion():
    levels=np.array([frame_to_code_rgb(constant_frame(y),16,16)[4,4,0] for y in range(512,520)])
    assert len(np.unique(levels))==8
    np.testing.assert_allclose(levels,(np.arange(512,520)-64)/876,atol=1e-4,rtol=0)


def test_unverified_or_hdr_tags_are_rejected_without_mutation():
    for tag,value in (('colorspace',2),('color_trc',16),('color_primaries',9),('color_range',0)):
        frame=constant_frame(512);setattr(frame,tag,value)
        with pytest.raises(ValueError):frame_to_srgb(frame,8,8)
        assert int(getattr(frame,tag))==value


def test_aspect_distortion_and_target_upsampling_are_rejected():
    frame=constant_frame(512)
    for width,height in ((8,10),(32,32),(0,0)):
        with pytest.raises(ValueError):frame_to_srgb(frame,width,height)


def test_srgb_transfer_against_independent_scalar_formula():
    levels=np.array([0,.02,.08,.1,.5,.9,1],np.float32)
    expected=[]
    for value in levels.astype(np.float64):
        linear=value/4.5 if value<.081 else ((value+.099)/1.099)**(1/.45)
        expected.append(linear*12.92 if linear<=.0031308 else 1.055*linear**(1/2.4)-.055)
    np.testing.assert_allclose(bt709_code_to_srgb(levels),expected,atol=2e-7,rtol=0)


def test_422_chroma_is_cosited_with_even_luma_columns():
    frame=constant_frame(512)
    plane=frame.planes[1]
    values=np.full((plane.height,plane.line_size//2),512,dtype='<u2');values[:,4]=576
    plane.update(values.tobytes())
    code=frame_to_code_rgb(frame,16,16)
    baseline=448/876;peak=1.8556*64/896
    np.testing.assert_allclose(code[8,[6,8,10],2],[baseline,baseline+peak,baseline],atol=1e-7,rtol=0)
    np.testing.assert_allclose(code[8,[7,9],2],baseline+.5625*peak,atol=1e-7,rtol=0)
