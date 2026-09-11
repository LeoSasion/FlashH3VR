"""Explicit tagged 10bit ProRes422 SDR conversion for the NASA research source."""
import numpy as np
from h3ce.data.decode import resize_rgb
from scripts.review_continuous_head_h3 import sample_axis

CONVERSION_ID='tagged_prores422p10_bt709_limited_float64_cosited_srgb_v2'


def validate_frame(frame,width,height):
    if frame.format.name!='yuv422p10le':
        raise ValueError('Only explicitly tagged10bit planar422 reference supported')
    tags={key:int(getattr(frame,key)) for key in ('color_range','colorspace','color_trc','color_primaries')}
    if list(tags.values())!=[1,1,1,1]:
        raise ValueError('Actual frame must declare limited-range BT709 matrix/transfer/primaries')
    if frame.interlaced_frame:
        raise ValueError('Interlaced source requires a separate declared transform')
    if width<=0 or height<=0 or width>frame.width or height>frame.height or width*frame.height!=height*frame.width:
        raise ValueError('Working target must preserve aspect without upsampling')
    if getattr(frame,'rotation',0) or any('DISPLAYMATRIX' in str(item.type).upper() for item in frame.side_data):
        raise ValueError('Display rotation requires a separate geometry contract')
    if any(any(marker in str(item.type).upper() for marker in ('MASTERING_DISPLAY','CONTENT_LIGHT','DYNAMIC_HDR','DOVI')) for item in frame.side_data):
        raise ValueError('HDR metadata conflicts with this fixed SDR source contract')
    return tags


def frame_to_code_rgb(frame,width,height):
    validate_frame(frame,width,height)
    planes=[np.frombuffer(p,dtype='<u2').reshape(p.height,p.line_size//2)[:,:p.width].astype(np.float64)
            for p in frame.planes]
    if len(planes)!=3 or any(p.min()<0 or p.max()>1023 for p in planes):
        raise ValueError('Expected three valid10bit planes')
    y=(planes[0]-64)/876
    # BT709422 chroma is co-sited with the first luma column, x=0,2,4,...
    centers=np.arange(frame.width,dtype=np.float64)/2
    cb=sample_axis((planes[1]-512)/896,centers,.5,-1)
    cr=sample_axis((planes[2]-512)/896,centers,.5,-1)
    kr=.2126;kb=.0722;kg=1-kr-kb
    r=y+2*(1-kr)*cr;b=y+2*(1-kb)*cb
    g=y-2*kb*(1-kb)/kg*cb-2*kr*(1-kr)/kg*cr
    code=np.stack([r,g,b],axis=-1).clip(0,1).astype(np.float32)
    return resize_rgb(code,(height,width))


def bt709_code_to_srgb(code):
    code=np.asarray(code,np.float32)
    if not np.isfinite(code).all() or code.min()<0 or code.max()>1:
        raise ValueError('Finite normalized RGB code required')
    linear=np.where(code<.081,code/4.5,((code+.099)/1.099)**(1/.45))
    return np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055).clip(0,1).astype(np.float32)


def frame_to_srgb(frame,width,height):
    return bt709_code_to_srgb(frame_to_code_rgb(frame,width,height))
