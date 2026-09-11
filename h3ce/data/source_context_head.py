"""Fill head-bucket letterbox with source context, preserving the write geometry."""
from copy import deepcopy
import torch
from .continuous_head import _resample, valid_bucket, crop_feather, inverse_delta, paste_delta


def from_roi_transform(transform):
    if transform.get('version') != 'continuous_head_keys_v1':
        raise ValueError('Verified continuous ROI transform required')
    result=deepcopy(transform)
    result['version']='source_context_head_keys_v1'
    result['padding']='Real source context outside the write ROI; source-image boundary replicate only; original ROI supervision mask retained'
    result['sampling']='Keys a=-0.5; same affine, scale and filter; sample centers clamped to full source canvas, not the write ROI'
    x0,y0,_,_=result['crop_xyxy'];left,_,top,_=result['pad_lrtb'];s=result['scale_xy'][0]
    h,w=result['bucket_hw']
    result['context_view_xyxy']=[x0-left/s,y0-top/s,x0+(w-left)/s,y0+(h-top)/s]
    return result


def pack_frame(frame, transform):
    if transform.get('version') != 'source_context_head_keys_v1':
        raise ValueError('Explicit source-context transform required')
    if frame.ndim!=4 or list(frame.shape[-2:])!=transform['canvas_hw']:
        raise ValueError('Frame and transform canvas differ')
    x0,y0,_,_=transform['crop_xyxy'];left,_,top,_=transform['pad_lrtb']
    h,w=transform['bucket_hw'];sh,sw=transform['canvas_hw'];scale=transform['scale_xy'][0]
    xs=(x0+(torch.arange(w,dtype=torch.float64)+.5-left)/scale-.5).clamp(0,sw-1)
    ys=(y0+(torch.arange(h,dtype=torch.float64)+.5-top)/scale-.5).clamp(0,sh-1)
    return _resample(frame,xs,ys,1/scale).clamp(0,1)
