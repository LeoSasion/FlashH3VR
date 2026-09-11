"""Experimental continuous head crop: shared paired geometry, no integer resize.

Pixel-center affine maps use one scale on both axes and fractional letterbox.
Separable Keys cubic (a=-0.5) widens its support for antialiased downsampling.
Filter taps sample the original image, including neighbors just outside the ROI;
the sample center is clamped to the continuous ROI for replicated letterbox.
Only a correction is inverse-mapped; the source image is never resampled to paste.
"""
from __future__ import annotations

import math
import torch
from h3ce.data.stable_head import POLICY, validate_stable_records


def head_transform(face_xyxy, canvas_hw, *, frame_index, pts, side=256,
                   expansion_xy=(1.5, 2.0), provenance='input_detector'):
    h, w = canvas_hw
    if (len(face_xyxy) != 4 or type(side) is not int or side < 1
            or any(type(v) is not int or v < 1 for v in (h, w))):
        raise ValueError('Invalid canvas/bucket')
    a, b, c, d = map(float, face_xyxy)
    ex, ey = map(float, expansion_xy)
    if (not all(math.isfinite(v) for v in (a,b,c,d,ex,ey,pts))
            or not (0 <= a < c <= w and 0 <= b < d <= h) or min(ex,ey) < 1):
        raise ValueError('Invalid face/expansion/PTS')
    cx, cy = (a+c)/2, (b+d)/2
    x0, x1 = max(0.,cx-(c-a)*ex/2), min(float(w),cx+(c-a)*ex/2)
    y0, y1 = max(0.,cy-(d-b)*ey/2), min(float(h),cy+(d-b)*ey/2)
    sw, sh = x1-x0, y1-y0
    if min(sw,sh) < 2:
        raise ValueError('At least two source pixels per crop dimension required')
    scale = side/max(sw,sh)
    left, top = (side-sw*scale)/2, (side-sh*scale)/2
    tx, ty = left+(scale-1)/2-scale*x0, top+(scale-1)/2-scale*y0
    return {
        'version':'continuous_head_keys_v1', 'frame_index':int(frame_index), 'pts':float(pts),
        'canvas_hw':[h,w], 'bucket_hw':[side,side], 'face_xyxy':[a,b,c,d],
        'bbox_provenance':provenance, 'expansion_xy':[ex,ey],
        'crop_xyxy':[x0,y0,x1,y1], 'source_hw':[sh,sw],
        'scale_xy':[scale,scale], 'resized_hw':[sh*scale,sw*scale],
        'pad_lrtb':[left,left,top,top],
        'paste_xyxy':[math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1)],
        'original_to_bucket':[[scale,0,tx],[0,scale,ty],[0,0,1]],
        'bucket_to_original':[[1/scale,0,-tx/scale],[0,1/scale,-ty/scale],[0,0,1]],
        'sampling':'Keys a=-0.5; antialias support max(1,source/output); FP32 RGB; full-frame filter taps; continuous clamped sample centers',
        'padding':'fractional centered letterbox; sample-center replicate; fractional valid area excludes artificial padding',
        'coordinates':'pixel centers; xyxy exclusive upper edges; align_corners=False',
    }


def from_stable_records(records, pts, canvas_hw, *, side=256):
    # Reuse verified float trajectories, not their integer transform. Reject gaps
    # before H3, even though the old geometric fitter already splits its windows.
    validate_stable_records(records,pts,canvas_hw,side=side)
    if any(b-a > POLICY['max_gap_seconds'] for a,b in zip(pts,pts[1:])):
        raise ValueError('Split PTS gaps before native video H3')
    plans=[]
    for r in records:
        g=head_transform(r['used_face_xyxy'],canvas_hw,frame_index=r['frame_index'],
                         pts=r['pts'],side=side,provenance='stabilized_input_detector')
        x0,y0,x1,y1=g['crop_xyxy'];a,b,c,d=r['raw_face_xyxy']
        if not (x0<=a and y0<=b and x1>=c and y1>=d):
            raise ValueError('Continuous crop lost raw input face coverage')
        plans.append(g)
    return plans


def _cubic(x):
    u=x.abs()
    return torch.where(u<1,1.5*u**3-2.5*u**2+1,
                       torch.where(u<2,-.5*u**3+2.5*u**2-4*u+2,0.))


def _axis_plan(centers, input_length, source_step, device):
    width=max(1.,float(source_step));radius=math.ceil(2*width)
    indices=centers.floor().long()[:,None]+torch.arange(-radius+1,radius+1)[None]
    weights=_cubic((centers[:,None]-indices)/width)
    weights=weights/weights.sum(-1,keepdim=True)
    return indices.clamp(0,input_length-1).to(device), weights.float().to(device)


def _sample_axis(x, centers, source_step, axis):
    indices, weights=_axis_plan(centers,x.shape[axis],source_step,x.device)
    values=x.movedim(axis,-1)
    sampled=values.index_select(-1,indices.flatten()).reshape(*values.shape[:-1],*indices.shape)
    return (sampled*weights).sum(-1).movedim(-1,axis)


def _resample(x, xs, ys, source_step):
    if x.dtype!=torch.float32:
        raise ValueError('Continuous geometry requires FP32 RGB/correction')
    return _sample_axis(_sample_axis(x,xs,source_step,-1),ys,source_step,-2)


def pack_frame(frame, transform):
    if frame.ndim!=4 or list(frame.shape[-2:])!=transform['canvas_hw']:
        raise ValueError('Frame and transform canvas differ')
    x0,y0,x1,y1=transform['crop_xyxy'];left,_,top,_=transform['pad_lrtb']
    h,w=transform['bucket_hw'];scale=transform['scale_xy'][0]
    xs=(x0+(torch.arange(w,dtype=torch.float64)+.5-left)/scale-.5).clamp(x0,x1-1)
    ys=(y0+(torch.arange(h,dtype=torch.float64)+.5-top)/scale-.5).clamp(y0,y1-1)
    return _resample(frame,xs,ys,1/scale).clamp(0,1)


def valid_bucket(transform, *, device=None):
    left,_,top,_=transform['pad_lrtb'];rh,rw=transform['resized_hw']
    h,w=transform['bucket_hw']
    def coverage(length, low, high):
        i=torch.arange(length,dtype=torch.float64)
        return (torch.minimum(i+1,torch.tensor(high))-torch.maximum(i,torch.tensor(low))).clamp(0,1)
    return (coverage(h,top,top+rh)[:,None]*coverage(w,left,left+rw)[None,:]).float().to(device)[None,None]


def pack_training_pair(x, y, transform):
    if x.shape!=y.shape:raise ValueError('Pair must share full-canvas geometry')
    return pack_frame(x,transform),pack_frame(y,transform),valid_bucket(transform,device=x.device)


def crop_feather(transform, *, device=None):
    a,b,c,d=transform['paste_xyxy'];x0,y0,x1,y1=transform['crop_xyxy']
    x=torch.arange(a,c,dtype=torch.float64);y=torch.arange(b,d,dtype=torch.float64)
    dx=torch.minimum(x-x0,x1-1-x)/max(1.,(x1-x0)*.05)
    dy=torch.minimum(y-y0,y1-1-y)/max(1.,(y1-y0)*.05)
    return torch.minimum(dy[:,None],dx[None,:]).clamp(0,1).float().to(device)[None,None]


def inverse_delta(delta, transform):
    if delta.ndim!=4 or list(delta.shape[-2:])!=transform['bucket_hw']:
        raise ValueError('Correction and bucket dimensions differ')
    a,b,c,d=transform['paste_xyxy'];x0,y0,_,_=transform['crop_xyxy']
    left,_,top,_=transform['pad_lrtb'];scale=transform['scale_xy'][0]
    xs=(torch.arange(a,c,dtype=torch.float64)-x0+.5)*scale-.5+left
    ys=(torch.arange(b,d,dtype=torch.float64)-y0+.5)*scale-.5+top
    return _resample(delta,xs,ys,scale)


def paste_delta(original, delta, transform):
    if (original.ndim!=4 or list(original.shape[-2:])!=transform['canvas_hw']
            or original.shape[:2]!=delta.shape[:2] or original.dtype!=torch.float32):
        raise ValueError('Original/correction/geometry mismatch')
    a,b,c,d=transform['paste_xyxy'];base=original[...,b:d,a:c]
    feather=crop_feather(transform,device=original.device)
    correction=inverse_delta(delta,transform)
    restored=original.clone()
    restored[...,b:d,a:c]=torch.where(feather==0,base,base+feather*correction)
    return restored
