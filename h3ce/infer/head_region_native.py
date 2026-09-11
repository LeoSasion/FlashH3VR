"""Experimental crop-first native H3, plus a geometry-matched full-H3 control."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from h3ce.data.head_bucket import head_transform,pack_frame,paste_delta
from h3ce.data.stable_head import validate_stable_records


@contextmanager
def spatial_precision():
    previous=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=previous


@torch.no_grad()
def restore_region(frames, meta, records, bridge, head, *, stable, native_scope,
                   side=256, head_batch=8):
    """No target RGB. H3 video time axis is never folded into image inference.

    Every call recomputes native E/D from this call's complete input sequence.
    """
    meta.validate(len(frames))
    if meta.kind!='video' or not meta.real_video or frames.ndim!=4 or frames.shape[1]!=3:
        raise ValueError('Expected real video TCHW RGB')
    if native_scope not in ('full','head') or side not in (256,512) or type(head_batch) is not int or head_batch<1:
        raise ValueError('Invalid native scope or bucket/batch')
    if any(p.requires_grad for m in (head,bridge.backend.model) for p in m.parameters()):raise ValueError('Frozen models required')
    hw=tuple(frames.shape[-2:]);h,w=hw
    if stable:gs=validate_stable_records(records,meta.pts,hw,side=side)
    else:
        gs=records
        if len(gs)!=len(frames):raise ValueError('Frame/geometry count mismatch')
        for i,g in enumerate(gs):
            if g!=head_transform(g['face_xyxy'],hw,frame_index=i,pts=meta.pts[i],side=side,provenance='input_detector'):
                raise ValueError('Raw detector geometry changed')
    if native_scope=='head':
        with spatial_precision():buckets=torch.cat([pack_frame(frames[i:i+1],g) for i,g in enumerate(gs)])
        video=buckets.permute(1,0,2,3)[None].contiguous()
    else:
        video=F.pad(frames,(0,(-w)%32,0,(-h)%32),mode='reflect').permute(1,0,2,3)[None].contiguous()
    latent=bridge.encode_rgb(video,meta)
    native=bridge.decode_latent(latent,grad=False,codec_pack=bridge.current_codec_pack())
    with spatial_precision():
        if native_scope=='head':condition=.5*(buckets+native[0].permute(1,0,2,3).clamp(0,1))
        else:
            full_condition=.5*(frames+native[0].permute(1,0,2,3)[...,:h,:w].clamp(0,1))
            condition=torch.cat([pack_frame(full_condition[i:i+1],g) for i,g in enumerate(gs)])
            buckets=torch.cat([pack_frame(frames[i:i+1],g) for i,g in enumerate(gs)])
        delta=torch.cat([head(condition[i:i+head_batch]) for i in range(0,len(frames),head_batch)])
        if delta.shape!=buckets.shape or not torch.isfinite(delta).all():raise ValueError('Invalid head correction')
        output=torch.cat([paste_delta(frames[i:i+1],delta[i:i+1],g) for i,g in enumerate(gs)])
    return {'prediction':output,'delta':delta,'native':native,'condition':condition,'buckets':buckets,
            'latent':latent.tensor,'transforms':gs,'pts':meta.pts}
