"""Research-only continuous dynamic bucket, native video H3, residual paste."""
import torch
from h3ce.data.continuous_head import from_stable_records,pack_frame,paste_delta
from h3ce.infer.head_region_native import spatial_precision


@torch.no_grad()
def restore_continuous_head(frames,meta,records,bridge,head,*,side=256,head_batch=8):
    meta.validate(len(frames))
    if (meta.kind!='video' or not meta.real_video or frames.ndim!=4 or frames.shape[1]!=3
            or side not in (256,512) or type(head_batch) is not int or head_batch<1):
        raise ValueError('Real TCHW video and supported bucket/batch required')
    if any(p.requires_grad for m in (head,bridge.backend.model) for p in m.parameters()):
        raise ValueError('Frozen models required')
    gs=from_stable_records(records,meta.pts,tuple(frames.shape[-2:]),side=side)
    with spatial_precision():buckets=torch.cat([pack_frame(frames[i:i+1],g) for i,g in enumerate(gs)])
    video=buckets.permute(1,0,2,3)[None].contiguous()
    latent=bridge.encode_rgb(video,meta)
    native=bridge.decode_latent(latent,grad=False,codec_pack=bridge.current_codec_pack())
    with spatial_precision():
        condition=.5*(buckets+native[0].permute(1,0,2,3).clamp(0,1))
        delta=torch.cat([head(condition[i:i+head_batch]) for i in range(0,len(frames),head_batch)])
        if delta.shape!=buckets.shape or not torch.isfinite(delta).all():raise ValueError('Invalid correction')
        output=torch.cat([paste_delta(frames[i:i+1],delta[i:i+1],g) for i,g in enumerate(gs)])
    return {'prediction':output,'delta':delta,'native':native,'condition':condition,'buckets':buckets,
            'latent':latent.tensor,'transforms':gs,'pts':meta.pts}
