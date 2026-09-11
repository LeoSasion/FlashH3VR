"""Frozen single-shot video core with native H3 and dynamic batched head repair."""
import torch
import torch.nn.functional as F
from h3ce.data.head_bucket import head_transform, pack_frame, paste_delta


class FrozenHeadVideoPipeline:
    """Input-only core; callers own YOLO11 detection, shots, media and PTS IO.

    No persistent input/latent cache. Each call runs native encoding and decoding.
    The supplied head maps a fixed-size condition bucket to an RGB correction.
    """
    def __init__(self, bridge, head, *, head_side=256, batch_size=8):
        if type(batch_size) is not int or batch_size<1 or head_side not in (256,512):
            raise ValueError('Invalid head size or batch size')
        self.bridge,self.head,self.head_side,self.batch_size=bridge,head,head_side,batch_size

    @torch.no_grad()
    def __call__(self, frames, meta, transforms):
        if frames.ndim!=4 or frames.shape[1]!=3 or frames.shape[0]!=len(transforms):
            raise ValueError('Expected TCHW RGB and one transform per source frame')
        meta.validate(len(frames))
        if meta.kind!='video' or not meta.real_video:
            raise ValueError('This core requires a real single-shot video')
        if any(p.requires_grad for m in (self.head,self.bridge.backend.model) for p in m.parameters()):
            raise ValueError('This core requires frozen models')
        h,w=frames.shape[-2:]
        if min(h,w)<32: raise ValueError('Video sides must be at least32')
        for i,g in enumerate(transforms):
            if g['bbox_provenance']!='input_detector':
                raise ValueError('Inference geometry must originate from input detection')
            expected=head_transform(g['face_xyxy'],(h,w),frame_index=i,pts=meta.pts[i],side=self.head_side,
                expansion_xy=g['expansion_xy'],provenance='input_detector')
            if g!=expected: raise ValueError('Transform/PTS/canvas mismatch')
        # H3 remains in original coordinates, with recorded implicit bottom/right pad.
        padded=F.pad(frames,(0,(-w)%32,0,(-h)%32),mode='reflect')
        video=padded.permute(1,0,2,3)[None].contiguous()
        codec=self.bridge.current_codec_pack()
        z=self.bridge.encode_rgb(video,meta)
        native=self.bridge.decode_latent(z,grad=False,codec_pack=codec)
        condition=.5*(frames+native[0].permute(1,0,2,3)[...,:h,:w].clamp(0,1))
        prior=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        try:
            buckets=torch.cat([pack_frame(condition[i:i+1],g) for i,g in enumerate(transforms)])
            delta=torch.cat([self.head(buckets[i:i+self.batch_size]) for i in range(0,len(frames),self.batch_size)])
            if delta.shape!=buckets.shape or not torch.isfinite(delta).all():
                raise ValueError('Invalid head correction')
            output=torch.cat([paste_delta(frames[i:i+1],delta[i:i+1],g) for i,g in enumerate(transforms)])
        finally:
            torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=prior
        if not torch.isfinite(output).all():raise ValueError('Nonfinite output')
        return {'prediction':output,'delta':delta,'native':native,'latent':z.tensor,'pts':meta.pts}
