"""Frozen long-video head research path, one geometry per real source frame.

Nonlearned overlap combines H3 reconstructions of the same frame. NAF then runs
once per eligible frame. Cuts, missing/ambiguous faces, gaps and isolated frames
do not share H3 context. Media decoding and input detection belong to the caller.
"""
import torch
from h3ce.vae.bridge import FrameMeta
from h3ce.data import head_bucket,continuous_head
from h3ce.data.stable_head import stabilize_head_boxes,POLICY
from h3ce.infer.head_region_native import spatial_precision
from scripts.video_benchmark_math import chunk_plan,blend_weights


def prepare_sequence(records,pts,canvas_hw,*,geometry='continuous',side=256):
    if geometry not in ('integer','continuous') or side not in (256,512) or len(records)!=len(pts):
        raise ValueError('Unsupported geometry/bucket or mismatched source frames')
    FrameMeta('video',tuple(pts),real_video=True).validate(len(records))
    boxes=[];shots=[]
    for i,r in enumerate(records):
        if r['frame_index']!=i or r['pts']!=pts[i] or r.get('bbox_provenance')!='input_detector':
            raise ValueError('Source frame/PTS/input detector provenance mismatch')
        box=r['face_xyxy'];reason=r['decision']
        if (box is None and reason not in ('missing','ambiguous')) or (box is not None and reason!='unique_face'):
            raise ValueError('Input eligibility must be explicit')
        boxes.append(box);shots.append(r['shot_id'])
    stable=stabilize_head_boxes(boxes,pts,shots,canvas_hw,side=side)
    segments=[];active=[]
    for i,box in enumerate(boxes):
        if i and (shots[i]!=shots[i-1] or pts[i]-pts[i-1]>POLICY['max_gap_seconds']):
            if active:segments.append(active);active=[]
        if box is None:
            if active:segments.append(active);active=[]
        else:active.append(i)
    if active:segments.append(active)
    chunks=[];used=[];skipped={i:r['decision'] for i,r in enumerate(records) if r['face_xyxy'] is None}
    for segment_id,indices in enumerate(segments):
        if len(indices)<2:
            skipped[indices[0]]='isolated_frame_no_video_context';continue
        used.extend(indices)
        for ch in chunk_plan([segment_id]*len(indices),size=22,overlap=5):
            ch={**ch,'start':ch['start']+indices[0],'stop':ch['stop']+indices[0],
                'source_shot_id':shots[indices[0]]}
            chunks.append(ch)
    module=continuous_head if geometry=='continuous' else head_bucket
    plans=[None]*len(records)
    for i in used:
        r=stable[i]
        g=module.head_transform(r['used_face_xyxy'],canvas_hw,frame_index=i,pts=pts[i],side=side,provenance='stabilized_input_detector')
        a,b,c,d=g['crop_xyxy'];ra,rb,rc,rd=boxes[i]
        if not (a<=ra and b<=rb and c>=rc and d>=rd):raise ValueError('Crop lost raw face coverage')
        plans[i]=g
    return {'geometry':geometry,'side':side,'pts':list(pts),'transforms':plans,'stable_records':stable,
            'chunks':chunks,'used_frames':used,'skipped_frames':skipped,'future_frames':2}


@torch.no_grad()
def native_sequence(buckets,pts,chunks,bridge,*,captured_chunks=None):
    """Time axis remains intact in each real H3 clip; no cross-frame resampling."""
    total=torch.zeros_like(buckets);weight=torch.zeros(len(buckets),device=buckets.device)
    context=[];pack=bridge.current_codec_pack()
    for ch in chunks:
        a,b=ch['start'],ch['stop']
        meta=FrameMeta('video',tuple(pts[a:b]),real_video=True,shot_id=str(ch['source_shot_id']))
        z=bridge.encode_rgb(buckets[a:b].permute(1,0,2,3)[None].contiguous(),meta)
        rgb=bridge.decode_latent(z,grad=False,codec_pack=pack)[0].permute(1,0,2,3)
        if rgb.shape!=buckets[a:b].shape or not torch.isfinite(rgb).all():raise ValueError('Invalid native video reconstruction')
        w=torch.from_numpy(blend_weights(ch)).to(buckets.device)
        total[a:b]+=rgb*w[:,None,None,None];weight[a:b]+=w
        if captured_chunks is not None:captured_chunks.append(rgb)
        context.append({**ch,'pts':list(meta.pts),'valid_frames':z.valid_frames,'padded_frames':z.padded_frames})
    valid=weight>0
    if not torch.allclose(weight[valid],torch.ones_like(weight[valid]),rtol=1e-6,atol=1e-6):
        raise ValueError('Overlap does not partition real-frame reconstruction')
    divisor=torch.where(valid,weight,torch.ones_like(weight))
    return total/divisor[:,None,None,None],weight,context


@torch.no_grad()
def restore_head_sequence(frames,meta,records,bridge,head,*,geometry='continuous',side=256,head_batch=8,retain_chunks=False):
    if (frames.ndim!=4 or frames.shape[1]!=3 or frames.dtype!=torch.float32
            or meta.kind!='video' or not meta.real_video or type(head_batch) is not int or head_batch<1):
        raise ValueError('Real FP32 TCHW video and positive batch required')
    meta.validate(len(frames))
    if any(p.requires_grad for m in (head,bridge.backend.model) for p in m.parameters()):raise ValueError('Frozen models required')
    if not torch.isfinite(frames).all() or frames.min()<0 or frames.max()>1:raise ValueError('Invalid input RGB')
    plan=prepare_sequence(records,meta.pts,tuple(frames.shape[-2:]),geometry=geometry,side=side)
    module=continuous_head if geometry=='continuous' else head_bucket
    buckets=torch.zeros(len(frames),3,side,side,device=frames.device)
    with spatial_precision():
        for i in plan['used_frames']:buckets[i:i+1]=module.pack_frame(frames[i:i+1],plan['transforms'][i])
    captured=[] if retain_chunks else None
    native,weight,context=native_sequence(buckets,meta.pts,plan['chunks'],bridge,captured_chunks=captured)
    expected=torch.zeros_like(weight,dtype=torch.bool);expected[plan['used_frames']]=True
    if not torch.equal(weight>0,expected):raise ValueError('Native coverage differs from eligible source frames')
    delta=torch.zeros_like(buckets);output=frames.clone()
    with spatial_precision():
        condition=.5*(buckets+native.clamp(0,1))
        indices=plan['used_frames']
        for start in range(0,len(indices),head_batch):
            selected=indices[start:start+head_batch];d=head(condition[selected])
            if d.shape!=condition[selected].shape or not torch.isfinite(d).all():raise ValueError('Invalid spatial correction')
            delta[selected]=d
        for i in indices:output[i:i+1]=module.paste_delta(frames[i:i+1],delta[i:i+1],plan['transforms'][i])
    return {'prediction':output,'delta':delta,'buckets':buckets,'native':native,'condition':condition,
            'weights':weight,'context':context,'plan':plan,'pts':list(meta.pts),'chunk_native':captured}
