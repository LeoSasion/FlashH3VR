"""Frozen head research integration for SDR video files without audio.

File open, decode, input detection, real video restoration, transfer and output
close are timed together. No input geometry/model-output disk cache is consumed.
"""
from pathlib import Path
from fractions import Fraction
import time
import numpy as np
import torch,av
from h3ce.data.video import iter_video_frames,detect_shots
from h3ce.data.yolo11_face_batch import detect_face_batch
from h3ce.data.yolo11_face_reuse import detect_face_batch_reuse
from h3ce.data.yolo11_face_executor import Yolo11BatchExecutor
from h3ce.data.face_preparation import frame_sha256
from h3ce.infer.head_region_native import spatial_precision
from h3ce.infer.head_video_sequence import restore_head_sequence
from h3ce.infer.video_input_overlap import read_detect_video_overlap
from h3ce.vae.bridge import FrameMeta


def _detect_video_faces(detector, arrays, *, reuse_execution):
    if isinstance(detector, Yolo11BatchExecutor):
        yield from detector.detect_frames(arrays)
    else:
        for a in range(0, len(arrays), 16):
            byte = [np.uint8(x.clip(0, 1)*255+.5) for x in arrays[a:a+16]]
            yield from (detect_face_batch_reuse if reuse_execution else detect_face_batch)(detector, byte)


def encode_srgb_h264(path,frames,pts_integer,time_bases,*,rate,reuse_reformatter=False):
    """Actual source PTS survive; explicit sRGB -> BT709 before limited YUV420."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if frames.ndim!=4 or frames.shape[1]!=3 or frames.dtype!=torch.float32:
        raise ValueError('Expected FP32 TCHW output')
    n,_,h,w=frames.shape
    if min(h,w)<2 or h%2 or w%2 or len(pts_integer)!=n or len(time_bases)!=n:
        raise ValueError('Even output dimensions and all source timestamps required')
    times=[Fraction(t)*Fraction(*tb) for t,tb in zip(pts_integer,time_bases)]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Output PTS must increase')
    # Set exact input time_base on the encoder instead of resampling source PTS
    # onto a guessed nominal rate. AVI/MP4 stream clocks are preserved as rationals.
    if len(set(tuple(tb) for tb in time_bases))!=1:raise ValueError('Changing source time bases require explicit remux support')
    clock=Fraction(*time_bases[0]);rate=Fraction(rate)
    if rate<=0:raise ValueError('Positive source rate required')
    with av.open(str(path),'w') as dest:
        stream=dest.add_stream('libx264',rate=rate);stream.width=w;stream.height=h;stream.pix_fmt='yuv420p'
        stream.time_base=clock;stream.codec_context.time_base=clock
        stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=4
        for key in ('colorspace','color_primaries','color_trc','color_range'):setattr(stream.codec_context,key,1)
        reformatter=av.video.reformatter.VideoReformatter() if reuse_reformatter else None
        for i in range(n):
            s=frames[i].clamp(0,1)
            linear=torch.where(s<=.04045,s/12.92,((s+.055)/1.055).pow(2.4))
            bt709=torch.where(linear<.018,4.5*linear,1.099*linear.pow(.45)-.099)
            rgb=bt709.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
            raw=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            f=(raw.reformat(format='yuv420p',dst_colorspace='ITU709') if reformatter is None else
               reformatter.reformat(raw,format='yuv420p',dst_colorspace='ITU709'))
            f.pts=int(pts_integer[i]);f.time_base=clock
            for packet in stream.encode(f):dest.mux(packet)
        for packet in stream.encode():dest.mux(packet)


@torch.no_grad()
def restore_video_file(source,destination,bridge,head,detector,*,max_frames,side=256,long_edge=768,reuse_execution=False,optimize_input=False,overlap_input=False):
    source=Path(source);destination=Path(destination)
    if destination.exists() or source.resolve()==destination.resolve():raise FileExistsError('Output must be a new file')
    if type(max_frames) is not int or max_frames<2:raise ValueError('Explicit finite frame limit required')
    if not str(detector.device).startswith('cuda:'):raise ValueError('This verified integration uses GPU face detection')
    torch.cuda.synchronize();start=time.perf_counter();stages={}
    if overlap_input:
        prepared=read_detect_video_overlap(source,detector,max_frames=max_frames,long_edge=long_edge,
                                          reuse_execution=reuse_execution,optimize_input=optimize_input)
        arrays,pts=prepared['arrays'],prepared['pts']
        integer_pts,time_bases=prepared['integer_pts'],prepared['time_bases']
        colors,rate=prepared['color'],prepared['rate']
        shots,shot_report,detected=prepared['shots'],prepared['shot_report'],prepared['faces']
        del prepared
        stages['input_read_shots_and_yolo_overlap']=time.perf_counter()-start;t=time.perf_counter()
    else:
        with av.open(str(source)) as container:
            if len(container.streams.video)!=1:raise ValueError('Exactly one input video stream required')
            if len(container.streams.audio):raise ValueError('Research file path has no audio remux yet; refusing to drop audio')
            rate=container.streams.video[0].average_rate
            if not rate:raise ValueError('Missing source nominal frame rate')
        decoded=list(iter_video_frames(source,working_long_edge_max=long_edge,max_frames=max_frames,reuse_reformatter=reuse_execution,reuse_buffers=optimize_input))
        if len(decoded)<2:raise ValueError('Real continuous input video required')
        arrays=np.stack([f.rgb for f in decoded]);pts=[f.pts for f in decoded]
        integer_pts=[f.pts_integer for f in decoded];time_bases=[list(f.time_base) for f in decoded]
        colors=decoded[0].color_contract;del decoded
        if arrays.shape[1]%2 or arrays.shape[2]%2:raise ValueError('This H264 writer requires even working dimensions')
        stages['file_open_decode_color_resize']=time.perf_counter()-start;t=time.perf_counter()
        shots,shot_report=detect_shots(list(arrays),packed_channels=optimize_input)
        stages['shot_detection']=time.perf_counter()-t;t=time.perf_counter()
        detected=_detect_video_faces(detector,arrays,reuse_execution=reuse_execution)
    records=[]
    with spatial_precision():
        for i,faces in enumerate(detected):
            eligible=[b for b in faces if min(b[2]-b[0],b[3]-b[1])>=64]
            box=eligible[0][:4] if len(eligible)==1 else None
            records.append({'frame_index':i,'pts':pts[i],'shot_id':shots[i],'bbox_provenance':'input_detector',
                'face_xyxy':box,'decision':'unique_face' if box is not None else ('missing' if not eligible else 'ambiguous'),
                'all_faces':faces,'eligible_faces':eligible,'input_rgb_sha256':frame_sha256(arrays[i])})
    if len(records)!=len(arrays):raise ValueError('Detector must return every source frame in order')
    torch.cuda.synchronize();stages['detection_records' if overlap_input else 'gpu_face_detection_and_records']=time.perf_counter()-t;t=time.perf_counter()
    frames=torch.from_numpy(arrays).permute(0,3,1,2).cuda()
    torch.cuda.synchronize();stages['rgb_transfer']=time.perf_counter()-t;t=time.perf_counter()
    out=restore_head_sequence(frames,FrameMeta('video',tuple(pts),real_video=True),records,bridge,head,geometry='continuous',side=side)
    torch.cuda.synchronize();stages['head_h3_naf_and_paste']=time.perf_counter()-t;t=time.perf_counter()
    if not out['plan']['used_frames']:raise ValueError('No eligible head segment; not recording an unchanged-copy speed result')
    encode_srgb_h264(destination,out['prediction'],integer_pts,time_bases,rate=rate,reuse_reformatter=reuse_execution)
    torch.cuda.synchronize();stages['output_color_transfer_encode_close']=time.perf_counter()-t
    seconds=time.perf_counter()-start
    return {'seconds':seconds,'stages':stages,'input':arrays,'restoration':out,'detections':records,'shot_report':shot_report,
            'metadata':{'source':str(source),'output':str(destination),'frame_count':len(arrays),'pts':pts,'pts_integer':integer_pts,
                'time_bases':time_bases,'source_rate':[rate.numerator,rate.denominator],'source_color':colors,
                'output_encoding':{'codec':'libx264','crf':18,'preset':'fast','pixel_format':'yuv420p','color':'limited BT709','threads':4},
                'audio_streams':0,'reuse_execution':reuse_execution,'optimize_input':optimize_input,'overlap_input':overlap_input,
                'detector_reuse_events':getattr(detector,'h3ce_reuse_events',[]) if reuse_execution else [],
                'detector_execution':detector.last_report if isinstance(detector,Yolo11BatchExecutor) else None,
                'timing_scope':'Loaded-model input file open through completed output close; includes first-call warmup, with its identical-input forward reused when enabled; excludes loading and post-run audit/artifact saving'}}
