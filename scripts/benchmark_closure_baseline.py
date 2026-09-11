"""One accepted-baseline closure run with external resource observation; no training."""
from pathlib import Path
import time
IMPORT_START = time.perf_counter()
import sys
import json
import hashlib
from fractions import Fraction
from contextlib import contextmanager, ExitStack
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
import numpy as np
import torch
from h3ce.cache.keys import file_sha256
from h3ce.config import load_config
from h3ce.data.face_preparation import frame_sha256
from h3ce.data.video import iter_video_frames, detect_shots
from h3ce.data.yolo11_face_executor import Yolo11BatchExecutor, PROFILES
from h3ce.infer.video_input_overlap import read_detect_video_overlap
from h3ce.data import continuous_head
from h3ce.infer.head_video_sequence import prepare_sequence
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget
from h3ce.train.guard import no_training_guard
from h3ce.vae.bridge import H3VAEBridge, FrameMeta
from h3ce.vae.int8_kitchen_backend import Int8KitchenH3Backend, KitchenInt8Linear, runtime
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head3 import NAFHead3, NAFHead3Inference
from scripts.research_head_tail2 import strict_spatial
from scripts.video_benchmark_math import blend_weights
from scripts.closure_resource_monitor import wait_idle
IMPORT_SECONDS = time.perf_counter()-IMPORT_START

ARM = 'dual16'
BYTE_PREPARATION = 'direct_bgr'
INPUT_PROBE = ROOT/'logs/detector-host-gpu-probe-20260911-v1'
REUSE = True
FACE_FORWARDS = 2 if ARM=='single32' else 4
FACE_EXPOSURES = 60
OUT = ROOT/'logs/closure-baseline-60-20260912-v1'
BASELINE = ROOT/'logs/detector-host-selected-1s-20260911-v1'
RUNTIME_STATE = {}
DATA = ROOT/'logs/one-second-inference-data-20260911-v1'
RUN = ROOT/'runs/multihead-balanced-mae32-20260910-v1'
QUALITY = ROOT/'logs/multihead-balanced-mae32-video-20260910-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class StageClock:
    def __init__(self):
        self.seconds, self.calls, self.spans = {}, {}, []

    @contextmanager
    def block(self, name):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            ended = time.perf_counter()
            self.seconds[name] = self.seconds.get(name,0.) + ended-started
            self.spans.append(dict(name=name, start=started, end=ended))
            self.calls[name] = self.calls.get(name,0) + 1


def encode(destination_path, frames, integer_pts, time_bases, rate, clock):
    with clock.block('output_container_open'):
        destination = av.open(str(destination_path), 'w')
        stream = destination.add_stream('libx264', rate=rate)
        stream.width,stream.height,stream.pix_fmt = 768,432,'yuv420p'
        exact_clock = Fraction(*time_bases[0])
        assert all(tb == time_bases[0] for tb in time_bases)
        stream.time_base = stream.codec_context.time_base = exact_clock
        stream.options = {'crf':'18','preset':'fast'}
        stream.codec_context.thread_count = 4
        reformatter = av.video.reformatter.VideoReformatter() if REUSE else None
        for field in ('colorspace','color_primaries','color_trc','color_range'):
            setattr(stream.codec_context,field,1)
    try:
        for index in range(len(frames)):
            with clock.block('output_srgb_to_bt709'):
                srgb = frames[index].clamp(0,1)
                linear = torch.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055).pow(2.4))
                bt709 = torch.where(linear<.018,4.5*linear,1.099*linear.pow(.45)-.099)
            with clock.block('output_rgb8_quantize_and_download'):
                rgb = bt709.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
            with clock.block('output_rgb_to_yuv420'):
                raw = av.VideoFrame.from_ndarray(rgb,format='rgb24')
                frame = (raw.reformat(format='yuv420p',dst_colorspace='ITU709') if reformatter is None else
                         reformatter.reformat(raw,format='yuv420p',dst_colorspace='ITU709'))
                frame.pts,frame.time_base = integer_pts[index],exact_clock
            with clock.block('output_h264_encode_and_mux'):
                for packet in stream.encode(frame):
                    destination.mux(packet)
        with clock.block('output_encoder_flush_and_close'):
            for packet in stream.encode():
                destination.mux(packet)
            destination.close()
    except BaseException:
        destination.close()
        raise


def main():
    assert sys.argv[1:] == ['--execute-one-run'] and not OUT.exists()
    OUT.mkdir(parents=True)
    accepted = read(ROOT/'configs/quality_baseline.current.json')
    assert accepted['baseline_id']=='naf-balanced32-direct-bgr-20260911'
    assert accepted['status']=='user_accepted_effect_baseline'
    assert not accepted['development_policy']['optimize_temporal']
    for evidence in accepted['evidence']:
        assert file_sha256(ROOT/evidence['path'])==evidence['sha256'], evidence['path']
    for path,digest in read(BASELINE/'protocol.json')['bindings'].items():
        assert file_sha256(path)==digest, path
    assert file_sha256(accepted['checkpoint']['path'])==accepted['checkpoint']['sha256']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_record = read(DATA/'protocol.json')
    assert source_record['status'] == 'exact60_frames_one_second_input_verified'
    source = Path(source_record['input'])
    assert file_sha256(source) == source_record['input_sha256']
    train = read(RUN/'training_report.json')
    quality = read(QUALITY/'cpu_review.json')
    assert train['status'] == 'completed32' and train['steps'] == 32
    assert quality['status'] == 'passed360_cpu_frame_geometry_metrics_pts' and quality['clean_mae_gate']
    cp = train['checkpoint']
    assert file_sha256(cp['path']) == cp['sha256']
    files = [Path(__file__),DATA/'protocol.json',source,RUN/'training_report.json',RUN/'training_contract.json',
             RUN/'execution_override.json',Path(cp['path']),QUALITY/'cpu_review.json',
             ROOT/'docs/DETECTOR_HOST_PROTOCOL_20260911.md',ROOT/'h3ce/data/face_preparation.py',INPUT_PROBE/'report.json',INPUT_PROBE/'protocol.json',ROOT/'h3ce/data/decode.py',ROOT/'h3ce/data/yolo11_face_executor.py',ROOT/'h3ce/data/yolo11_face_reuse.py',ROOT/'h3ce/infer/video_input_overlap.py',ROOT/'h3ce/infer/head_video_file.py',
             ROOT/'h3ce/infer/head_video_sequence.py',ROOT/'h3ce/data/continuous_head.py',ROOT/'h3ce/data/stable_head.py',
             ROOT/'h3ce/data/video.py',ROOT/'h3ce/data/yolo11_face_batch.py',ROOT/'h3ce/data/detect_yolo11.py',
             ROOT/'h3ce/vae/int8_kitchen_backend.py',ROOT/'h3ce/vae/int8_convrot_backend.py',ROOT/'h3ce/vae/bridge.py',
             ROOT/'configs/project.int8.yaml',ROOT/'configs/components.int8.lock.json',ROOT/'configs/convrot_cuda_runtime.lock.json',
             ROOT/'scripts/research_naf_head3.py',ROOT/'scripts/research_nafnet_gopro32.py',ROOT/'scripts/video_benchmark_math.py']
    files += [ROOT/'configs/quality_baseline.current.json',BASELINE/'protocol.json',BASELINE/'summary.json',
              ROOT/'scripts/closure_resource_monitor.py', ROOT/'scripts/run_closure_baseline.py',
              ROOT/'scripts/review_closure_baseline.py', ROOT/'docs/CLOSURE_BASELINE_TEST_PROTOCOL_20260912.md']
    protocol = dict(authorization='User requested closure and CPU/GPU observation, then explicitly said start. One accepted-baseline 60-frame run; no candidate optimization.', arm=ARM,reuse_execution=REUSE,optimize_input=True,overlap_input=True,input_execution='read_detect_overlap',byte_preparation=BYTE_PREPARATION,direct_frame_hash=True,
                    checkpoint=cp,selected_run=RUN.name,source=str(source),source_sha256=source_record['input_sha256'],
                    frames=60,fps=60,source_seconds=1.,output_hw=[432,768],head_side=256,full_runs=1,
                    detector_batch=PROFILES[ARM][0],detector_workers=PROFILES[ARM][1],naf_batch=8,extra_manual_warmup=0,
                    h3='Actual Comfy Kitchen CUDA INT8 GEMM, user-selected INT8 ConvRot source, frozen native22/overlap5, no cached model output.',
                    limits=dict(detector_explicit_batches=FACE_FORWARDS,detector_actual_forwards=FACE_FORWARDS,detector_exposures_with_warmup=FACE_EXPOSURES,
                                encode=4,decode=4,upstream_int8_linears=576,backbone=8,tail=8,reference=0,lpips=0,updates=0),
                    historical_quality=dict(clean_protection_passed=True,numerical_and_geometry_review_passed=True,
                                            reference_is_user_accepted=True,temporal_optimization=False),
                    timing='Exclusive synchronized wall totals; input decode/shot/detector overlap is one stage, internal CPU/worker/CUDA intervals are not added. Loaded-model source open through completed output close includes first-use backend initialization and original NMS warmup.',
                    exclusions=['input preparation','preflight hash/idle gates','model loading','post-run CPU evidence saving and audit'],
                    imports_seconds=IMPORT_SECONDS,bindings={str(p):file_sha256(p) for p in files})
    save(OUT/'protocol.json',protocol)
    wait_idle(OUT,'before_load')
    clock = StageClock()
    counts = dict(encode=0,decode=0,backbone=0,tail=0,reference=0)
    face_calls,kernel_calls = [],[]
    RUNTIME_STATE.update(counts=counts, kernel_calls=kernel_calls)
    with no_training_guard() as guard, ExitStack() as resources:
        load_started = time.perf_counter()
        cfg = load_config(ROOT/'configs/project.int8.yaml')
        cuda,_,runtime_identity = runtime()
        bridge = H3VAEBridge(Int8KitchenH3Backend.from_locked(project_root=ROOT,
            weights='models/minimax_h3_video_vae_int8_convrot.safetensors',components_lock='configs/components.int8.lock.json'))
        assert bridge.inspect_contract().native == read(RUN/'execution_override.json')['h3_contract']['native']
        detector = resources.enter_context(Yolo11BatchExecutor(cfg,ROOT,profile=ARM,device='cuda:0',byte_preparation=BYTE_PREPARATION))
        backbone,component_info = load_model()
        head = NAFHead3(backbone)
        state = CheckpointManager(ROOT/'runs',RUN,contract=read(RUN/'training_contract.json')).read(Path(cp['path']))
        assert state['step'] == 32
        head.tail.load_state_dict(state['model'],strict=True)
        del state
        head.cuda().eval().requires_grad_(False)
        inference_wrapper = NAFHead3Inference(head)  # Validates the official-reference reuse contract.
        del inference_wrapper
        pack = bridge.current_codec_pack()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-load_started
        save(OUT/'load.json',dict(model_load_seconds=load_seconds,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
             pyav=av.__version__,h3_contract=bridge.inspect_contract().native,runtime=runtime_identity,
             detector_contracts=[w.resource.detector.contract_id for w in detector.workers],naf_components=component_info,naf_constructor_cpu_shape_forwards=1))
        def increment(name):
            counts[name] += 1
            assert counts[name] <= protocol['limits'][name], counts
        for name,module in [('backbone',head.backbone),('tail',head.tail),('reference',head.reference)]:
            module.register_forward_pre_hook(lambda m,a,name=name: increment(name))
        def wrap(fn,name):
            def call(*args,**kwargs):
                increment(name)
                return fn(*args,**kwargs)
            return call
        bridge.backend.encode_mean_raw = wrap(bridge.backend.encode_mean_raw,'encode')
        bridge.backend.decode_raw = wrap(bridge.backend.decode_raw,'decode')
        original_kernels = {name:getattr(cuda._C,name) for name in ('cutlass_int8_dequant','cublas_gemm_int8')}
        for name,fn in original_kernels.items():
            def kernel(*args,_name=name,_fn=fn,**kwargs):
                result = _fn(*args,**kwargs)
                kernel_calls.append(dict(kernel=_name,success=bool(result) if _name=='cutlass_int8_dequant' else True))
                return result
            setattr(cuda._C,name,kernel)
        modules = [head,bridge.backend.model]
        versions = lambda: [{n:(id(p),p._version) for n,p in module.named_parameters()} for module in modules]
        initial_versions = versions()
        wait_idle(OUT,'before_inference')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        whole_started = time.perf_counter()
        with clock.block('input_read_shots_and_yolo_overlap'):
            prepared = read_detect_video_overlap(source,detector,max_frames=60,long_edge=768,
                                                reuse_execution=True,optimize_input=True)
            arrays,pts = prepared['arrays'],prepared['pts']
            integer_pts,time_bases = prepared['integer_pts'],prepared['time_bases']
            color,rate = prepared['color'],prepared['rate']
            shots,shot_report,detected = prepared['shots'],prepared['shot_report'],prepared['faces']
            assert arrays.shape==(60,432,768,3) and pts==[i/60 for i in range(60)]
            del prepared
        with clock.block('detection_filter_and_records'):
            records = []
            for index,faces in enumerate(detected):
                eligible = [b for b in faces if min(b[2]-b[0],b[3]-b[1])>=64]
                box = eligible[0][:4] if len(eligible)==1 else None
                records.append(dict(frame_index=index,pts=pts[index],shot_id=shots[index],bbox_provenance='input_detector',
                    face_xyxy=box,decision='unique_face' if box is not None else ('missing' if not eligible else 'ambiguous'),
                    all_faces=faces,eligible_faces=eligible,input_rgb_sha256=frame_sha256(arrays[index])))
        with clock.block('rgb_upload'):
            frames = torch.from_numpy(arrays).permute(0,3,1,2).cuda()
        with clock.block('stable_dynamic_geometry'):
            plan = prepare_sequence(records,pts,(432,768),geometry='continuous',side=256)
            assert plan['used_frames']==list(range(60)) and not plan['skipped_frames']
            assert [(c['start'],c['stop']) for c in plan['chunks']]==[(0,22),(17,39),(34,56),(51,60)]
        with clock.block('continuous_head_crop_and_pad'):
            buckets = torch.zeros((60,3,256,256),device='cuda')
            with strict_spatial():
                for index in plan['used_frames']:
                    buckets[index:index+1] = continuous_head.pack_frame(frames[index:index+1],plan['transforms'][index])
        with clock.block('native_buffer_setup'):
            total,weights = torch.zeros_like(buckets),torch.zeros(60,device='cuda')
            context,chunk_outputs = [],[]
        for chunk in plan['chunks']:
            start,stop = chunk['start'],chunk['stop']
            with clock.block('h3_encode'):
                meta = FrameMeta('video',tuple(pts[start:stop]),real_video=True,shot_id=str(chunk['source_shot_id']))
                latent = bridge.encode_rgb(buckets[start:stop].permute(1,0,2,3)[None].contiguous(),meta)
            with clock.block('h3_decode'):
                native_chunk = bridge.decode_latent(latent,grad=False,codec_pack=pack)[0].permute(1,0,2,3)
            with clock.block('native_overlap_accumulation'):
                assert native_chunk.shape==buckets[start:stop].shape and torch.isfinite(native_chunk).all()
                weight = torch.from_numpy(blend_weights(chunk)).cuda()
                total[start:stop] += native_chunk * weight[:,None,None,None]
                weights[start:stop] += weight
                context.append(dict(**chunk,pts=list(meta.pts),valid_frames=latent.valid_frames,padded_frames=latent.padded_frames))
                chunk_outputs.append(native_chunk)
                del latent
        with clock.block('native_normalization_and_spatial_condition'):
            assert torch.allclose(weights,torch.ones_like(weights),rtol=1e-6,atol=1e-6)
            native = total/weights[:,None,None,None]
            condition = .5*(buckets+native.clamp(0,1))
            delta,output = torch.zeros_like(buckets),frames.clone()
        with strict_spatial():
            for start in range(0,60,8):
                selected = list(range(start,min(start+8,60)))
                with clock.block('naf_backbone_and_official_reference'):
                    cached = head.features(condition[selected])
                with clock.block('naf_adapted_tail'):
                    positive = head.tail(cached['features'],cached['skip_mid'],cached['skip_full'])
                with clock.block('naf_residual_assembly'):
                    delta[selected] = positive-cached['official_ending']
                    assert torch.isfinite(delta[selected]).all()
                    del cached,positive
            with clock.block('continuous_inverse_paste'):
                for index in plan['used_frames']:
                    output[index:index+1] = continuous_head.paste_delta(frames[index:index+1],delta[index:index+1],plan['transforms'][index])
        with clock.block('output_validation'):
            assert output.shape==(60,3,432,768) and torch.isfinite(output).all()
        destination = OUT/'output_1s.mp4'
        encode(destination,output,integer_pts,time_bases,rate,clock)
        torch.cuda.synchronize()
        whole_seconds = time.perf_counter()-whole_started
        peak_allocated,peak_reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
        save(OUT/'stage_spans.json', dict(clock='time.perf_counter Windows system monotonic shared by processes',
             phases=[dict(name='model_load',start=load_started,end=load_started+load_seconds),
                     dict(name='whole_inference',start=whole_started,end=whole_started+whole_seconds)],stages=clock.spans))
        for name,fn in original_kernels.items():
            setattr(cuda._C,name,fn)
        assert counts==dict(encode=4,decode=4,backbone=8,tail=8,reference=0)
        executor_report = detector.last_report
        assert executor_report['byte_preparation']==BYTE_PREPARATION
        face_calls = [c for w in executor_report['workers'] for c in w['calls']]
        assert executor_report['counts']==dict(forwards=FACE_FORWARDS,exposures=60)
        assert len(face_calls)==FACE_FORWARDS and all(c['completed'] and c['dtype']=='torch.float32' for c in face_calls)
        assert sum(c['shape'][0] for c in face_calls)==FACE_EXPOSURES
        int8_calls = sum(m.calls for m in bridge.backend.model.modules() if isinstance(m,KitchenInt8Linear))
        assert int8_calls==576
        kernel_counts = {name:sum(c['kernel']==name for c in kernel_calls) for name in original_kernels}
        kernel_successes = {name:sum(c['kernel']==name and c['success'] for c in kernel_calls) for name in original_kernels}
        assert kernel_successes['cutlass_int8_dequant']+kernel_successes['cublas_gemm_int8']==576
        assert versions()==initial_versions
        summed = sum(clock.seconds.values())
        assert 0 <= whole_seconds-summed <= max(.2,whole_seconds*.05)
        timing = dict(status='one_full_fast_head_inference_completed',source_frames=60,source_seconds=1.,
            whole_file_seconds=whole_seconds,frames_per_processing_second=60/whole_seconds,model_load_seconds=load_seconds,
            model_load_plus_inference_seconds=load_seconds+whole_seconds,exclusive_stage_seconds=clock.seconds,stage_calls=clock.calls,
            summed_stage_seconds=summed,orchestration_and_timer_seconds=whole_seconds-summed,counts=counts,
            actual_yolo_forwards=len(face_calls),yolo_exposures_including_warmup=FACE_EXPOSURES,actual_int8_linears=int8_calls,
            kernel_calls=kernel_counts,kernel_successes=kernel_successes,peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,output=str(destination),output_sha256=file_sha256(destination),
            gpu=torch.cuda.get_device_name(),guard=dict(guard),frozen_parameters_unchanged=True,additional_full_runs=0,
            baseline_id=accepted['baseline_id'],new_quality_adoption=False,resource_monitor_external=True,cpu_review_pending=True)
        save(OUT/'timing.json',timing)
        save(OUT/'face_model_calls.json',face_calls)
        save(OUT/'detector_executor.json',executor_report)
        save(OUT/'detector_reuse_events.json',[e for w in executor_report['workers'] for e in w['reuse_events']])
        save(OUT/'kernel_calls.json',kernel_calls)
        save(OUT/'input_detections.json',records)
        save(OUT/'plan.json',plan)
        save(OUT/'context.json',context)
        save(OUT/'metadata.json',dict(pts=pts,pts_integer=integer_pts,time_bases=time_bases,source_color=color,source_rate=str(rate),shot_report=shot_report))
        for name,value in [('input',frames),('prediction',output),('buckets',buckets),('native',native),('condition',condition),('delta',delta),('weights',weights)]:
            np.save(OUT/f'{name}.npy',value.cpu().numpy())
        for index,value in enumerate(chunk_outputs):
            np.save(OUT/f'native_chunk_{index}.npy',value.cpu().numpy())
        files = [p for p in OUT.iterdir() if p.is_file() and p.name!='summary.json']
        save(OUT/'summary.json',dict(status='one_full_fast_head_inference_and_evidence_saved',timing=timing,
                                   artifacts={str(p):file_sha256(p) for p in files}))
        print(f'ONE {ARM} head 1s clip: {whole_seconds:.6f}s; model load {load_seconds:.6f}s; kernels {kernel_successes}',flush=True)


if __name__=='__main__':
    try:
        with TrainingBudget(ROOT/'runs',172800,phase='closure_baseline60_resource_20260912') as budget:
            budget.check()
            main()
        save(OUT/'budget_after.json',budget.snapshot())
    except BaseException as error:
        if OUT.exists():
            save(OUT/'failure.json',dict(error=repr(error),automatic_restart=False,runtime_state=RUNTIME_STATE))
        raise
