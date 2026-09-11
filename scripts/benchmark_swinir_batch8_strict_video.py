"""One input-only full video pass through the current spatial-delta + native H3 candidate."""
from pathlib import Path
import sys,os,json,time,subprocess,threading,hashlib
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import av
from h3ce.cache.keys import file_sha256
from h3ce.train.checkpoint import TrainingBudget,CheckpointManager
from h3ce.train.guard import no_training_guard
from h3ce.train.pipeline import outer_box_feather,decoded_delta
from h3ce.vae.aitoolkit_h3_backend import AIToolkitH3Backend
from h3ce.vae.bridge import H3VAEBridge,FrameMeta
from h3ce.data.video import detect_shots
from scripts.probe_swinir_baseline import load_model,check_components
from scripts.swinir_baseline_math import tiled_restore,tile_grid
from scripts.video_benchmark_math import chunk_plan,blend_weights

OUT=ROOT/'logs/h3ce-video-swinir-batch8-strict-20260909-v1'
SOURCE=ROOT/'dataset/external_candidates/pexels_854620/blurry_people_4s_1080p25.mp4'
PARENT=ROOT/'runs/pretrained-spatial-delta-fullbatch-20260909-v1'
REFERENCE=ROOT/'logs/vosr2-video-baseline-20260909-v1/user_selected_reference.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def gpu_sample():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],text=True).strip()
    v=raw.splitlines()[0].split(',')
    return {'monotonic':time.perf_counter(),'utilization_percent':float(v[0]),'memory_mib':float(v[1]),
            'temperature_c':float(v[2]),'power_w':float(v[3]),'sm_mhz':float(v[4])}

def wait_idle(label):
    samples=[];good=0;start=time.perf_counter()
    while time.perf_counter()-start<180:
        s=gpu_sample();samples.append(s);good=good+1 if s['utilization_percent']<=10 else 0
        save(OUT/f'{label}_idle.json',samples)
        if good>=10:return
        if len(samples)%20==0:print(f'Waiting for idle GPU: {label}, utilization={s["utilization_percent"]}%',flush=True)
        time.sleep(1)
    raise RuntimeError('GPU did not remain idle for ten samples; no model forward started')

class Telemetry:
    def __init__(self):self.rows=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop.is_set():
            try:self.rows.append(gpu_sample())
            except Exception as e:self.rows.append({'error':str(e)})
            if self.stop.wait(2):break
    def start(self):self.thread.start()
    def finish(self):self.stop.set();self.thread.join(timeout=10);save(OUT/'gpu_telemetry.json',self.rows)

def main():
    assert sys.argv[1:]==['--execute-one-run'] and not OUT.exists(),'One new run only; never overwrite or auto-repeat'
    assert file_sha256(SOURCE)=='0b32a2894f6dbc0e6a5d8eecaa0e6e9198ca352cf1dee5694ef6ddd2e673e1b1'
    reference=read(REFERENCE);assert reference['whole_video_seconds']==44.16887560000032
    component=check_components();checkpoint=read(PARENT/'report.json')['checkpoint']
    assert file_sha256(checkpoint['path'])==checkpoint['sha256']
    parent_protocol=read(ROOT/'logs/pretrained-spatial-delta-fullbatch-20260909-v1/protocol.json')
    for name,digest in parent_protocol['code_sha256'].items():assert file_sha256(ROOT/name)==digest,name
    # CPU-only source inspection; repeats below inside the full-video timer.
    with av.open(str(SOURCE)) as c:
        assert not c.streams.audio
        meta=[(f.width,f.height,float(f.pts*f.time_base)) for f in c.decode(video=0)]
    assert meta==[(1920,1080,i/25) for i in range(100)]
    OUT.mkdir();torch.set_num_threads(4)
    files=[Path(__file__),ROOT/'scripts/video_benchmark_math.py',ROOT/'components.lock.json',REFERENCE,
           PARENT/'training_contract.json',Path(checkpoint['path']),SOURCE,ROOT/'logs/h3ce-video-swinir-fullframe-20260909-v1/summary.json',ROOT/'logs/h3ce-video-swinir-fullframe-20260909-v1/review.json',ROOT/'logs/h3ce-video-swinir-batch8-20260909-v1/summary.json',ROOT/'logs/swinir-batch-numerics-20260909-v1/review.json']
    protocol={'kind':'swinir_batch8_strict_convolution_video_ablation','authorization':'New active four-hour autonomous optimization goal; independent bounded inference ablation C; disable cuDNN TF32 only during spatial batch8; keep H3 precision unchanged; reuse completed B control',
      'full_runs':1,'warmup_forwards':0,'unique_frames':100,'fps':25,'input_hw':[1080,1920],'output_hw':[432,768],
      'native_canvas_hw':[448,768],'padding':'reflect bottom16, crop to432 after native processing',
      'chunk_frames':22,'overlap':5,'tail':'repeat last at bridge to17n+5; discard padded outputs; preserve source PTS',
      'formula':'P=X+S128(area_quarter(X))-S0(area_quarter(X)); Y=X+feather*(D(E(clamp(P)))-D(E(X)))',
      'candidate':'SwinIR-L pretrained spatial delta fullbatch128; frozen original H3 native video; full-frame research candidate, no character adapters',
      'spatial_precision':'FP32 no autocast; whole-frame model native forward/window padding replacing tiled wrapper and its extra mirror padding; no learned temporal branch',
      'color':'same PyAV bicubic768x432 RGB decode as reference; explicit BT709 transfer to sRGB for H3, inverse before BT709 output encoding',
      'runtime_cache':'Only current-pass input-dependent spatial predictions retained for overlapping chunks; no cross-run or disk model-output/latent cache',
      'blending':'Only same source frame/PTS, complementary linear ramps in overlap; no blending across detected cuts',
      'maximum_spatial_forwards':26,'maximum_h3_encodes':12,'maximum_h3_decodes':12,
      'optimizer_updates':0,'new_targets':0,'timing':'After loaded and idle: source open through finished output close; includes decode/resize/color/shot detection/padding/spatial/H3/blending/transfers/finite checks/encoding; excludes model loading and post-run validation',
      'encoding':{'codec':'libx264','crf':18,'preset':'fast','pix_fmt':'yuv420p','threads':4,'fps':25,'matrix':'BT709'},
      'comparison_reference':reference,'checkpoint':checkpoint,'bindings':{str(p):file_sha256(p) for p in files},
      'spatial_native_input_hw':[112,192], 'spatial_batch_size':8, 'changed_factor':'Only disable cuDNN TF32 during spatial batch8; restore original flag before H3; same weights, geometry and output','gpu_idle_gate':'Ten consecutive one-second samples <=10% before load and before inference; max180sec per gate; no extra forward',
      'quality_acceptance':False,'product_tier_acceptance':False}
    save(OUT/'protocol.json',protocol)
    counts={'spatial_forwards':0,'encodes':0,'decodes':0,'chunks':0};stages={};telemetry=Telemetry();handles=[]
    try:
      with TrainingBudget(ROOT/'runs',172800,phase='swinir_batch8_strict_video_ablation') as budget,no_training_guard() as guard:
        wait_idle('before_load');budget.check();start=time.perf_counter()
        base,_=load_model(component);candidate,_=load_model(component)
        manager=CheckpointManager(ROOT/'runs',PARENT,contract=read(PARENT/'training_contract.json'))
        state=manager.read(Path(checkpoint['path']));assert state['step']==128
        candidate.load_state_dict(state['model'],strict=True);del state
        base.cuda().eval().requires_grad_(False);candidate.cuda().eval().requires_grad_(False)
        bridge=H3VAEBridge(AIToolkitH3Backend.from_locked(project_root=ROOT,weights='models/minimax_h3_video_vae_fp16.safetensors'))
        pack=bridge.current_codec_pack();models=(base,candidate,bridge.backend.model)
        versions=[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models]
        torch.cuda.synchronize();save(OUT/'load.json',{'seconds':time.perf_counter()-start,'torch':torch.__version__,'cuda':torch.version.cuda,'h3_contract':bridge.inspect_contract().native})
        def hook(m,args):
            counts['spatial_forwards']+=1
            assert counts['spatial_forwards']<=26
        handles=[m.register_forward_pre_hook(hook) for m in (base,candidate)]
        wait_idle('before_inference');budget.check()
        (OUT/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True),encoding='utf-8')
        print('Starting the ONE full 100-frame native video run',flush=True)
        telemetry.start();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();t=start
        arrays=[];pts=[];integer_pts=[];timebases=[]
        with av.open(str(SOURCE)) as source:
            color={k:getattr(source.streams.video[0].codec_context,k) for k in ('colorspace','color_range','color_primaries','color_trc')}
            assert color=={'colorspace':1,'color_range':1,'color_primaries':1,'color_trc':1}
            for i,f in enumerate(source.decode(video=0)):
                assert i<100
                arrays.append(f.reformat(width=768,height=432,format='rgb24',interpolation='BICUBIC').to_ndarray())
                pts.append(float(f.pts*f.time_base));integer_pts.append(f.pts);timebases.append(f.time_base)
        assert pts==[i/25 for i in range(100)]
        shots,shot_report=detect_shots([a.astype(np.float32)/255 for a in arrays]);chunks=chunk_plan(shots)
        assert len(chunks)==6,'This bounded single-shot protocol must be redeclared if cuts are detected'
        raw=torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).float().cuda()/255;del arrays
        linear=torch.where(raw<.081,raw/4.5,((raw+.099)/1.099).pow(1/.45))
        srgb=torch.where(linear<=.0031308,linear*12.92,1.055*linear.pow(1/2.4)-.055).clamp(0,1);del raw,linear
        x=F.pad(srgb,(0,0,0,16),mode='reflect').permute(1,0,2,3).unsqueeze(0).contiguous();del srgb
        spatial=torch.empty_like(x)
        valid=torch.zeros(1,1,1,448,768,device='cuda');valid[...,:432,:]=1;feather=outer_box_feather(valid)
        torch.cuda.synchronize();stages['decode_resize_color_shot_pad']=time.perf_counter()-t;t=time.perf_counter()
        previous_cudnn_tf32=torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32=False
        for i in range(0,100,8):
            end=min(i+8,100)
            frames=x[0,:,i:end].permute(1,0,2,3).contiguous()
            quarter=F.interpolate(frames,scale_factor=.25,mode='area')
            b=base(quarter);c=candidate(quarter)
            assert b.shape==c.shape==frames.shape and torch.isfinite(b).all() and torch.isfinite(c).all()
            spatial[:,:,i:end]=(x[:,:,i:end]+(c-b).permute(1,0,2,3).unsqueeze(0)).clamp(0,1)
            del quarter,b,c,frames
            if end%40==0 or end==100:print(f'Spatial batch8 frames {end}/100; elapsed {time.perf_counter()-start:.2f}s',flush=True)
        torch.backends.cudnn.allow_tf32=previous_cudnn_tf32
        torch.cuda.synchronize();stages['spatial']=time.perf_counter()-t;t=time.perf_counter()
        summed=torch.zeros_like(x);weights=torch.zeros(100,device='cuda');chunk_rows=[]
        for index,ch in enumerate(chunks):
            a,b=ch['start'],ch['stop'];chunk_start=time.perf_counter()
            meta=FrameMeta('video',tuple(pts[a:b]),real_video=True,shot_id=str(ch['shot']))
            zi=bridge.encode_rgb(x[:,:,a:b],meta);zp=bridge.encode_rgb(spatial[:,:,a:b],meta);counts['encodes']+=2
            negative=bridge.decode_latent(zi,grad=False,codec_pack=pack);positive=bridge.decode_latent(zp,grad=False,codec_pack=pack);counts['decodes']+=2
            y=decoded_delta(x[:,:,a:b],positive,negative,feather)
            w=torch.from_numpy(blend_weights(ch)).cuda();summed[:,:,a:b]+=y*w[None,None,:,None,None];weights[a:b]+=w
            torch.cuda.synchronize();counts['chunks']+=1
            chunk_rows.append({**ch,'padded_frames':zi.padded_frames,'seconds':time.perf_counter()-chunk_start})
            del zi,zp,negative,positive,y,w
            print(f'Native H3 chunk {index+1}/6 complete; elapsed {time.perf_counter()-start:.2f}s',flush=True)
        assert counts=={'spatial_forwards':26,'encodes':12,'decodes':12,'chunks':6}
        assert torch.allclose(weights,torch.ones_like(weights),rtol=1e-6,atol=1e-6)
        result=(summed/weights[None,None,:,None,None])[...,:432,:].clamp(0,1)
        assert torch.isfinite(result).all()
        torch.cuda.synchronize();stages['native_h3_and_blend']=time.perf_counter()-t;t=time.perf_counter()
        path=OUT/'run_01.mp4'
        with av.open(str(path),'w') as dest:
            stream=dest.add_stream('libx264',rate=25);stream.width=768;stream.height=432;stream.pix_fmt='yuv420p'
            stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=4
            for key,value in color.items():setattr(stream.codec_context,key,value)
            for i in range(100):
                s=result[0,:,i];linear=torch.where(s<=.04045,s/12.92,((s+.055)/1.055).pow(2.4))
                bt709=torch.where(linear<.018,4.5*linear,1.099*linear.pow(.45)-.099)
                rgb=bt709.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                frame=av.VideoFrame.from_ndarray(rgb,format='rgb24').reformat(format='yuv420p',dst_colorspace='ITU709')
                frame.pts=integer_pts[i];frame.time_base=timebases[i]
                for packet in stream.encode(frame):dest.mux(packet)
            for packet in stream.encode():dest.mux(packet)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;stages['output_color_transfer_encode']=time.perf_counter()-t
        telemetry.finish();budget.check()
        assert versions==[{n:(id(v),v._version) for n,v in m.named_parameters()} for m in models]
        assert all(not p.requires_grad and p.grad is None for m in models for p in m.parameters())
        with av.open(str(path)) as c:actual=[(f.width,f.height,float(f.pts*f.time_base)) for f in c.decode(video=0)]
        assert actual==[(768,432,i/25) for i in range(100)]
        np.savez_compressed(OUT/'float_selected_frames.npz',indices=np.array([0,49,99]),prediction=result[:,:, [0,49,99]].cpu().numpy(),input=x[:,:, [0,49,99],:432,:].cpu().numpy())
        save(OUT/'shot_detection.json',shot_report)
        save(OUT/'summary.json',{'status':'one_full_native_video_run_verified','frames':100,'seconds':elapsed,'seconds_per_frame':elapsed/100,'fps':100/elapsed,
            'user_reference_seconds':reference['whole_video_seconds'],'ratio_to_user_reference':elapsed/reference['whole_video_seconds'],
            'numerically_meets_user_time_target':elapsed<=reference['whole_video_seconds'],'reference_environment_caveat':True,
            'stages':stages,'counts':counts,'chunks':chunk_rows,'guard':dict(guard),'output':str(path),'sha256':file_sha256(path),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'unique_pts':pts,'output_verified':True,'optimizer_updates':0,'additional_runs':0,'budget':budget.snapshot()})
        (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True),encoding='utf-8')
        print(f'ONE run finished: {elapsed:.6f}s, ratio to chosen VOSR2 {elapsed/reference["whole_video_seconds"]:.4f}',flush=True)
    except BaseException as error:
        if telemetry.thread.is_alive():telemetry.finish()
        save(OUT/'failure.json',{'type':type(error).__name__,'message':str(error),'counts':counts});raise
    finally:
        for handle in handles:handle.remove()

if __name__=='__main__':main()
