"""Independent NumPy continuous resampling + saved full-video CPU review."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from h3ce.components import sha256_file
from scripts.benchmark_h3ce_native_video import read
from scripts.benchmark_native_head_fastpath import save
from scripts.motion_quality_math import video_errors

RUN=ROOT/'logs/continuous-head-h3-20260910-v1'


def sample_axis(x,centers,step,axis):
    width=max(1.,step);radius=math.ceil(2*width)
    ii=np.floor(centers).astype(np.int64)[:,None]+np.arange(-radius+1,radius+1)[None]
    u=np.abs((centers[:,None]-ii)/width)
    weights=np.where(u<1,1.5*u**3-2.5*u**2+1,np.where(u<2,-.5*u**3+2.5*u**2-4*u+2,0.))
    weights/=weights.sum(-1,keepdims=True)
    v=np.moveaxis(x,axis,-1)
    out=(np.take(v,ii.clip(0,x.shape[axis]-1),axis=-1)*weights).sum(-1)
    return np.moveaxis(out,-1,axis)


def reference_pack(x,g):
    x0,y0,x1,y1=g['crop_xyxy'];l,_,t,_=g['pad_lrtb'];s=g['scale_xy'][0]
    xx=np.clip(x0+(np.arange(256)+.5-l)/s-.5,x0,x1-1)
    yy=np.clip(y0+(np.arange(256)+.5-t)/s-.5,y0,y1-1)
    return sample_axis(sample_axis(x.astype(np.float64),xx,1/s,-1),yy,1/s,-2).clip(0,1)


def reference_paste(x,delta,g):
    a,b,c,d=g['paste_xyxy'];x0,y0,x1,y1=g['crop_xyxy'];l,_,t,_=g['pad_lrtb'];s=g['scale_xy'][0]
    xx=(np.arange(a,c)-x0+.5)*s-.5+l;yy=(np.arange(b,d)-y0+.5)*s-.5+t
    correction=sample_axis(sample_axis(delta.astype(np.float64),xx,s,-1),yy,s,-2)
    dx=np.minimum(np.arange(a,c)-x0,x1-1-np.arange(a,c))/max(1.,(x1-x0)*.05)
    dy=np.minimum(np.arange(b,d)-y0,y1-1-np.arange(b,d))/max(1.,(y1-y0)*.05)
    feather=np.minimum(dy[:,None],dx[None,:]).clip(0,1)
    out=x.astype(np.float64).copy();out[...,b:d,a:c]+=feather*correction
    return out,feather


def main():
    summary=read(RUN/'summary.json');protocol=read(RUN/'protocol.json')
    for path,digest in summary['artifacts'].items():assert sha256_file(path)==digest
    for path,digest in protocol['bindings'].items():assert sha256_file(path)==digest
    review={'frames_checked':0,'continuous_frames_independent_numpy':0,'cases':[],'updates':0,
        'numerical_tolerance':'Independent float64 Keys reference vs saved FP32 GPU: max abs <= 2e-6 for buckets and pasted RGB; geometry algebra <= 1e-10',
        'lpips':'CPU verifies recorded frame values/means only, no model inference'}
    for case in protocol['cases']:
        folder=RUN/case['kind'];x=np.load(case['input']);y=np.load(case['target']);mask=np.load(case['mask']);a,b,c,d=case['evaluation_only_union_xyxy']
        preds={'input':x};metrics={};maxpack=maxpaste=0.
        for name in protocol['arms']:
            p=np.load(folder/f'{name}_prediction.npy');gs=read(folder/f'{name}_transforms.json')
            assert p.shape==(22,3,432,768) and np.isfinite(p).all()
            if name=='continuous':
                buckets=np.load(folder/f'{name}_buckets.npy');deltas=np.load(folder/f'{name}_delta.npy')
            for i,g in enumerate(gs):
                key='paste_xyxy' if name=='continuous' else 'crop_xyxy';x0,y0,x1,y1=g[key]
                outside=np.ones(x.shape[1:3],bool);outside[y0:y1,x0:x1]=False
                assert np.array_equal(p[i].transpose(1,2,0)[outside],x[i][outside])
                review['frames_checked']+=1
                if name=='continuous':
                    np.testing.assert_allclose(np.array(g['bucket_to_original'])@g['original_to_bucket'],np.eye(3),atol=1e-10)
                    ref=reference_pack(x[i].transpose(2,0,1),g);maxpack=max(maxpack,float(np.max(np.abs(ref-buckets[i]))))
                    pasted,feather=reference_paste(x[i].transpose(2,0,1),deltas[i],g);maxpaste=max(maxpaste,float(np.max(np.abs(pasted-p[i]))))
                    assert np.array_equal(p[i,:,y0:y1,x0:x1][:,feather==0],x[i,y0:y1,x0:x1].transpose(2,0,1)[:,feather==0])
                    review['continuous_frames_independent_numpy']+=1
            preds[name]=p.transpose(0,2,3,1)
        assert maxpack<=2e-6 and maxpaste<=2e-6,(maxpack,maxpaste)
        for name,p in preds.items():metrics[name]=video_errors(p[:,b:d,a:c],y[:,b:d,a:c],mask[:,b:d,a:c])
        scores=read(folder/'lpips_frames.json');recorded=read(folder/'result.json')
        for name in protocol['arms']:
            for key,value in metrics[name].items():assert value==recorded['arms'][name]['metrics'][key]
            assert len(scores[name])==22 and np.isfinite(scores[name]).all()
            assert float(np.mean(scores[name]))==recorded['arms'][name]['lpips_mean']
        row={'kind':case['kind'],'metrics':metrics,'lpips_means':{k:float(np.mean(v)) for k,v in scores.items()},
            'outside_and_zero_feather_exact':True,'max_bucket_error_float64_vs_gpu':maxpack,'max_paste_error_float64_vs_gpu':maxpaste}
        review['cases'].append(row)
        panel=Image.new('RGB',(4*190,3*230+28),(24,24,26));draw=ImageDraw.Draw(panel)
        for col,(label,q) in enumerate([('Input',x),('Integer crop',preds['integer']),('Continuous crop',preds['continuous']),('Original target',y)]):
            draw.text((col*190+4,6),label,fill='white')
            for j,i in enumerate((0,10,21)):
                im=Image.fromarray(np.uint8(q[i,b:d,a:c].clip(0,1)*255+.5));im.thumbnail((188,228));panel.paste(im,(col*190,28+j*230))
        panel.save(folder/'comparison.png')
    blur,clean=review['cases'];m=blur['metrics'];keys=('rgb_mae','high_mse','edge_mse','temporal_difference_mae')
    gates={k:m['continuous'][k]<=m['integer'][k] for k in keys}
    gates['lpips']=blur['lpips_means']['continuous']<=blur['lpips_means']['integer']
    gates['clean']=clean['metrics']['continuous']['rgb_mae']<=.001
    review['retention_gates']=gates;review['retention_all_passed']=all(gates.values())
    review['status']='passed_independent_sampling_and_artifact_review'
    save(RUN/'cpu_review.json',review);print((RUN/'cpu_review.json').read_text(encoding='utf-8'))


if __name__=='__main__':main()
