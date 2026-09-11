"""Zero-shot official pretrained SwinIR-L comparison, isolated from H3 training.

Outputs are evaluation artifacts only. No new target, H3 forward, or optimizer.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from h3ce.cache.keys import canonical_json,file_sha256
from h3ce.cache.store import atomic_write,assert_no_links
from h3ce.config import load_config,write_resolved
from h3ce.train.checkpoint import TrainingBudget
from h3ce.train.guard import no_training_guard
from h3ce.train.preflight import open_dataset,require,seed_model
from h3ce.train.pipeline import move_sample
from scripts.gpt_clarity_probe_common import read,bound,code_hashes
from scripts.probe_refiner_pixel_start import check as source_check
from scripts.swinir_baseline_math import tiled_restore,tile_grid,image_metrics
from scripts.verify_loss_balance_initialization import state_hash
from scripts.summarize_supervision_gradients import near
from scripts.summarize_refiner_pixel_start import decision
from scripts.diagnose_detail_frequency import frequency_metrics
from scripts.evaluate_gpt_clarity_probe import edge_mse

KIND='swinir_pretrained_spatial_baseline'
SOURCE=ROOT/'logs/refiner-pixel-start-20260909-v1'
COMPONENT=ROOT/'logs/swinir-baseline-preparation-20260909-v1/components.lock.json'
ARCH={'upscale':4,'in_chans':3,'img_size':64,'window_size':8,'img_range':1.,'depths':[6]*9,'embed_dim':240,'num_heads':[8]*9,'mlp_ratio':2,'upsampler':'nearest+conv','resi_connection':'3conv'}


def codes():
    result=code_hashes()
    for name in ('probe_swinir_baseline.py','swinir_baseline_math.py','probe_refiner_pixel_start.py','summarize_refiner_pixel_start.py','detail_supervision_math.py'):
        result['scripts/'+name]=file_sha256(ROOT/'scripts'/name)
    return result


def check_components():
    component=read(COMPONENT)
    require(component['revision']=='6545850fbf8df298df73d81f3e8cba638787c8bd' and component['dependency_version']=='timm==0.9.16'
            and component['pretrained'] and component['state_key']=='params_ema','Wrong official pretrained component')
    for b in component['sources']+[component['weights'],component['dependency']]+component['runtime_files']:
        require(file_sha256(b['path'])==b['sha256'],'Isolated pretrained component changed')
    return component


def check(p):
    source_check(read(SOURCE/'protocol.json'));check_components()
    require(p['kind']==KIND and p['code_sha256']==codes() and p['architecture']==ARCH and p['images']==16 and p['tile']==256 and p['overlap']==32
            and p['new_training_targets']==0 and p['optimizer_updates']==0 and p['h3_forwards']==0 and not p['trained_base_accepted'],'Baseline scope changed')
    for b in p['evidence']:require(file_sha256(b['path'])==b['sha256'],'Baseline evidence changed')


def declare(folder):
    assert_no_links(folder);require(folder.is_relative_to(ROOT/'logs') and not folder.exists(),'Use a fresh spatial baseline')
    component=check_components();source=read(SOURCE/'protocol.json');source_check(source);summary=read(SOURCE/'summary/summary.json');config=load_config(source['config'])
    with no_training_guard(),TrainingBudget(ROOT/'runs',config.project.budget_seconds,phase=KIND+'_declare') as budget:
        data=open_dataset(config,ROOT,Path(source['manifest']));audit=data.audit(budget_check=budget.check);grids={str(i):tile_grid(*data[i]['x'].shape[-2:]) for i in source['evaluation_indices']}
        folder.mkdir();write_resolved(config,folder/'resolved.yaml')
        p={'kind':KIND,'authorization':'Four-hour autonomous GPU research; actual pretrained spatial prior comparison after random-R objectives failed quality',
           'code_sha256':codes(),'components':bound(COMPONENT),'config':str(folder/'resolved.yaml'),'manifest':source['manifest'],'indices':source['evaluation_indices'],'rois':source['rois'],
           'architecture':ARCH,'images':16,'tile':256,'overlap':32,'tile_grids':grids,'maximum_model_tile_forwards':sum(len(g['positions']) for g in grids.values()),
           'preprocessing':'Existing RGB X, official mirror padding to next window8 including +8 at exact multiples; uniform overlap tile accumulation; crop native x4 output; area downsample to original working canvas',
           'output_role':'Evaluation only; neither native x4 output nor same-canvas output enters training manifests or pseudo-target caches',
           'precision':'Pretrained model FP32 eval/no_grad, no autocast; no image clamp before metrics','new_training_targets':0,'optimizer_updates':0,'h3_forwards':0,'r_forwards':0,
           'gate':source['gate'],'reference_focus_indices':[0,5,14],'all_eight_degraded_reported':True,'source_control_run':source['arms']['cold_pixel'],
           'comparison_scope':'Zero-shot pretrained image-space baseline at same working size; not native x4 SR score, H3 end-to-end inference or base acceptance. Prior pretraining dataset overlap is not established.',
           'trained_base_accepted':False,'independent_validation':False,'data_audit':audit,'evidence':summary['evidence']+source['evidence']+[bound(SOURCE/'protocol.json'),bound(SOURCE/'summary/summary.json'),bound(COMPONENT),bound(folder/'resolved.yaml')]}
        atomic_write(folder/'protocol.json',canonical_json(p));check(p)


def load_model(component):
    sys.path.insert(0,component['isolated_runtime'])
    source=Path(component['source_root'])/'models/network_swinir.py'
    spec=importlib.util.spec_from_file_location('h3ce_official_swinir_baseline',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.SwinIR(**ARCH)
    state=torch.load(component['weights']['path'],map_location='cpu',weights_only=True)
    require(isinstance(state,dict) and 'params_ema' in state and all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state['params_ema'].values()),'Official EMA weight structure differs')
    model.load_state_dict(state['params_ema'],strict=True);model.requires_grad_(False);model.eval()
    require(not any(isinstance(m,(torch.nn.Conv3d,torch.nn.ConvTranspose3d)) for m in model.modules()),'Unexpected temporal architecture')
    return model,state_hash(model.state_dict())


def execute(folder):
    p=read(folder/'protocol.json');check(p);dest=folder/'inference';require(not dest.exists(),'Baseline inference cannot repeat');dest.mkdir();config=load_config(p['config']);torch.set_num_threads(4)
    started=time.monotonic()
    with no_training_guard() as guard,TrainingBudget(ROOT/'runs',config.project.budget_seconds,phase=KIND+'_gpu') as budget:
        require(torch.cuda.is_available(),'Actual pretrained GPU evaluation requires CUDA')
        data=open_dataset(config,ROOT,Path(p['manifest']));data.audit(budget_check=budget.check);seed_model(42);model,model_hash=load_model(check_components());model.cuda()
        rows=[];counter={'tiles':0};before={n:(id(v),v._version) for n,v in model.named_parameters()};torch.cuda.reset_peak_memory_stats()
        def hook(module,args):counter['tiles']+=1;require(counter['tiles']<=p['maximum_model_tile_forwards'],'Declared tile forward cap exceeded')
        handle=model.register_forward_pre_hook(hook)
        try:
            for index in p['indices']:
                budget.check();sample=move_sample(data[index],'cuda');count_before=counter['tiles']
                pred,native,grid=tiled_restore(model,sample['x'][:,:,0],budget.check,p['tile'],p['overlap']);pred=pred.unsqueeze(2)
                require(grid==p['tile_grids'][str(index)] and counter['tiles']-count_before==len(grid['positions']),'Actual tile geometry differs')
                path=dest/f'case_{index:02d}.npz';np.savez_compressed(path,full_prediction=pred.cpu().numpy(),native_x4=native.cpu().numpy())
                x0,y0,x1,y1=p['rois'][str(index)]
                def roi(t):return t[0,:,0,y0:y1,x0:x1].cpu().permute(1,2,0).numpy().copy()
                x,y,out=[roi(t) for t in (sample['x'],sample['y'],pred)]
                rows.append({'index':index,'view_id':sample['view_id'],'group':sample['supervision_group'],'reference_focus':index in p['reference_focus_indices'],'trained_on_view':False,
                    'metrics':image_metrics(pred,sample),'input_metrics':image_metrics(sample['x'],sample),'frequency':frequency_metrics(x,y,out),
                    'edge_mse':{'input':edge_mse(x,y),'prediction':edge_mse(out,y)},'arrays':bound(path),'model_tile_forwards':counter['tiles']-count_before})
                print(f'Official SwinIR image {index} done; {counter["tiles"]} total tiles',flush=True)
        finally:handle.remove()
        require(counter['tiles']==p['maximum_model_tile_forwards'] and before=={n:(id(v),v._version) for n,v in model.named_parameters()}
                and state_hash({n:v.cpu() for n,v in model.state_dict().items()})==model_hash,'Pretrained model changed or coverage incomplete')
        check(p);atomic_write(dest/'report.json',canonical_json({'status':'completed_swinir_pretrained_baseline','cases':rows,'model_state_sha256':model_hash,'model_tile_forwards':counter['tiles'],
            'images':16,'optimizer_updates':0,'new_training_targets':0,'h3_forwards':0,'guard':dict(guard),'elapsed_seconds':time.monotonic()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'budget':budget.snapshot(),'trained_base_accepted':False}))


def verify(folder):
    p=read(folder/'protocol.json');check(p);dest=folder/'summary';require(not dest.exists(),'Summary cannot overwrite');report=read(folder/'inference/report.json');config=load_config(p['config']);torch.set_num_threads(4)
    with no_training_guard() as guard,TrainingBudget(ROOT/'runs',config.project.budget_seconds,phase=KIND+'_cpu') as budget:
        data=open_dataset(config,ROOT,Path(p['manifest']));data.audit(budget_check=budget.check)
        require(report['status']=='completed_swinir_pretrained_baseline' and report['guard']==dict(guard) and report['images']==16 and report['optimizer_updates']==0
                and report['h3_forwards']==0 and report['new_training_targets']==0 and report['model_tile_forwards']==p['maximum_model_tile_forwards'],'Baseline execution accounting differs')
        require([r['index'] for r in report['cases']]==p['indices'],'Baseline coverage differs')
        evidence=[bound(folder/'protocol.json'),bound(folder/'inference/report.json')];rows=[];images={}
        for row in report['cases']:
            budget.check();index=row['index'];sample=data[index];evidence.append(row['arrays'])
            require(row['view_id']==sample['view_id'] and row['group']==sample['supervision_group'] and not row['trained_on_view'],'Baseline input mapping differs')
            require(row['model_tile_forwards']==len(p['tile_grids'][str(index)]['positions']),'Per-image tile count differs')
            with np.load(row['arrays']['path'],allow_pickle=False) as archive:arr={k:archive[k].copy() for k in archive.files}
            require(set(arr)=={'full_prediction','native_x4'} and all(a.dtype==np.float32 and np.isfinite(a).all() for a in arr.values()),'Invalid floating output')
            pred=torch.from_numpy(arr['full_prediction']);native=torch.from_numpy(arr['native_x4']);h,w=sample['x'].shape[-2:]
            require(pred.shape==sample['x'].shape and native.shape==(1,3,h*4,w*4),'Pretrained output geometry differs')
            expected=native.reshape(1,3,h,4,w,4).mean(dim=(3,5)).unsqueeze(2)
            np.testing.assert_allclose(pred.numpy(),expected.numpy(),rtol=0,atol=2e-7)
            current=image_metrics(pred,sample);baseline=image_metrics(sample['x'],sample)
            near(row['metrics'],current,'CPU image-only metrics',rtol=2e-4,atol=2e-7);near(row['input_metrics'],baseline,'CPU original baseline',rtol=2e-4,atol=2e-7)
            x0,y0,x1,y1=p['rois'][str(index)];roi=lambda t:t[0,:,0,y0:y1,x0:x1].permute(1,2,0).numpy().copy()
            x,y,out=[roi(t) for t in (sample['x'],sample['y'],pred)];freq=frequency_metrics(x,y,out);edges={'input':edge_mse(x,y),'prediction':edge_mse(out,y)}
            near(row['frequency'],freq,'CPU ROI frequency');near(row['edge_mse'],edges,'CPU ROI edge')
            rows.append({'index':index,'group':row['group'],'reference_focus':index in p['reference_focus_indices'],'trained_on_view':False,
                'global_mae':current['rgb_global_mae'],'global_mae_input':baseline['rgb_global_mae'],'high_mse':freq['high']['output_error_mse'],'high_mse_input':freq['high']['input_error_mse'],
                'edge_mse':edges['prediction'],'edge_mse_input':edges['input']});images[index]={'input':x,'target':y,'prediction':out}
        gate=decision([{**r,'trained_view':r['reference_focus']} for r in rows])
        gate['passing_reference_degraded_cases']=gate.pop('passing_train_degraded_cases')
        gate['all_reference_degraded_regression_guard']=gate.pop('all_train_degraded_regression_guard')
        all_degraded=[]
        for r in rows:
            if r['group']=='original_degraded':
                changes={k:r[k]/r[k+'_input']-1 for k in ('global_mae','high_mse','edge_mse')}
                all_degraded.append({'index':r['index'],'changes':changes,'numerical_pass':changes['global_mae']<=0 and changes['high_mse']<=-.05 and changes['edge_mse']<=-.05})
        dest.mkdir();import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;artifacts=[]
        for group in ('original_degraded','original_clean'):
            indices=[r['index'] for r in rows if r['group']==group]
            for offset in (0,4):
                fig,axes=plt.subplots(4,4,figsize=(11,11))
                for axrow,index in zip(axes,indices[offset:offset+4]):
                    v=images[index]
                    with np.load(Path(p['source_control_run'])/'fixed_step_0192'/f'case_{index:02d}.npz',allow_pickle=False) as old:control=old['prediction'].copy()
                    for ax,im,label in zip(axrow,(v['target'],v['input'],control,v['prediction']),('Original','Input','Existing cold pixel R192','Pretrained SwinIR-L')):
                        ax.imshow(np.clip(im,0,1));ax.set_title(f'{index}: {label}',fontsize=8);ax.axis('off')
                fig.suptitle(group+'; zero-shot spatial baseline at same canvas; display clipping only');fig.tight_layout();path=dest/f'{group}_{offset//4+1}.png';fig.savefig(path,dpi=140);plt.close(fig);artifacts.append(bound(path))
        for b in evidence+p['evidence']:require(file_sha256(b['path'])==b['sha256'],'Baseline evidence changed')
        check(p);atomic_write(dest/'summary.json',canonical_json({'status':'verified_swinir_pretrained_baseline','rows':rows,'reference_three_case_gate':gate,'all_eight_degraded':all_degraded,
            'all_eight_passing':sum(r['numerical_pass'] for r in all_degraded),'evidence':evidence,'artifacts':artifacts,'source_images':16,'source_model_tile_forwards':report['model_tile_forwards'],
            'new_model_forwards':0,'optimizer_updates':0,'h3_forwards':0,'new_training_targets':0,'trained_base_accepted':False,'independent_validation':False,'budget':budget.snapshot()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=('declare','run','verify'),required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();{'declare':declare,'run':execute,'verify':verify}[a.action](a.output.resolve())
