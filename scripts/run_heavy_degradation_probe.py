"""One declared 64-update q=0.09 experiment; frozen H3, existing audited pixels."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''): sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from h3ce.config import load_config
from h3ce.cache.keys import canonical_json, file_sha256, digest
from h3ce.cache.store import atomic_write
from h3ce.train.checkpoint import TrainingBudget
from h3ce.train.guard import no_training_guard
from h3ce.train.preflight import open_dataset
from h3ce.train.sampler import StatefulSampler
from scripts.run_scaled_training_probe import train, evaluate, check as training_check
from scripts.run_lr_length_probe import read, require
from scripts.evaluate_bootstrap_checkpoint import read_committed, view_geometry
from scripts.verify_scaled_length_probe import saved_at
from scripts.verify_loss_balance_initialization import assert_equal, check_optimizer, state_hash

REFERENCE=ROOT/'runs/20260907T223915Z-train-e1c845e9'

def heavy_config(source):
    require(source['data']['degradation']['resolution']['ratio_range']==[.18,.18], 'Expected medium control')
    require(source['native_temporal']['mode']=='frozen' and not source['vae']['encoder_trainable']
        and source['training']['stages']['bootstrap_pixel']['lr']==1e-5
        and source['training']['gradient_accumulation']==4 and source['project']['seed']==42, 'Wrong control settings')
    result=copy.deepcopy(source)
    result['data']['degradation']['resolution']['ratio_range']=[.09,.09]
    return result

def compare_configs(reference,candidate):
    require(candidate==heavy_config(reference), 'Only q may change; keep targets, losses, LR and freeze settings')

def compare_views(reference,candidate):
    require(len(reference.views)==len(candidate.views)==16, 'Expected sixteen training pairs')
    rows=[]
    for old,new in zip(reference.views,candidate.views):
        require(view_geometry(old)==view_geometry(new), 'Source order, geometry or target latent changed')
        a=reference.variants[old['variant_id']];b=candidate.variants[new['variant_id']]
        require(a['clean_pair']==b['clean_pair'], 'Pair ordering changed')
        x0,x1=np.load(old['x_crop_path']),np.load(new['x_crop_path'])
        y0,y1=np.load(old['y_crop_path']),np.load(new['y_crop_path'])
        require(np.array_equal(y0,y1),'Target pixels changed')
        mask=np.load(new['pad_valid_map'])>0
        # Valid masks have H,W or H,W,1 shape; flatten only actual RGB elements.
        mask=np.broadcast_to(mask[...,None] if mask.ndim==2 else mask,y1.shape)
        m0=float(np.abs(x0-y0)[mask].mean());m1=float(np.abs(x1-y1)[mask].mean())
        if b['clean_pair']:
            require(np.array_equal(x0,y0) and np.array_equal(x1,y1), 'Clean input changed')
        else: require(m1>m0, 'Stronger input must increase measured target difference in every case')
        rows.append({'source':Path(candidate.sources[new['asset_id']]['path']).name,'mode':new['mode'],
            'clean_pair':b['clean_pair'],'medium_valid_mae':m0,'heavy_valid_mae':m1,'lr_hw':b['lr_hw']})
    return rows

def check(p):
    training_check(p)
    require(p['entrypoint_sha256']==file_sha256(Path(__file__)), 'Heavy runner changed')
    require(p['steps']==64 and p['evaluation_steps']==[0,64], 'Scope changed')
    for path,sha in p['sources'].items(): require(file_sha256(Path(path))==sha,'Source changed: '+path)
    compare_configs(load_config(REFERENCE/'resolved.yaml').model_dump(mode='json'),
                    load_config(p['arm']['config']).model_dump(mode='json'))

def declare(output):
    require(output.is_relative_to(ROOT/'logs') and not output.exists(), 'Use a fresh experiment directory')
    cfgdir=ROOT/'configs'/output.name;require(not cfgdir.exists(),'Config directory exists')
    source=load_config(REFERENCE/'resolved.yaml').model_dump(mode='json');cfg=heavy_config(source)
    cfgdir.mkdir();cfgpath=cfgdir/'heavy.yaml';cfgpath.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    config=load_config(cfgpath);compare_configs(source,config.model_dump(mode='json'))
    p=read(ROOT/'logs/scaled-training-20260908/protocol.json')
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),status='declared_before_training',
        authorization='User explicitly requested stronger degradation and continued experiments',
        changed_factor='resolution.ratio_range 0.18 -> 0.09 only; same starting state, LR and targets',
        entrypoint_sha256=file_sha256(Path(__file__)),reference_run=str(REFERENCE),
        arm={'name':'heavy_q009_lr1e5','lr':1e-5,'config':str(cfgpath),'config_sha256':file_sha256(cfgpath)},
        control='Medium input is a different difficulty; raw losses across severities are not comparable quality scores',
        old_initial_checkpoint=str(saved_at(REFERENCE,0)),inputs={},sources={},maximum_new_optimizer_updates=64)
    p['old_initial_checkpoint_sha256']=file_sha256(Path(p['old_initial_checkpoint']))
    base=ROOT/'runs/quality-calibration-20260908/encoded/heavy'
    for key,name in [('training','overfit_manifest.jsonl'),('degraded','degraded_manifest.jsonl'),('clean','clean_manifest.jsonl'),('full','training_manifest.jsonl')]:
        path=base/name;p['inputs'][key]={'path':str(path),'sha256':file_sha256(path)}
    for path in [ROOT/'runs/quality-calibration-20260908/encoded/report.json',ROOT/'runs/quality-calibration-20260908/preview-v2/report.json',
                 REFERENCE/'resolved.yaml',REFERENCE/'training_contract.json',
                 ROOT/'scripts/evaluate_bootstrap_checkpoint.py',ROOT/'scripts/render_resolution_probe.py']:
        p['sources'][str(path)]=file_sha256(path)
    with no_training_guard() as guard,TrainingBudget(ROOT/'runs',config.project.budget_seconds,phase='heavy_input_preflight') as budget:
        original=open_dataset(load_config(REFERENCE/'resolved.yaml'),ROOT,ROOT/'runs/quality-calibration-20260908/encoded/medium/overfit_manifest.jsonl')
        candidate=open_dataset(config,ROOT,base/'overfit_manifest.jsonl')
        audit=candidate.audit(budget_check=budget.check)
        cases=compare_views(original,candidate)
        preview=read(ROOT/'runs/quality-calibration-20260908/preview-v2/report.json')
        for case in preview['cases']:
            require(file_sha256(Path(case['preview']))==case['preview_sha256'],'Input preview changed')
        output.mkdir()
        atomic_write(output/'preflight.json',canonical_json({'status':'passed_actual_cached_data_audit','audit':audit,'cases':cases,
            'additional_optimizer_steps':0,'execution_guard':dict(guard),'visual_review':'All eight original medium/heavy ladder ROIs viewed before declaration; heavy eye/eyebrow/skin detail visibly more blurred.',
            'pixel_and_latent_preparation':'Reused existing audited heavy tier; no new encoding'}))
    p['sources'][str(output/'preflight.json')]=file_sha256(output/'preflight.json')
    atomic_write(output/'protocol.json',canonical_json(p));check(p)
    return p

def verify(output,step):
    p=read(output/'protocol.json');check(p);require(step in [0,64],'Undeclared boundary')
    run=Path(read(output/'progress.json')['run'])
    with no_training_guard() as guard:
        _,initial,cp0=read_committed(ROOT,run,saved_at(run,0))
        _,reference,_=read_committed(ROOT,REFERENCE,saved_at(REFERENCE,0))
        compare_configs(reference['resolved_config'],initial['resolved_config'])
        for key in ['model','optimizer','scheduler','sampler','rng','scaler']:assert_equal(reference[key],initial[key],'same initial '+key)
        expected=copy.deepcopy(reference['contract']);expected['resolved_sha256']=digest(initial['resolved_config'])
        expected['manifest_sha256']=p['inputs']['training']['sha256'];expected['compatibility']['data']=initial['resolved_config']['data']
        require(expected==initial['contract'],'Unexpected training contract difference')
        result={'status':'passed','step':step,'initial_model_sha256':state_hash(initial['model']),
            'initial_checkpoint_sha256':file_sha256(cp0),'scope':'CPU actual saved-state audit; no GPU forward',
            'equal_initial_fields':['model','optimizer','scheduler','sampler','rng','scaler']}
        if step:
            _,current,cp=read_committed(ROOT,run,saved_at(run,64))
            require(current['contract']==initial['contract'] and current['step']==64 and current['extra']['phase_complete'],'Final contract differs')
            result['optimizer']=check_optimizer(initial['optimizer'],current['optimizer'],current['model'],64)
            sampler=StatefulSampler(16,seed=42)
            for _ in range(64):
                sampler.begin_window()
                for _ in range(4):sampler.next_index()
                sampler.commit_window()
            assert_equal(sampler.state_dict(),current['sampler'],'final sampler')
            rows=[__import__('json').loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
            require([x['step'] for x in rows]==list(range(1,65)) and all(x['optimizer_updated'] and x['learning_rate']==1e-5 for x in rows),'Update log mismatch')
            require(read(run/'training_report.json')['vae_parameters_unchanged'],'H3 changed')
            result.update(checkpoint=str(cp),checkpoint_sha256=file_sha256(cp),sample_visits=256,per_view_visits=16,
                clipped_steps=[x['step'] for x in rows if x['gradient_norm_before_clip']>1],
                max_gradient_norm=max(x['gradient_norm_before_clip'] for x in rows))
        result.update(additional_optimizer_steps=0,execution_guard=dict(guard),trained_base_accepted=False)
        atomic_write(output/f'verified_step_{step:04d}.json',canonical_json(result));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['declare','train','verify0','verify64','evaluate'],required=True)
    a=parser.parse_args();out=a.output.resolve()
    if a.action=='declare':declare(out)
    elif a.action.startswith('verify'):print(canonical_json(verify(out,int(a.action[6:]))).decode())
    else:
        p=read(out/'protocol.json');check(p)
        if a.action=='train':train(p,out)
        else:verify(out,64);evaluate(p,read(out/'execution.json'),out)
