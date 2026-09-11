"""One fresh 64-update gradient-scaling comparison; never repeats existing training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from h3ce.cache.keys import canonical_json, file_sha256
from h3ce.cache.store import atomic_write
from h3ce.train.engine import implementation_hashes
from h3ce.train.precision import POLICY
from scripts.run_lr_length_probe import read, require, find_new_run, chart_for_run, checkpoint_at, validate_result
from scripts.run_clean_replay_probe import execute
from scripts.run_loss_balance_probe import cli_result


def declare(output):
    require(output.is_relative_to(ROOT/'logs') and not (output/'protocol.json').exists(), 'Use an undeclared log directory')
    before=read(output/'before_fix.json')
    require(file_sha256(output/'engine.before_loss_scaling.py')==before['engine_sha256'], 'Historical engine changed')
    cfg=ROOT/'configs/lr-length-20260908-v2/lr_1e5.yaml'
    base=ROOT/'runs/quality-calibration-20260908/encoded/medium'
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'declared_before_training',
       'authorization':'User requested continued experiments and tuning; controlled loss-scaling fix',
       'steps':64,'evaluation_steps':[0,64],'resume':'none','seed':42,
       'changed_factor':'FP16 decoder backward dynamic loss scaling; unscale before gradient clipping',
       'policy':POLICY,'implementation_sha256':implementation_hashes(ROOT),
       'runner_sha256':file_sha256(Path(__file__)),
       'arm':{'name':'scaled_1e5','lr':1e-5,'config':str(cfg),'config_sha256':file_sha256(cfg)},
       'control':'Preserved unscaled LR 1e-5 first 64 successful updates; no historical checkpoint migration',
       'control_summary':{'path':str(ROOT/before['old_baseline_summary']), 'sha256':file_sha256(ROOT/before['old_baseline_summary'])},
       'old_initial_checkpoint':str(ROOT/'runs/20260907T183732Z-train-3dfbe756/checkpoints/checkpoint-overfit-000000000000-4335c63b8a13.pt'),
       'independent_validation':False,'trained_base_accepted':False,'inputs':{}}
    p['old_initial_checkpoint_sha256']=file_sha256(Path(p['old_initial_checkpoint']))
    for key,name in [('training','overfit_manifest.jsonl'),('degraded','degraded_manifest.jsonl'),('clean','clean_manifest.jsonl'),('full','training_manifest.jsonl')]:
        path=base/name;p['inputs'][key]={'path':str(path),'sha256':file_sha256(path)}
    atomic_write(output/'protocol.json',canonical_json(p))


def check(p):
    require(p['implementation_sha256']==implementation_hashes(ROOT),'Training code changed after declaration')
    require(p['runner_sha256']==file_sha256(Path(__file__)),'Runner changed after declaration')
    for v in [*p['inputs'].values(),p['control_summary']]:
        require(file_sha256(Path(v['path']))==v['sha256'],'Evidence changed')
    require(file_sha256(Path(p['arm']['config']))==p['arm']['config_sha256'],'Config changed')
    require(file_sha256(Path(p['old_initial_checkpoint']))==p['old_initial_checkpoint_sha256'],'Old initial checkpoint changed')


def train(p,output):
    check(p)
    arm=p['arm'];log_path=output/'training.log'
    require(not log_path.exists(),'Existing partial or complete training must not be repeated')
    previous=set((ROOT/'runs').iterdir());current=None;errors=[]
    command=[sys.executable,'-u','-m','h3ce','train','--config',arm['config'],'--stage','bootstrap',
             '--phase','overfit','--manifest',p['inputs']['training']['path'],'--max-steps',str(p['steps']),'--resume','none']
    with log_path.open('x',encoding='utf-8') as log:
        log.write(json.dumps({'command':command})+'\n');log.flush()
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
            for line in child.stdout:
                log.write(line);log.flush()
                try:row=json.loads(line)
                except ValueError:continue
                if not isinstance(row,dict):continue
                if row.get('event')=='overflow_retry':print(json.dumps(row),flush=True)
                if row.get('event')!='optimizer_step':continue
                try:
                    if current is None:current=find_new_run(previous,arm)
                    atomic_write(output/'progress.json',canonical_json({'run':str(current),'limit':p['steps'],**row}))
                    if row['step']==1 or row['step']%8==0:chart_for_run(arm,current,output)
                except Exception as exc:
                    errors.append({'step':row['step'],'error':str(exc)})
                    atomic_write(output/'observer_errors.json',canonical_json(errors))
                if row['step']==1 or row['step']%8==0:print(json.dumps(row),flush=True)
            code=child.wait();log.write(json.dumps({'exit_code':code})+'\n')
    value=cli_result(log_path.read_text(encoding='utf-8'))
    result={'command':command,'exit_code':code,**value,'observer_errors':errors,'log_sha256':file_sha256(log_path)}
    atomic_write(output/'execution.json',canonical_json(result))
    validate_result(value,code,p['steps'])
    if current is None:current=find_new_run(previous,arm)
    require(Path(value['run'])==current,'Observed run differs from final report')
    chart_for_run(arm,current,output);check(p)
    return result


def evaluate(p,result,output):
    records=[];run=Path(result['run'])
    for step in p['evaluation_steps']:
        checkpoint=checkpoint_at(run,step)
        for kind in ['degraded','clean']:
            check(p);destination=run/f'step_{step:04d}_{kind}_evaluation'
            require(not destination.exists(),'Existing evaluation must be inspected, not overwritten')
            cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_bootstrap_checkpoint.py'),'--run',str(run),
                 '--checkpoint',str(checkpoint),'--manifest',p['inputs'][kind]['path'],'--output',str(destination)]
            if kind=='clean':cmd+=['--companion-source-manifest',p['inputs']['full']['path']]
            label=f'{step:04d}_{kind}'
            rec={'step':step,'kind':kind,**execute(cmd,output/(label+'.log'),label)}
            require(rec['exit_code']==0,'Evaluation failed; inspect retained log')
            path=destination/'metrics.json';data=read(path)
            require(data['additional_optimizer_steps']==0 and all(v==0 for v in data['execution_guard'].values())
                    and data['h3_parameters_frozen_and_unchanged'] and data['models']['current']['optimizer_steps_in_checkpoint']==step,
                    'Evaluation guard mismatch')
            rec.update(report=str(path),report_sha256=file_sha256(path));records.append(rec)
            atomic_write(output/'evaluations.json',canonical_json(records))
            print(json.dumps({'event':'fixed_step_completed','step':step,'kind':kind,'metrics':data['summaries']['current']['source_equal_mean']}),flush=True)
    atomic_write(output/'completed.json',canonical_json({'status':'64_updates_and_four_fixed_evaluations_completed',
                 'new_optimizer_steps':64,'trained_base_accepted':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['declare','run'],required=True)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='declare':declare(out)
    else:
        protocol=read(out/'protocol.json');result=train(protocol,out);evaluate(protocol,result,out)
