"""Predeclared 256-update frozen-H3 length probe under the verified scaled trainer."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
from h3ce.cache.keys import canonical_json,file_sha256
from h3ce.cache.store import atomic_write
from scripts.run_scaled_training_probe import read,require,check as training_check,train
from scripts.run_lr_length_probe import checkpoint_at
from scripts.run_clean_replay_probe import execute


def bound(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':file_sha256(path)}


def declare(output):
    require(output.is_relative_to(ROOT/'logs') and not output.exists(),'Use a new experiment directory')
    previous=ROOT/'logs/scaled-training-20260908'
    p=copy.deepcopy(read(previous/'protocol.json'));training_check(p)
    verification=read(previous/'verification.json')
    require(verification['status']=='passed' and verification['new_optimizer_steps']==64,'Prior scaled probe was not verified')
    for path,sha in verification['files'].items():
        require(file_sha256(Path(path))==sha,'Prior evidence changed')
    require(read(previous/'execution.json')['status']=='passed_overfit_probe','Scaled numeric probe must pass first')
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),steps=256,evaluation_steps=[0,64,128,256],
        authorization='User active goal: continued experiments and tuning; test training length after verified scaling fix',
        changed_factor='Successful optimizer-update count only; identical training implementation, configuration and data',
        control='Prior scaled 64-update run; verify equal initial state and inspect first-64 reproduction',
        reference_scaled_run=read(previous/'execution.json')['run'],
        reference_scaled_summary=bound(previous/'comparison/summary.json'),
        reference_scaled_verification=bound(previous/'verification.json'),
        entrypoint_sha256=file_sha256(Path(__file__)),
        maximum_new_optimizer_updates=256,automatic_training_beyond_limit=False,
        initialization_reason='Fresh same-seed start avoids modifying the completed 64-step checkpoint contract',
        helper_sources={})
    p['arm']['name']='scaled_length_1e5'
    for name in ['run_scaled_training_probe.py','run_lr_length_probe.py','run_clean_replay_probe.py',
                 'run_loss_balance_probe.py','plot_training_losses.py','evaluate_bootstrap_checkpoint.py']:
        path=ROOT/'scripts'/name;p['helper_sources'][str(path)]=file_sha256(path)
    output.mkdir(parents=True)
    atomic_write(output/'protocol.json',canonical_json(p))
    return p


def check(p):
    training_check(p)  # runner_sha256 binds the reused training runner, retained unchanged.
    require(file_sha256(Path(__file__))==p['entrypoint_sha256'],'Length entrypoint changed')
    require(p['steps']==p['maximum_new_optimizer_updates']==256 and p['evaluation_steps']==[0,64,128,256], 'Declared bounds differ')
    for item in [p['reference_scaled_summary'],p['reference_scaled_verification']]:
        require(file_sha256(Path(item['path']))==item['sha256'],'Scaled control changed')
    for path,sha in p['helper_sources'].items():
        require(file_sha256(Path(path))==sha,'Observer or evaluator implementation changed')


def evaluate(p,result,output):
    check(p);run=Path(result['run']);records=[]
    require(result['optimizer_steps']==256,'The declared training has not completed')
    for step in p['evaluation_steps']:
        checkpoint=checkpoint_at(run,step)
        for kind in ['degraded','clean']:
            check(p);destination=run/f'step_{step:04d}_{kind}_evaluation'
            require(not destination.exists(),'Existing partial or complete evaluation requires inspection')
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_bootstrap_checkpoint.py'),
                '--run',str(run),'--checkpoint',str(checkpoint),'--manifest',p['inputs'][kind]['path'],
                '--output',str(destination)]
            if kind=='clean':command+=['--companion-source-manifest',p['inputs']['full']['path']]
            label=f'{step:04d}_{kind}'
            record={'step':step,'kind':kind,**execute(command,output/(label+'.log'),label)}
            require(record['exit_code']==0,'Evaluation failed; preserve its log')
            path=destination/'metrics.json';data=read(path)
            require(data['additional_optimizer_steps']==0 and all(v==0 for v in data['execution_guard'].values())
                and data['h3_parameters_frozen_and_unchanged'] and data['models']['current']['optimizer_steps_in_checkpoint']==step,
                'Evaluation mutation or checkpoint mismatch')
            record.update(report=str(path),report_sha256=file_sha256(path));records.append(record)
            atomic_write(output/'evaluations.json',canonical_json(records))
            print(canonical_json({'event':'fixed_step_completed','step':step,'kind':kind,
                'metrics':data['summaries']['current']['source_equal_mean']}).decode(),flush=True)
    atomic_write(output/'completed.json',canonical_json({'status':'256_updates_and_eight_fixed_evaluations_completed',
        'new_optimizer_steps':256,'trained_base_accepted':False,'independent_validation':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['declare','run','evaluate'],required=True)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='declare':declare(out)
    else:
        protocol=read(out/'protocol.json');check(protocol)
        result=train(protocol,out) if args.action=='run' else read(out/'execution.json')
        evaluate(protocol,result,out)
