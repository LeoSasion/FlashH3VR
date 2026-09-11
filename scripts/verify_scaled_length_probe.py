"""Read-only CPU audit of an initial or saved boundary in the scaled length probe."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
from h3ce.cache.keys import canonical_json,file_sha256,digest
from h3ce.cache.store import atomic_write
from h3ce.train.guard import no_training_guard
from h3ce.train.sampler import StatefulSampler
from scripts.run_scaled_length_probe import read,require,check
from scripts.evaluate_bootstrap_checkpoint import read_committed
from scripts.verify_loss_balance_initialization import assert_equal,check_optimizer,state_hash
from scripts.plot_training_losses import read_records,contributions


def saved_at(run,step):
    receipts=list((run/'checkpoints').glob(f'checkpoint-overfit-{step:012d}-*.json'))
    require(receipts,f'No saved boundary at step {step}; a live training log is not a checkpoint')
    choose=min if step==0 else max
    return choose(receipts,key=lambda path:read(path)['created_ns']).with_suffix('.pt')


def verify(experiment,through_step):
    p=read(experiment/'protocol.json');check(p)
    run=Path(read(experiment/'progress.json')['run'])
    control=read(p['reference_scaled_summary']['path'])
    with no_training_guard() as guard:
        initial_cp=saved_at(run,0)
        cfg,initial,_=read_committed(ROOT,run,initial_cp)
        reference_cp=Path(control['initial']['checkpoint'])
        _,reference,_=read_committed(ROOT,reference_cp.parent.parent,reference_cp)
        equal_fields=['model','optimizer','scheduler','scaler','sampler','rng','resolved_config']
        for key in equal_fields:assert_equal(reference[key],initial[key],'same scaled initial '+key)
        expected=dict(reference['contract']);expected['max_steps']=256
        require(initial['contract']==expected,'More than the declared max_steps changed')
        require(initial['step']==0 and initial['resolved_config']==cfg.model_dump(mode='json'),'Initial config differs')
        result={'status':'passed','run':str(run),'through_step':through_step,
            'scope':'CPU saved-state inspection; no additional GPU forward or optimizer update',
            'initial_checkpoint':str(initial_cp),'initial_sha256':file_sha256(initial_cp),
            'initial_model_sha256':state_hash(initial['model']),'equal_initial_fields':equal_fields,
            'reference_initial_checkpoint':str(reference_cp),'only_changed_contract_field':'max_steps',
            'new_optimizer_steps_in_verifier':0,'trained_base_accepted':False}
        if through_step:
            cp=saved_at(run,through_step);_,current,_=read_committed(ROOT,run,cp)
            require(current['step']==through_step and current['contract']==initial['contract'],'Saved boundary contract differs')
            result['optimizer']=check_optimizer(initial['optimizer'],current['optimizer'],current['model'],through_step)
            sampler=StatefulSampler(16,seed=42);assert_equal(initial['sampler'],sampler.state_dict(),'fresh sampler')
            indices=[]
            for _ in range(through_step):
                sampler.begin_window();indices.extend(sampler.next_index() for _ in range(4));sampler.commit_window()
            assert_equal(sampler.state_dict(),current['sampler'],'saved sample endpoint')
            require(Counter(indices)=={i:through_step//4 for i in range(16)},'Sample visits differ')
            rows,_=read_records(run/'metrics.jsonl',live=True);rows=[r for r in rows if r['step']<=through_step]
            require([r['step'] for r in rows]==list(range(1,through_step+1)),'Metrics have gaps or duplicate updates')
            require(all(r['optimizer_updated'] and r['learning_rate']==1e-5 for r in rows),'Optimizer policy changed')
            weights,_=contributions(rows,cfg.model_dump(mode='json'))
            require(current['scheduler']['last_epoch']==through_step and current['scaler']['scale']==rows[-1]['loss_scale_after'],'Schedule/scaler mismatch')
            result.update(checkpoint=str(cp),checkpoint_sha256=file_sha256(cp),effective_weights=weights,
                sample_visits=len(indices),per_view_visits=through_step//4,sampler_trace_sha256=digest(indices),
                sampler_scope='Deterministic reconstruction checked against saved endpoints, not a per-microbatch recording')
            if through_step>=64:
                new_cp=saved_at(run,64);_,new64,_=read_committed(ROOT,run,new_cp)
                old_cp=Path(control['final_checkpoint']);_,old64,_=read_committed(ROOT,old_cp.parent.parent,old_cp)
                prefix_fields=['model','optimizer','scheduler','scaler','sampler','rng']
                equality={k:state_hash(old64[k])==state_hash(new64[k]) for k in prefix_fields}
                old_rows,_=read_records(old_cp.parent.parent/'metrics.jsonl')
                def numeric(row):return {k:v for k,v in row.items() if k!='used_seconds'}
                log_equal=[numeric(r) for r in rows[:64]]==[numeric(r) for r in old_rows]
                result['first64_reproduction']={'exact_state_equality':equality,'all_numeric_metric_records_equal':log_equal,
                    'reference_checkpoint_sha256':file_sha256(old_cp),'candidate_checkpoint_sha256':file_sha256(new_cp),
                    'status':'exact_match' if all(equality.values()) and log_equal else 'numerical_difference_requires_review',
                    'excluded':'Elapsed budget; final-only after-probe and phase_complete evidence'}
                delta2=sum(float((new64['model'][k].double()-v.double()).square().sum()) for k,v in old64['model'].items())
                norm2=sum(float(v.double().square().sum()) for v in old64['model'].values())
                result['first64_reproduction'].update(parameter_delta_l2=math.sqrt(delta2),
                    relative_parameter_delta_l2=math.sqrt(delta2/norm2),
                    max_absolute_logged_total_difference=max(abs(a['losses']['total']-b['losses']['total']) for a,b in zip(old_rows,rows[:64])),
                    deterministic_algorithms=initial['contract']['compatibility']['runtime']['deterministic_algorithms'])
            if through_step==256:
                report=read(run/'training_report.json')
                require(current['extra']['phase_complete'] and report['optimizer_steps']==256 and report['vae_parameters_unchanged'], 'Final training report missing or incomplete')
        result['execution_guard']=dict(guard)
        require(all(v==0 for v in guard.values()),'Read-only guard triggered')
        atomic_write(experiment/f'verified_step_{through_step:04d}.json',canonical_json(result))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--through-step',type=int,choices=[0,64,128,256],required=True)
    args=parser.parse_args();result=verify(args.experiment.resolve(),args.through_step)
    print(canonical_json({'status':result['status'],'through_step':result['through_step'],
        'reproduction':result.get('first64_reproduction')}).decode(),flush=True)
