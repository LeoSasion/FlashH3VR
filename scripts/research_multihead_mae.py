"""Shared explicit MAE objective for the new frozen-condition research path."""
import numpy as np
import torch
from h3ce.data.continuous_head import paste_delta
from h3ce.train.head_video_mae_loss import target_transition_mae
from scripts.probe_narrator_transition_direction import load_case
from scripts.probe_multihead_video_direction import average, gates_for
from scripts.train_head_condition_int8 import objective
from scripts.research_head_tail2 import metrics


def temporal(delta, data, case):
    x, y, support, plan, _ = data
    pred = torch.cat([paste_delta(x[i:i+1], delta[i:i+1], g) for i, g in enumerate(plan['transforms'])])
    loss = target_transition_mae(pred, y, support, case['pts'], [0]*22)['total']
    return loss * (2. if case['clean'] else 1.), pred


def calculate(delta, data, case, metric):
    tm, pred = temporal(delta, data, case)
    terms=[]
    for i, sample in enumerate(data[4]):
        a,b,c,d = sample['geometry']['paste_xyxy']
        terms.append(objective(pred[i:i+1,:,None,b:d,a:c],sample,metric))
    spatial = torch.stack([t['total'] for t in terms]).mean()
    def avg(k): return torch.stack([t[k]*t['sample_weight'] for t in terms]).mean()
    effective = dict(rgb=avg('rgb'), lighting=.2*avg('lighting'), detail=.5*avg('detail'), lpips=.05*avg('perceptual'), motion=.1*tm)
    return spatial+.1*tm, spatial, tm, effective, pred


def evaluate(delta, data, case, metric):
    total, spatial, tm, _, pred = calculate(delta,data,case,metric)
    values=[]; originals=[]
    for i, sample in enumerate(data[4]):
        a,b,c,d=sample['geometry']['paste_xyxy']
        values.append(metrics(pred[i:i+1,:,None,b:d,a:c],sample)); originals.append(metrics(sample['x'],sample))
    row={k:case[k] for k in ('case_id','dataset','clip_start','kind','clean','purpose')}
    row.update(spatial_loss=float(spatial.detach()),transition_loss=float(tm.detach()),joint_loss=float(total.detach()),transition_function='absolute',
        metrics={k:float(np.mean([v[k] for v in values])) for k in values[0]},
        input_metrics={k:float(np.mean([v[k] for v in originals])) for k in originals[0]})
    return spatial,tm,row
