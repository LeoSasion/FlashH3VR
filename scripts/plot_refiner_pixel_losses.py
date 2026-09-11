"""Observe existing cold/warm R pixel records with the actual .5 detail contribution."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import threading
import numpy as np
from h3ce.cache.keys import file_sha256,canonical_json
from h3ce.cache.store import atomic_write
from scripts import plot_training_losses as engine
from scripts.gpt_clarity_probe_common import read
from h3ce.train.preflight import require

_LOCK=threading.RLock()


def contributions(rows,config):
    require(rows and [r['step'] for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=192,"Bounded contiguous R pixel rows required")
    weights={'rgb':1.,'latent':0.,'perceptual':0.,'lighting_target':.2,'detail':.5}
    require(all(config['training']['losses'][k]==weights[k] for k in engine.COMPONENTS),"Original application weights changed")
    require(all(r['phase']=='pixel' and r['diagnostic_kind']=='refiner_pixel_start_probe' for r in rows),"Wrong pixel experiment")
    values={key:np.asarray([r['losses'][key] for r in rows])*weight for key,weight in weights.items()}
    application=np.asarray([r['losses']['application_total'] for r in rows]);total=np.asarray([r['losses']['total'] for r in rows])
    require(all(np.isfinite(v).all() and (v>=0).all() for v in values.values()),"Nonfinite weighted loss")
    require(np.allclose(sum(values[k] for k in engine.COMPONENTS),application,rtol=2e-6,atol=1e-9)
            and np.allclose(application+values['detail'],total,rtol=2e-6,atol=1e-9),"Pixel detail contribution differs")
    return weights,values


def plot(runs,output,window=24):
    bindings=[]
    for run in runs:
        contract=read(run/'training_contract.json');request=contract['experiment']
        require(request['kind']=='refiner_pixel_start_probe' and request['pixel_losses']=={'rgb':1.,'lighting_target':.2,'detail':.5,'latent':0.,'perceptual':0.},"Wrong R pixel contract")
        bindings.append({'run':str(run),'contract_sha256':file_sha256(run/'training_contract.json'),'protocol_sha256':request['protocol_sha256']})
    with _LOCK:
        previous=engine.contributions
        try:
            engine.contributions=contributions
            result=engine.plot(runs,output,window)
        finally:engine.contributions=previous
    result['pixel_start_observer']={'bindings':bindings,'detail_weight':.5,'new_optimizer_updates':0}
    result['notes'].append('Application subtotal is not counted twice; raw latent coefficient is zero. Both R starts use the same original-only pixel supervision.')
    atomic_write(output.with_suffix('.json'),canonical_json(result))
    return result
