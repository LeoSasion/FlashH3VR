"""Adapt actual fixed-video records to the existing training-loss observer.

Historical plot_training_losses.py is hash-bound by older experiments. Keep its
file unchanged; use its plot/smoothing/hash writer with explicit schema adapters.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from matplotlib.figure import Figure
from scripts import plot_training_losses as observer
from scripts.research_multihead_balanced_mae import WEIGHT

RUN=ROOT/'runs/multihead-balanced-mae32-20260910-v1'
OUT=ROOT/'logs/multihead-balanced-mae32-20260910-v1/loss_observer.png'

def records(path,*,live=False):
    assert not live
    raw=path.read_bytes();source=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [r['step'] for r in source]==list(range(1,len(source)+1))
    rows=[]
    for r in source:
        assert r['transition_function']=='absolute' and r['temporal_weight']==WEIGHT
        assert r['optimizer_updated'] and r['fit_clips']==14 and r['real_frames_per_update']==308
        assert np.isfinite(r['total']) and abs(r['total']-sum(r['effective'].values()))<=2e-7
        rows.append({'step':r['step'],'phase':'pixel','losses':{'total':r['total'],'latent':0},'effective':r['effective']})
    # Return the original bytes, so the observer's receipt binds the real log.
    return rows,raw

def contributions(rows,config):
    weights={k:1. for k in rows[0]['effective']};weights['latent']=0.
    values={k:np.array([r['effective'][k] for r in rows]) for k in rows[0]['effective']}
    values['latent']=np.zeros(len(rows))
    assert np.allclose(sum(values.values()),[r['losses']['total'] for r in rows],rtol=2e-6,atol=1e-9)
    return weights,values

def main():
    fixed=[]
    for path in sorted(RUN.glob('fixed_step_*/report.json')):
        report=json.loads(path.read_text());rows=[r for r in report['rows'] if r['purpose']=='fit' and not r['clean']]
        fixed.append({'step':report['step'],'degraded_fit_transition_loss':float(np.mean([r['transition_loss'] for r in rows]))})
    old_read,old_contributions,old_save=observer.read_records,observer.contributions,Figure.savefig
    def savefig(fig,*args,**kwargs):
        fig.suptitle('H3CE gradient-balanced L1 training: fixed14 tracks per update, native INT8 H3 conditions')
        fig.axes[0].set_title('Actual pre-update joint loss')
        fig.axes[1].set_title('Effective terms, including clean sample weight')
        for text,label in zip(fig.axes[1].get_legend().get_texts(),['RGB x1','Lighting x0.2','Detail x0.5','LPIPS x0.05','Motion L1 x1.57991']):text.set_text(label)
        ax=fig.axes[2];ax.clear();ax.plot([r['step'] for r in fixed],[r['degraded_fit_transition_loss'] for r in fixed],'o-',label='Measured degraded-fit motion')
        ax.set(title='Fixed evaluations only; latent objective disabled',xlabel='Optimizer step',ylabel='Target-transition absolute MAE');ax.grid(alpha=.18);ax.legend(fontsize=8)
        fig.set_size_inches(17,5);fig.tight_layout(pad=2.5)
        return old_save(fig,*args,**kwargs)
    try:
        observer.read_records=records;observer.contributions=contributions;Figure.savefig=savefig
        result=observer.plot([RUN],OUT,window=4)
    finally:observer.read_records=old_read;observer.contributions=old_contributions;Figure.savefig=old_save
    result['notes']=['Actual fixed fourteen-track pre-update losses; no shuffled-view claim.','Effective input terms already include coefficients and clean weighting, so observer aggregation weights are1.','Disabled latent is not a measured loss; third panel replaced by actual fixed video evaluations.','Original source metrics bytes are hashed; no training started or resumed.']
    result['effective_coefficients']={'rgb':1.,'lighting':.2,'detail':.5,'lpips':.05,'motion':WEIGHT,'clean_multiplier':2.}
    result['temporal_function']='absolute'
    result['fixed_evaluations']=fixed
    result['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['base_observer_sha256']=hashlib.sha256(Path(observer.__file__).read_bytes()).hexdigest()
    OUT.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
