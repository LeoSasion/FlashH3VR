"""Finite multihead gradient preflight with exact frozen Narrator cache reuse."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from h3ce.components import sha256_file
from h3ce.config import load_config
from h3ce.train.checkpoint import CheckpointManager, TrainingBudget
from h3ce.train.perceptual import load_perceptual
from scripts.research_nafnet_gopro32 import load_model
from scripts.research_naf_head3 import NAFHead3
from scripts.research_head_tail2 import strict_spatial, metrics
from scripts.probe_narrator_transition_direction import load_case
from scripts.train_narrator_transition32 import calculate
from scripts import benchmark_swinir_batch8_strict_video as timing

OLD = ROOT / 'logs/narrator-transition-direction-20260910-v1'
TANGO = ROOT / 'runs/tango-video-data-20260910-v1'
PARENT = ROOT / 'runs/head-condition-int8-20260910-v1'
OUT = ROOT / 'logs/multihead-video-direction-20260910-v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def check(bindings):
    for f, h in bindings.items(): assert sha256_file(f) == h, f


def evaluate(delta, data, case, metric):
    total, spatial, temporal, effective, pred = calculate(delta, data, case, metric)
    values, originals = [], []
    for i, sample in enumerate(data[4]):
        a, b, c, d = sample['geometry']['paste_xyxy']
        values.append(metrics(pred[i:i+1, :, None, b:d, a:c], sample))
        originals.append(metrics(sample['x'], sample))
    row = {k: case[k] for k in ('case_id', 'dataset', 'clip_start', 'kind', 'clean', 'purpose')}
    row.update(spatial_loss=float(spatial.detach()), transition_loss=float(temporal.detach()), joint_loss=float(total.detach()),
        metrics={k: float(np.mean([v[k] for v in values])) for k in values[0]},
        input_metrics={k: float(np.mean([v[k] for v in originals])) for k in originals[0]})
    return spatial, temporal, row


def average(rows, key, dataset=None):
    return float(np.mean([r[key] for r in rows if r['purpose'] == 'fit' and (dataset is None or r['dataset'] == dataset)]))


def gates_for(initial, proposed, dp, dm):
    return dict(spatial_direction=bool(dp > 0), temporal_direction=bool(dm > 0),
        fit_spatial_decrease=average(proposed, 'spatial_loss') < average(initial, 'spatial_loss'),
        fit_temporal_decrease=average(proposed, 'transition_loss') < average(initial, 'transition_loss'),
        narrator_joint_retention=average(proposed, 'joint_loss', 'narrator') <= average(initial, 'joint_loss', 'narrator'),
        tango_joint_retention=average(proposed, 'joint_loss', 'tango') <= average(initial, 'joint_loss', 'tango'),
        clean_mean=bool(np.mean([r['metrics']['rgb_mae'] for r in proposed if r['clean']]) <= .001),
        every_clean=all(not r['clean'] or r['metrics']['rgb_mae'] <= .001 for r in proposed),
        every_degraded_rgb=all(r['clean'] or r['metrics']['rgb_mae'] <= 1.05*r['input_metrics']['rgb_mae'] for r in proposed))


def main():
    assert sys.argv[1:] == ['--run'] and not OUT.exists()
    old = read(OLD / 'protocol.json'); old_summary = read(OLD / 'summary.json'); old_review = read(OLD / 'cpu_review/review.json')
    assert old_summary['passed'] and old_review['passed']
    check(old['bindings']); check(old_summary['artifacts']); check(old_review['bindings'])
    tango = read(TANGO / 'data_protocol.json'); tango_review = ROOT / 'logs/tango-video-data-20260910-v1/review.json'
    assert read(tango_review)['condition_frames'] == 308
    check(read(tango_review)['bindings']); check(read(TANGO / 'protocol.json')['bindings'])
    assert read(TANGO / 'h3_contract.json')['native'] == read(ROOT / 'runs/narrator-video-data-20260910-v2/h3_contract.json')['native']
    cases = [dict(c, dataset='narrator', case_id=f'narrator_{c["clip_start"]:03d}_{c["kind"]}') for c in old['cases']]
    cases += [dict(c, dataset='tango') for c in tango['cases']]
    assert len(cases) == 20 and sum(c['purpose'] == 'fit' for c in cases) == 14
    for c in cases: check(c['artifacts'])
    assert sha256_file(old['checkpoint']['path']) == old['checkpoint']['sha256']
    OUT.mkdir(); torch.set_num_threads(4); timing.OUT = OUT
    files = [Path(__file__), OLD/'protocol.json', OLD/'summary.json', OLD/'cpu_review/review.json', tango_review,
        TANGO/'data_protocol.json', TANGO/'h3_contract.json', ROOT/'logs/tango-head-round-20260910-v1/completion.json',
        ROOT/'docs/MULTIHEAD_VIDEO_DIRECTION_PROTOCOL_20260910.md', ROOT/'scripts/train_narrator_transition32.py',
        ROOT/'scripts/probe_narrator_transition_direction.py', ROOT/'scripts/research_naf_head3.py',
        ROOT/'scripts/research_head_tail2.py', ROOT/'scripts/train_head_condition_int8.py', ROOT/'h3ce/train/head_video_loss.py']
    protocol = dict(authorization='Active research, new finite joint-data direction preflight, zero updates', cases=cases,
        checkpoint=old['checkpoint'], source_groups=1, head_side=256, fit_cases=14, check_cases=6,
        loss='Unchanged RGB1 light.2 detail.5 LPIPS.05 temporal.1 clean2; equal14 fit-track weights',
        counts_max=dict(backbone=14, reference=14, tail=44, lpips=968, autograd_grad=20, h3=0, updates=0),
        reused_narrator=dict(features=6, initial_outputs=6, initial_frames=132, fit_gradient_cases=4, prior_vjp_calls=8),
        cpu_constructor_shape_forwards=1, bindings={str(f): sha256_file(f) for f in files})
    save(OUT/'protocol.json', protocol); counts = {k: 0 for k in protocol['counts_max']}
    def inc(k):
        counts[k] += 1
        assert counts[k] <= protocol['counts_max'][k], counts
    with TrainingBudget(ROOT/'runs', 172800, phase='multihead_video_direction') as budget:
        timing.wait_idle('before_load')
        base, info = load_model(); model = NAFHead3(base).cuda().eval()
        metric = load_perceptual(load_config(ROOT/'configs/project.int8.yaml'), ROOT, force=True)
        state = CheckpointManager(ROOT/'runs', PARENT, contract=read(PARENT/'training_contract.json')).read(Path(old['checkpoint']['path']))
        assert state['step'] == 64
        model.tail.load_state_dict(state['model'], strict=True); del state
        params = dict(model.tail.named_parameters()); assert sum(v.numel() for v in params.values()) == 199747
        before = {k: v.detach().clone() for k, v in params.items()}
        gradients_old = torch.load(OLD/'gradient_and_hypothesis.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(v.cpu(), gradients_old['initial'][k]) for k, v in before.items())
        versions = [{n: (id(v), v._version) for n, v in m.named_parameters()} for m in (base, model.reference, metric)]
        for key, module in [('backbone', base), ('reference', model.reference), ('tail', model.tail), ('lpips', metric)]:
            module.register_forward_pre_hook(lambda m, a, key=key: inc(key))
        cache, negative, data, manifest, initial = [], [], [], [], []
        old_initial = read(OLD/'initial.json')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), strict_spatial():
            for i, case in enumerate(cases):
                data.append(load_case(case, 'cuda'))
                if i < 6:
                    feature_path = OLD/f'features_{i}.pt'; reference_path = OLD/f'reference_{i}.npy'
                    features = {k: v.cuda() for k, v in torch.load(feature_path, map_location='cpu', weights_only=True).items()}
                    ref = torch.from_numpy(np.load(reference_path)).cuda()
                    row = dict(old_initial[i], dataset=case['dataset'], case_id=case['case_id'])
                    delta_path = OLD/f'initial_delta_{i}.npy'
                    provenance = 'exact_audited_narrator_features_reference_initial'
                else:
                    condition = torch.from_numpy(np.load(Path(case['directory'])/'condition.npy')).cuda()
                    c = model.features(condition)
                    ref = model.reference(c['features'], c['skip_mid'], c['skip_full'])
                    assert torch.equal(ref, c['official_ending'])
                    features = {k: c[k] for k in ('features', 'skip_mid', 'skip_full')}
                    feature_path = OUT/f'features_{i}.pt'; reference_path = OUT/f'reference_{i}.npy'
                    torch.save({k: v.cpu() for k, v in features.items()}, feature_path)
                    np.save(reference_path, ref.cpu().numpy())
                    delta = model.tail(**features)-ref
                    sp, tm, row = evaluate(delta, data[i], case, metric)
                    delta_path = OUT/f'initial_delta_{i}.npy'; np.save(delta_path, delta.cpu().numpy())
                    provenance = 'new_frozen_input_features_and_initial'
                    del c, condition, delta, sp, tm
                cache.append(features); negative.append(ref); initial.append(row)
                manifest.append(dict(features=str(feature_path), reference=str(reference_path), initial_delta=str(delta_path), provenance=provenance,
                    artifacts={str(f): sha256_file(f) for f in (feature_path, reference_path, delta_path)}))
                print('Prepared', case['case_id'], provenance, flush=True); budget.check()
        save(OUT/'features_manifest.json', manifest); save(OUT/'initial.json', initial)
        gp_tango = {k: torch.zeros_like(v) for k, v in params.items()}; gm_tango = {k: torch.zeros_like(v) for k, v in params.items()}
        with strict_spatial():
            for i, case in enumerate(cases):
                if case['dataset'] != 'tango' or case['purpose'] != 'fit': continue
                delta = model.tail(**cache[i])-negative[i]
                sp, tm, _ = evaluate(delta, data[i], case, metric)
                ap = torch.autograd.grad(sp, tuple(params.values()), retain_graph=True); inc('autograd_grad')
                am = torch.autograd.grad(tm, tuple(params.values())); inc('autograd_grad')
                for (k, _), a, b in zip(params.items(), ap, am):
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    gp_tango[k] += a.detach()/10; gm_tango[k] += b.detach()/10
                del delta, sp, tm, ap, am
                print('Gradient', case['case_id'], flush=True); budget.check()
        gp_narrator = {k: v.cuda() for k, v in gradients_old['spatial'].items()}
        gm_narrator = {k: v.cuda() for k, v in gradients_old['temporal'].items()}
        gp = {k: gp_narrator[k]*(4/14)+gp_tango[k]*(10/14) for k in params}
        gm = {k: gm_narrator[k]*(4/14)+gm_tango[k]*(10/14) for k in params}
        joint = {k: gp[k]+.1*gm[k] for k in params}
        norm = torch.sqrt(sum(v.square().sum() for v in joint.values())); assert torch.isfinite(norm) and norm > 0
        clip = min(1., 1/(float(norm)+1e-6))
        hypothesis = {k: before[k]-1e-5*(joint[k]*clip)/((joint[k]*clip).abs()+1e-8) for k in params}
        tensors = dict(initial=before, hypothesis=hypothesis, spatial=gp, temporal=gm,
            narrator_spatial=gp_narrator, narrator_temporal=gm_narrator, tango_spatial=gp_tango, tango_temporal=gm_tango)
        torch.save({label: {k: v.cpu() for k, v in values.items()} for label, values in tensors.items()}, OUT/'gradient_and_hypothesis.pt')
        proposed = []
        with torch.no_grad(), strict_spatial():
            for i, case in enumerate(cases):
                delta = torch.func.functional_call(model.tail, hypothesis, (), cache[i])-negative[i]
                sp, tm, row = evaluate(delta, data[i], case, metric); proposed.append(row)
                np.save(OUT/f'hypothesis_delta_{i}.npy', delta.cpu().numpy())
                del delta, sp, tm
                print('Hypothesis', case['case_id'], flush=True); budget.check()
        save(OUT/'hypothesis.json', proposed)
        dp = float(sum((gp[k]*joint[k]).sum() for k in params)); dm = float(sum((gm[k]*joint[k]).sum() for k in params))
        gates = gates_for(initial, proposed, dp, dm)
        assert all(torch.equal(v, before[k]) and v.grad is None for k, v in params.items())
        assert versions == [{n: (id(v), v._version) for n, v in m.named_parameters()} for m in (base, model.reference, metric)]
        assert counts == protocol['counts_max'], counts
        files = [f for f in OUT.iterdir() if f.is_file()]
        save(OUT/'summary.json', dict(status='completed_zero_update_multihead_direction', counts=counts, gates=gates, passed=all(gates.values()),
            spatial_dot_joint=dp, temporal_dot_joint=dm, joint_gradient_norm=float(norm),
            initial_fit={k: average(initial, k) for k in ('spatial_loss', 'transition_loss', 'joint_loss')},
            hypothesis_fit={k: average(proposed, k) for k in ('spatial_loss', 'transition_loss', 'joint_loss')},
            per_dataset={ds: {stage: {k: average(rows, k, ds) for k in ('spatial_loss', 'transition_loss', 'joint_loss')}
                for stage, rows in [('initial', initial), ('hypothesis', proposed)]} for ds in ('narrator', 'tango')},
            all_weights_unchanged=True, optimizer_constructions=0, updates=0, h3_calls=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), accepted_base=False, budget=budget.snapshot(),
            artifacts={str(f): sha256_file(f) for f in files}))
        print(json.dumps(dict(gates=gates, counts=counts, updates=0)), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as e:
        if OUT.exists(): save(OUT/'failure.json', dict(error=repr(e), automatic_restart=False, updates=0))
        raise
