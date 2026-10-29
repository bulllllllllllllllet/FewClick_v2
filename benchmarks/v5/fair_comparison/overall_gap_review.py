"""Read-only, posthoc synthesis of accepted local counts and actual training logs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METHODS = ('sam_ft', 'wsi_sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell')
BUDGETS = ('1+1', '3+1', '3+2', '5+1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def key(row):
    return row['dataset'], row['query_id'], row['target']


def band(fraction):
    return '<5%' if fraction < .05 else '5-20%' if fraction < .2 else '20-50%' if fraction < .5 else '>=50%'


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def aggregate(rows):
    if not rows:
        return dict(pairs=0, rows=0)
    pairs = defaultdict(list)
    for row in rows:
        tp, fp, fn = (row[n] for n in ('tp', 'fp', 'fn'))
        assert tp+fn == row['positive'] and tp+fp <= row['valid']
        assert abs(row['dice']-2*tp/(2*tp+fp+fn)) < 1e-12
        pairs[key(row)].append(dict(dice=row['dice'], precision=tp/(tp+fp) if tp+fp else 0.,
            recall=tp/(tp+fn), fp_per_valid=fp/row['valid'], bf1=row['bf1_um2']))
    result = dict(pairs=len(pairs), rows=len(rows))
    for name in ('dice', 'precision', 'recall', 'fp_per_valid', 'bf1'):
        result[name] = mean(mean(r[name] for r in group if r[name] is not None)
                            for group in pairs.values() if any(r[name] is not None for r in group))
    tp, fp, fn = (sum(r[n] for r in rows) for n in ('tp', 'fp', 'fn'))
    result['micro_dice_diagnostic_only'] = 2*tp/(2*tp+fp+fn)
    return result


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = a.output/'statistics'
    output.mkdir(exist_ok=False)
    bindings = {}
    def bind(p):
        bindings[str(p)] = sha(p)
        return p
    prior = read(bind(a.prior/'summary.json'))
    rows = {}
    for method in METHODS:
        p = a.root/f'evaluate_{method}_1025_131451/rows.jsonl'
        bind(p)
        if str(p) in prior['bindings']:
            assert bindings[str(p)] == prior['bindings'][str(p)]
        rows[method] = lines(p)
        assert len(rows[method]) == 6040
    ref = {(key(r), r['prompt_group'], r['click_budget']): r for r in rows['sam_ft']}
    assert len(ref) == 6040
    for method, values in rows.items():
        index = {(key(r), r['prompt_group'], r['click_budget']): r for r in values}
        assert set(index) == set(ref) and len(index) == len(values)
        for k, row in index.items():
            for name in ('positive', 'valid', 'positive_xy', 'negative_xy', 'true_boundary_pixels'):
                assert row[name] == ref[k][name]
    methods = {}
    for method, values in rows.items():
        primary = [r for r in values if r['click_budget'] == '3+2']
        part = dict(primary=aggregate(primary), budgets={}, strata={}, development_history=[], optimization={})
        for budget in BUDGETS:
            current = [r for r in values if r['click_budget'] == budget]
            part['budgets'][budget] = dict(all=aggregate(current), by_area={})
            for label in ('<5%', '5-20%', '20-50%', '>=50%'):
                part['budgets'][budget]['by_area'][label] = aggregate([r for r in current if band(r['positive']/r['valid']) == label])
        for domain, target in sorted({(r['dataset'], r['target']) for r in primary}):
            current = [r for r in primary if (r['dataset'], r['target']) == (domain, target)]
            part['strata'][f'{domain}|{target}'] = dict(all=aggregate(current), by_area={
                label: aggregate([r for r in current if band(r['positive']/r['valid']) == label])
                for label in ('<5%', '5-20%', '20-50%', '>=50%')})
        run = a.root/f'train_{method}_1025_131451'
        part['registered_parameters'] = read(bind(run/'run.json'))['parameters']
        selected = read(bind(run/'best_checkpoint.json'))
        part['selected_checkpoint'] = selected
        for step in range(1000, 10001, 1000):
            history = read(bind(run/f'validation_{step:05d}_1025/rows.json'))
            assert len(history) == 302
            part['development_history'].append(dict(step=step, all=aggregate(history),
                small=aggregate([r for r in history if r['positive']/r['valid'] < .05])))
        progress = lines(bind(run/'progress.jsonl'))
        assert len(progress) == 10000
        norms = np.array([math.hypot(r['gradient_norms']['heads'], r['gradient_norms']['backbone']) for r in progress[1000:]])
        factors = np.minimum(1., 1./np.maximum(norms, 1e-12))
        part['optimization'] = dict(joint_steps=len(norms), fraction_norm_over_clip1=float((norms>1).mean()),
            global_grad_norm_median=float(np.median(norms)), global_clip_factor_median=float(np.median(factors)),
            clip_factor_description='Pre-clip gradient norm ratio; NOT Adam parameter-update ratio or a causal diagnosis.')
        methods[method] = part
    tasks = a.root.parent/'tasks_1025_125400'
    contract = read(bind(tasks/'contract.json'))
    catalog = Path(contract['catalog'])/'tiles.parquet'
    bind(catalog)
    assert bindings[str(catalog)] == contract['catalog_sha256']
    meta = {r['tile_id']:r for r in pd.read_parquet(catalog).to_dict('records')}
    def fraction(p):
        m = meta[p['query_id']]
        counts = {int(k):v for k,v in json.loads(m['counts_json']).items()}
        valid = sum(v for k,v in counts.items() if k != 255)
        positive = counts[p['target']]
        if 'valid' in p:
            assert valid == p['valid'] and positive == p['positive']
        assert m['split'] == p['split'] and m['dataset'] == p['dataset']
        return positive/valid
    training = lines(bind(a.root.parent/'realized_train_1025_125928/training_plans.jsonl'))
    assert len(training) == 40000
    validation = [r for r in rows['sam_ft'] if r['click_budget'] == '3+2' and r['prompt_group'] == 0]
    exposure = {}
    for name, values in [('actual_training', training), ('validation_pairs', validation)]:
        area = Counter(); strata = defaultdict(Counter); budgets = Counter()
        for p in values:
            label = band(fraction(p)); area[label] += 1
            strata[f"{p['dataset']}|{p['target']}"][label] += 1
            budgets[f"{len(p['positive_xy'])}+{len(p['negative_xy'])}"] += 1
        exposure[name] = dict(n=len(values), unique_fields=len({p['query_id'] for p in values}),
            unique_target_pairs=len({key(p) for p in values}), area_counts=area,
            area_fractions={k:v/len(values) for k,v in area.items()}, by_class=dict(strata), budgets=budgets)
    sam = methods['sam_ft']; few = methods['fewclick_m3d_xcell']
    area_decomposition = {}
    total_gap = sam['primary']['dice']-few['primary']['dice']
    for label in ('<5%', '5-20%', '20-50%', '>=50%'):
        sr = sam['budgets']['3+2']['by_area'][label]; fr = few['budgets']['3+2']['by_area'][label]
        contribution = sr['pairs']/302*(sr['dice']-fr['dice'])
        area_decomposition[label] = dict(pairs=sr['pairs'], gap=sr['dice']-fr['dice'],
            weighted_gap=contribution, share_of_total_gap=contribution/total_gap)
    assert abs(sum(x['weighted_gap'] for x in area_decomposition.values())-total_gap)<1e-12
    small_strata = []
    for label in sam['strata']:
        s = sam['strata'][label]['by_area']['<5%']; f = few['strata'][label]['by_area']['<5%']
        if s['pairs']:
            small_strata.append(dict(stratum=label, pairs=s['pairs'], gap=s['dice']-f['dice']))
    result = dict(status='complete_readonly_statistics_pending_independent_recount', worker_pid=os.getpid(),
        command=sys.argv, source_sha256=sha(__file__), methods=methods, exposure=exposure,
        area_decomposition=area_decomposition, within_class_small_gaps=small_strata, bindings=bindings,
        limitations=['Posthoc single-seed descriptive decomposition; no new causal, blind-test, or WSI claim.',
                    'Training areas derive from previously accepted catalog histograms, not a new GT image audit.',
                    'Validation group0 selects checkpoints; all five groups remain development data.',
                    'No representative training-set Dice exists; cannot diagnose overfitting from loss alone.'])
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(worker_pid=os.getpid(), output=str(output), exposure=exposure,
        area_decomposition=area_decomposition, optimization={m:v['optimization'] for m,v in methods.items()}),ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
