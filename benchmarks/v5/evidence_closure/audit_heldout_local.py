"""Independently recount completed heldout local training and terminal scores."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.fair_comparison.metrics import VERSION, boundary
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.heldout_comparison.audit import FOLDS


def read(p):
    return json.loads(Path(p).read_text())


@lru_cache(maxsize=None)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), message)


def key(r):
    return r['pair_id'], r['prompt_group'], r['click_budget']


def checker(tasks):
    contract = read(tasks/'contract.json')
    catalog = Path(contract['catalog'])/'tiles.parquet'
    require(sha(str(catalog)) == contract['catalog_sha256'], 'catalog SHA')
    fields = {r['tile_id']: r for r in pd.read_parquet(catalog).to_dict('records')}
    plans = read(tasks/'validation_maximum.json')
    require(sha(str(tasks/'validation_maximum.json')) == contract['validation_sha256'], 'validation SHA')
    expected = {key(p): p for m in plans for p in [nested(m, b) for b in contract['report_budgets']]}
    denominators = {}
    for tile, target in {(p['query_id'], p['target']) for p in plans}:
        field = fields[tile]
        gt = np.asarray(Image.open(field['labels']))
        require(hashlib.sha256(gt.tobytes()).hexdigest() == field['labels_array_sha256'], 'actual label SHA')
        valid = gt != 255
        truth = gt == target
        denominators[tile, target] = dict(positive=int((truth & valid).sum()), valid=int(valid.sum()),
            true_boundary_pixels=int(boundary(truth, valid).sum()))

    def validate(row):
        k = key(row)
        require(k in expected, 'unexpected task')
        for f, v in expected[k].items():
            require(row[f] == v, 'task/point mismatch: '+f)
        for f, v in denominators[row['query_id'], row['target']].items():
            require(row[f] == v, 'GT denominator: '+f)
        tp, fp, fn = [row[f] for f in ('tp', 'fp', 'fn')]
        require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)), 'integer counts')
        require(tp+fn == row['positive'] and fp <= row['valid']-row['positive'], 'count constraints')
        close(row['dice'], 2*tp/(2*tp+fp+fn), 'Dice replay')
        close(row['false_positive_fraction'], fp/row['valid'], 'FP fraction replay')
        require(row['boundary_evaluable'] == (row['true_boundary_pixels'] > 0), 'boundary eligibility')
        if row['true_boundary_pixels'] == 0:
            require(row['bf1_um2'] is None, 'undefined BF1')
        else:
            require(row['bf1_um2'] is not None and 0 <= row['bf1_um2'] <= 1, 'BF1 range')
            if row['predicted_boundary_pixels'] == 0:
                close(row['bf1_um2'], 0., 'missing predicted boundary')
        return k
    return contract, expected, validate


def means(rows):
    out = {}
    for budget in sorted({r['click_budget'] for r in rows}):
        selected = [r for r in rows if r['click_budget'] == budget]
        pairs = defaultdict(list)
        for r in selected:
            pairs[r['pair_id']].append(r['dice'])
        bf = [r['bf1_um2'] for r in selected if r['bf1_um2'] is not None]
        out[budget] = dict(rows=len(selected), pairs=len(pairs),
            dice=math.fsum(math.fsum(v)/len(v) for v in pairs.values())/len(pairs),
            bf1=math.fsum(bf)/len(bf) if bf else None,
            datasets={d: math.fsum(r['dice'] for r in selected if r['dataset'] == d)/sum(r['dataset'] == d for r in selected)
                      for d in sorted({r['dataset'] for r in selected})})
    return out


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    save('status.json', dict(status='running', worker_pid=os.getpid()))
    try:
        require(read(root/'summary.json')['status'] == 'complete_local_execution', 'coordinator incomplete')
        require(root.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'coordinator exit')
        jobs = read(root/'jobs.json')
        require(len(jobs) == 9 and len({j['key'] for j in jobs}) == 9, 'nine unique conditions')
        completed = read(root/'summary.json')['complete']
        checkers = {}; results = {}; total = 0
        for job in jobs:
            name = job['key']; fold = job['fold']; method = job['method']
            tasks = Path(job['tasks']); test = Path(job['heldout_local'])
            tc = read(tasks/'contract.json'); ec = read(test/'contract.json')
            require(tc['heldout_excluded_labels'] == FOLDS[fold]['exclude'], 'fold exclusions')
            require(ec['heldout_test_targets'] == FOLDS[fold]['test'] and ec['checkpoint_selection_allowed'] is False, 'terminal test scope')
            require(ec['training_protocol_sha256'] == sha(str(tasks/'contract.json')), 'terminal training binding')
            for p in (tasks, test):
                if str(p) not in checkers:
                    checkers[str(p)] = checker(p)
            _, seen_expected, seen_validate = checkers[str(tasks)]
            _, expected, validate = checkers[str(test)]
            train = Path(completed[name+'|train']); scored = Path(completed[name+'|score']); gate = Path(completed[name+'|gate'])
            tr = read(train/'summary.json'); runmeta = read(train/'run.json'); contract = runmeta['contract']
            er = read(scored/'summary.json'); em = read(scored/'run.json'); best = read(train/'best_checkpoint.json')
            require(tr['status'] == er['status'] == 'complete' and tr['gate'] is False and er['gate'] is False, 'formal scope')
            require(runmeta['args']['resume'] is None and runmeta['args']['gate'] is False, 'fresh formal training')
            require(contract['steps'] == tr['step'] == 10000 and contract['accumulate'] == 4 and contract['warmup'] == 1000, 'budget')
            actual = str(Path(job['realized'])/'training_plans.jsonl')
            actual_sha = sha(actual)
            require(actual_sha == sha(str(train/'training_consumed.jsonl')) == tr['training_consumed_sha256'] == contract['training_plans_sha256'] == job['training_plans_sha256'], 'actual training consumption')
            require(sum(1 for _ in Path(actual).open()) == 40000, '40k actual tasks')
            require(tr['full_manifest_parity'] and tr['checkpoint_reload_exact'] and tr['nonzero_joint_backbone_gradient'], 'training acceptance flags')
            require(contract['protocol_sha256'] == em['training_protocol_sha256'] == sha(str(tasks/'contract.json')), 'protocol binding')
            require(em['test_protocol_sha256'] == sha(str(test/'contract.json')) and em['metric_version'] == VERSION, 'test/metric binding')
            for group in ('source_sha256', 'heldout_source_sha256'):
                for p, value in contract.get(group, {}).items():
                    require(sha(p) == value, 'source changed: '+p)
            for p, value in em['source_sha256'].items():
                require(sha(p) == value, 'scorer changed: '+p)
            gr = read(gate/'summary.json'); gm = read(gate/'run.json')
            require(gr['gate'] and gr['status'] == 'complete' and gr['fresh_gate_resume'] and gr['checkpoint_reload_exact'] and gr['nonzero_joint_backbone_gradient'], 'native gate')
            require(Path(gm['args']['output']).resolve() != train.resolve(), 'gate/formal isolation')
            history = tr['history']
            require([h['step'] for h in history] == list(range(1000, 10001, 1000)), 'seen selection schedule')
            seen_keys = {k for k in seen_expected if k[1] == 0 and k[2] == '3+2'}
            for h in history:
                rows = read(train/f'validation_{h["step"]:05d}_1025/rows.json')
                require(len(rows) == len(seen_keys) and {seen_validate(r) for r in rows} == seen_keys, 'seen selection coverage')
                require(all(r['target'] not in FOLDS[fold]['exclude'][r['dataset']] for r in rows), 'excluded selection target')
                close(h['score'], math.fsum(r['dice'] for r in rows)/len(rows), 'seen score recount')
            chosen = max(history, key=lambda h: (h['score'], h['step']))
            require(best['step'] == chosen['step'] == em['selected_seen_step'] and best['score'] == chosen['score'], 'seen-only best selection')
            require(Path(best['path']).resolve() == Path(em['checkpoint']).resolve(), 'selected checkpoint path')
            require(sha(best['path']) == best['checkpoint_sha256'] == em['checkpoint_sha256'], 'selected weight SHA')
            require(sha(str(scored/'rows.jsonl')) == er['rows_sha256'], 'rows SHA')
            rows = [json.loads(line) for line in (scored/'rows.jsonl').read_text().splitlines()]
            require(len(rows) == len(expected) == job['expected_rows'] and {validate(r) for r in rows} == set(expected), 'terminal full coverage')
            require(all(r['target'] in FOLDS[fold]['test'][r['dataset']] for r in rows), 'terminal heldout targets')
            metrics = means(rows)
            for b, m in metrics.items():
                close(m['dice'], er['metrics'][b]['macro_pair_dice'], 'reported macro Dice')
                close(m['bf1'], er['metrics'][b]['bf1_um2'], 'reported BF1 mean')
            for p in (train, scored, gate):
                require(p.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'stage exit')
            results[name] = dict(metrics=metrics, training=str(train), scoring=str(scored), checkpoint_sha256=best['checkpoint_sha256'],
                selected_seen_step=best['step'], training_coordinate_sha256=actual_sha, rows_sha256=er['rows_sha256'])
            total += len(rows)
            save('status.json', dict(status='running', worker_pid=os.getpid(), accepted_conditions=len(results), rows=total))
            print(json.dumps(dict(condition=name, status='accepted_counts_and_binding', primary=metrics['3+2'])), flush=True)
        report = dict(status='accepted_local_counts_and_binding_qualified', worker_pid=os.getpid(), conditions=results,
            terminal_rows=total, seconds=time.time()-started, source_sha256=sha(__file__),
            limits=['single seed', 'no full prediction-mask replay or full BF1 distance recount', 'paired statistics pending',
                    'label-supervision heldout; morphology/public pretraining not excluded', 'local simulated query clicks; separate remote table'])
        save('summary.json', report)
        save('status.json', dict(status=report['status'], worker_pid=os.getpid(), rows=total))
    except Exception:
        save('status.json', dict(status='failed', worker_pid=os.getpid(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
