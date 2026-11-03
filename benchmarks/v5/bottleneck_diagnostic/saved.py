"""Audit existing P3 masks: errors, area strata, budgets and matched tasks."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha
from benchmarks.v5.fair_comparison.metrics import boundary


def summarize(rows):
    def mean(key, group):
        v = [r[key] for r in group if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    present = [r for r in rows if r['positive'] > 0]
    absent = [r for r in rows if r['positive'] == 0]
    per_target = defaultdict(list)
    for r in present:
        per_target[r['dataset']+'|'+str(r['target'])].append(r)
    tp, fp, fn = (sum(r[x] for r in present) for x in ['tp', 'fp', 'fn'])
    return dict(n=len(rows), present_n=len(present), absent_n=len(absent),
                mean_Dice=mean('dice', present),
                macro_Dice=float(np.mean([mean('dice', v) for v in per_target.values()])) if per_target else None,
                mean_BF1=mean('bf1_um2', present),
                absent_FPA=mean('false_positive_fraction', absent),
                mean_precision=mean('precision', present), mean_recall=mean('recall', present),
                pixel_precision=tp/(tp+fp) if tp+fp else None,
                pixel_recall=tp/(tp+fn) if tp+fn else None,
                empty_present_fraction=mean('empty', present),
                mean_area_ratio=mean('pred_to_true_area', present),
                FP_outside_8um_fraction=sum(r['fp_far8'] for r in present)/fp if fp else None,
                FN_inside_8um_fraction=1-sum(r['fn_far8'] for r in present)/fn if fn else None)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    contract = json.loads((root/'contract.json').read_text()); formal = Path(contract['formal'])
    _, inventory = load_cell_catalog(TASKS)
    out = root/'saved_masks'; out.mkdir(exist_ok=False)
    started = time.monotonic()

    @lru_cache(maxsize=256)
    def labels(identity):
        r = inventory[identity]
        gt = np.asarray(Image.open(r['labels'])).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest() == r['labels_array_sha256']
        return gt, np.asarray(json.loads(r['input_mpp_xy_json']))

    @lru_cache(maxsize=96)
    def truth_distance(identity, target):
        gt, mpp = labels(identity); valid = gt != 255
        edge = boundary(gt == target, valid)
        return distance_transform_edt(~edge, sampling=mpp[::-1]) if edge.any() else None

    results, evidence = {}, {}
    for arm in contract['arms']:
        proof = json.loads((formal/arm/'independent_acceptance.json').read_text())
        assert proof['rows_sha256'] == sha(formal/arm/'rows.json')
        assert proof['masks_sha256'] == sha(formal/arm/'masks.npz')
        evidence[arm] = proof
        rows = json.loads((formal/arm/'rows.json').read_text())
        with np.load(formal/arm/'masks.npz') as f:
            masks = f['masks'].copy()
        enriched = []; budget_rows = defaultdict(list)
        for index, row in enumerate(rows):
            plan = row['plan']; m = row['metrics']
            n = m['tp']+m['fp']; positive = m['positive']
            entry = dict(m, dataset=plan['dataset'], target=plan['target'], kind=plan['kind'],
                         scope=row['scope'], query_id=plan['query_id'], support_id=plan['support_id'],
                         budget=row['budget'], positive_fraction=positive/m['valid'],
                         precision=m['tp']/n if n else (0. if positive else None),
                         recall=m['tp']/positive if positive else None,
                         pred_to_true_area=n/positive if positive else None, empty=n==0,
                         fp_far8=0, fn_far8=0)
            budget_rows[row['budget']+'|'+row['scope']].append(entry)
            if row['budget'] != '3+2':
                continue
            gt, mpp = labels(plan['query_id']); valid = gt != 255; truth = gt == plan['target']
            pred = np.unpackbits(masks[index], count=512*512).reshape(512, 512).astype(bool)
            tp = int((pred & truth & valid).sum()); fp = pred & ~truth & valid; fn = ~pred & truth & valid
            assert (tp, int(fp.sum()), int(fn.sum()), int(valid.sum())) == (m['tp'], m['fp'], m['fn'], m['valid'])
            distance = truth_distance(plan['query_id'], plan['target'])
            if distance is not None:
                entry['fp_far8'] = int((fp & (distance > 8.)).sum())
                entry['fn_far8'] = int((fn & (distance > 8.)).sum())
            else:
                entry['fp_far8'] = int(fp.sum()); entry['fn_far8'] = int(fn.sum())
            entry['FP_true_labels'] = {str(int(k)):int(v) for k,v in zip(*np.unique(gt[fp], return_counts=True))}
            enriched.append(entry)
            if len(enriched) % 100 == 0:
                state = dict(worker_pid=os.getpid(), arm=arm, rows=len(enriched), total=1237, seconds=time.monotonic()-started)
                (out/'status.json').write_text(json.dumps(state)); print(json.dumps(state), flush=True)
        assert len(enriched) == 1237
        groups = {'all': enriched}
        for row in enriched:
            domain, scope, target = row['dataset'], row['scope'], row['target']
            for key in [scope, domain+'|'+scope, domain+'|'+str(target)+'|'+scope]:
                groups.setdefault(key, []).append(row)
            if row['positive'] > 0:
                area = row['positive_fraction']
                key = scope+'|area_'+('lt5pct' if area < .05 else '5to25pct' if area < .25 else 'ge25pct')
                groups.setdefault(key, []).append(row)
        local = {(r['query_id'],r['target']):r for r in enriched if r['scope']=='local'}
        paired = []
        for r in enriched:
            key = (r['query_id'],r['target'])
            if r['scope']=='remote' and key in local and r['positive'] > 0:
                paired.append(dict(query_id=key[0], target=key[1], dataset=r['dataset'],
                                   local_Dice=local[key]['dice'], remote_Dice=r['dice'],
                                   difference=r['dice']-local[key]['dice']))
        curves = {}
        for step in [0,250,500,1000,2000]:
            monitor = formal/arm/f'train_monitor_{step}.json'
            data = json.loads(monitor.read_text()); curves[str(step)] = data['groups']
        results[arm] = dict(groups={k:summarize(v) for k,v in groups.items()},
                           budgets={k:{x:y for x,y in summarize(v).items() if x not in ['FP_outside_8um_fraction','FN_inside_8um_fraction']} for k,v in budget_rows.items()},
                           same_query_target_pairs=paired, same_query_target_mean_difference=float(np.mean([r['difference'] for r in paired])) if paired else None,
                           train32_curves=curves)
        (out/(arm+'_rows.json')).write_text(json.dumps(enriched, allow_nan=False)+'\n')
    report = dict(status='accepted_saved_mask_error_decomposition', source_sha256=sha(__file__),
                  evidence=evidence, results=results, main_masks_counted=2474, threshold=.5,
                  no_new_inference=True, no_model_selection=True,
                  limitations=['8um band is a fixed error diagnostic, not a new acceptance metric',
                               'local/remote marginal distributions differ; paired subset uses existing prompts only',
                               'first32 train monitor is not representative or a matched generalization estimate'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], output=str(out))), flush=True)


if __name__ == '__main__':
    main()
