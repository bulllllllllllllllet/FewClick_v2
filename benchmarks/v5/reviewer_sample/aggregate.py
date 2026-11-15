"""Stratum summaries plus original class composition weighting, never silently pooled."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def summarize(rows, tasks, sampling):
    assert set(rows) == {t['index'] for t in tasks}
    by = defaultdict(list)
    for t in tasks:
        s = sampling['task_strata'][str(t['index'])]; m = rows[t['index']]
        assert m['positive'] == t['query_positive_pixels'] and m['valid'] == s['valid']
        by[t['target'], s['stratum']].append(m)
    classes = {}
    for target, strata in sampling['strata'].items():
        target = int(target); groups = {}; numerator = denom = 0
        for kind in ['small', 'large', 'absent']:
            rr = by[target, kind]; n = strata[kind]['population']
            assert len(rr) == strata[kind]['selected']
            value = float(np.mean([r['dice' if kind != 'absent' else 'false_positive_fraction'] for r in rr])) if rr else None
            groups[kind] = dict(n=len(rr), population=n, mean=value)
            if kind != 'absent' and n:
                assert value is not None; numerator += n*value; denom += n
        classes[target] = dict(strata=groups, present_Dice_original_composition=numerator/denom)
    return dict(classes=classes, macro_Dice=float(np.mean([v['present_Dice_original_composition'] for v in classes.values()])),
        small_macro_Dice=float(np.mean([v['strata']['small']['mean'] for v in classes.values() if v['strata']['small']['mean'] is not None])),
        large_macro_Dice=float(np.mean([v['strata']['large']['mean'] for v in classes.values() if v['strata']['large']['mean'] is not None])),
        macro_absent_FPA=float(np.mean([v['strata']['absent']['mean'] for v in classes.values() if v['strata']['absent']['mean'] is not None])),
        note='Fixed spatial-balanced development subset. Original within-class small/large composition weights, class macro; not a new blind test or unbiased probability-sample estimator.')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args(); r = a.root
    c = json.loads((r/'contract.json').read_text()); tasks = json.loads((r/'tasks.json').read_text()); rows = {}
    assert sha(r/'tasks.json') == c['tasks_sha256']
    for shard in [0, 1]:
        out = r/f"{c['method']}_{shard}"; acc = json.loads((out/'acceptance.json').read_text())
        assert acc['status'] == 'complete_accepted' and acc['rows'] == sum(t['index']%2 == shard for t in tasks)
        assert acc['tasks_sha256'] == c['tasks_sha256'] and acc['checkpoint_sha256'] == c['checkpoints'][c['method']]['sha256']
        assert sha(out/'rows.json') == acc['rows_sha256'] and sha(out/'masks.npy') == acc['masks_sha256']
        for row in json.loads((out/'rows.json').read_text()):
            assert row['index'] not in rows; rows[row['index']] = row['metrics']
    result = summarize(rows, tasks, json.loads((r/'sampling.json').read_text()))
    dump(r/'report.json', dict(status='complete_accepted', method=c['method'], **result))
    dump(r/'acceptance.json', dict(status='complete_accepted', rows=len(tasks), tasks_sha256=c['tasks_sha256'], report_sha256=sha(r/'report.json')))


if __name__ == '__main__': main()
