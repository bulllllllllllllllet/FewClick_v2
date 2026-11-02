"""Describe every accepted mask, including empty predictions and BF1 denominators."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.v5.encoder_research.continuous_data import sha


def describe(items):
    present = [r for r in items if r['positive'] > 0]
    absent = [r for r in items if r['positive'] == 0]
    evaluable = [r for r in present if r['bf1_um2'] is not None]
    def mean(values):
        return float(np.mean(values)) if values else None
    return dict(tasks=len(items), present=len(present), absent=len(absent),
                BF1_evaluable_present=len(evaluable), BF1_NA_present=len(present)-len(evaluable),
                raw_empty_present=sum(r['raw_prediction_pixels'] == 0 for r in present),
                valid_empty_present=sum(r['tp'] + r['fp'] == 0 for r in present),
                raw_full_present=sum(r['raw_prediction_pixels'] == 512*512 for r in present),
                valid_full_present=sum(r['tp'] + r['fp'] == r['valid'] for r in present),
                valid_empty_absent=sum(r['tp'] + r['fp'] == 0 for r in absent),
                present_Dice=mean([r['dice'] for r in present]),
                present_BF1=mean([r['bf1_um2'] for r in evaluable]),
                absent_FPA=mean([r['false_positive_fraction'] for r in absent]),
                mean_present_prediction_fraction=mean([(r['tp'] + r['fp']) / r['valid'] for r in present]),
                mean_present_GT_fraction=mean([r['positive'] / r['valid'] for r in present]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    accepted = json.loads((a.run / 'independent_acceptance.json').read_text())
    assert accepted['status'] == 'accepted_P2_vocabulary_evaluation' and not accepted['smoke']
    assert accepted['rows'] == 3711 and accepted['report_sha256'] == sha(a.run / 'report.json')
    report = json.loads((a.run / 'report.json').read_text())
    assert report['updates'] == 2000 and report['checkpoint_sha256'] == accepted['checkpoint_sha256']
    raw = json.loads((a.run / 'rows.json').read_text())
    with np.load(a.run / 'masks.npz', allow_pickle=False) as f:
        packed = f['masks'].copy()
    assert packed.dtype == np.uint8 and packed.shape == (len(raw), 32768) and len(raw) == 3711
    counts = np.concatenate([np.unpackbits(packed[i:i+64], axis=1).sum(1) for i in range(0, len(raw), 64)])
    # Independent byte popcounts, rather than the unpackbits counting path.
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    independent = table[packed].sum(1)
    assert np.array_equal(counts, independent)
    groups = defaultdict(list)
    for source, count in zip(raw, counts):
        row = dict(source['plan'], **source['metrics'], raw_prediction_pixels=int(count))
        assert 0 < row['valid'] <= 512*512
        predicted_valid = row['tp'] + row['fp']
        assert 0 <= predicted_valid <= row['valid'] and predicted_valid <= count
        assert int(count) - predicted_valid <= 512*512 - row['valid']
        assert (row['bf1_um2'] is not None) == bool(row['boundary_evaluable'])
        assert bool(row['boundary_evaluable']) == (row['true_boundary_pixels'] > 0)
        key = source['scope'] + '|' + source['budget']
        groups[key].append(row)
        groups[key + '|' + row['dataset'] + '|' + str(row['target'])].append(row)
    coverage = {k: describe(v) for k, v in sorted(groups.items())}
    assert {k: v['tasks'] for k, v in coverage.items() if k.count('|') == 1} == {
        f'{scope}|{budget}': n for scope, n in [('local', 302), ('remote', 935)]
        for budget in ['1+1', '3+2', '5+1']}
    result = dict(status='accepted_complete_mask_coverage_diagnostics', worker_pid=os.getpid(),
                  arm=report['arm'], coverage=coverage,
                  evidence={name: sha(a.run / name) for name in ['report.json', 'independent_acceptance.json', 'rows.json', 'masks.npz']},
                  source_sha256=sha(__file__), byte_popcount_replay_exact=True,
                  inference_or_threshold_changed=False, P2_accepted=False,
                  limitations=['counts describe existing predictions; no validation-driven repair or selection',
                               'tasks and budgets reuse queries and supports; not independent WSI samples',
                               'raw masks and scored-valid masks have different denominators'])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in coverage.items() if k in ['local|3+2', 'remote|3+2']}, indent=2))


if __name__ == '__main__':
    main()
