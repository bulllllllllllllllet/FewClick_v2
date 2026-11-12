import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .model import MODES
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(indices, tasks, metrics):
    groups = defaultdict(list)
    for i in indices:
        groups[tasks[i]['target']].append(metrics[i])
    classes = {}
    for target, rows in groups.items():
        positive = [r for r in rows if r['positive']]
        absent = [r for r in rows if not r['positive']]
        classes[target] = dict(present=len(positive), absent=len(absent),
                               dice=mean([r['dice'] for r in positive]),
                               FPA=mean([r['false_positive_fraction'] for r in absent]),
                               precision=mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0 for r in positive]),
                               recall=mean([r['tp']/(r['tp']+r['fn']) for r in positive]),
                               bf1=mean([r['bf1_um2'] for r in positive if r['boundary_evaluable']]))
    return dict(n=len(indices), classes=classes, **{
        'macro_'+key: mean([r[key] for r in classes.values() if r[key] is not None])
        for key in ['dice', 'FPA', 'precision', 'recall', 'bf1']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    c = json.loads((root/'contract.json').read_text())
    tasks = json.loads((root/'tasks.json').read_text())
    assert len(tasks) == 11000 and sha(root/'tasks.json') == c['tasks_sha256']
    all_metrics = {mode: {} for mode in MODES}
    for shard in [0, 1]:
        out = root/f'shard_{shard}'
        acceptance = json.loads((out/'acceptance.json').read_text())
        assert acceptance['status'] == 'complete_accepted' and acceptance['rows'] == 5500
        assert acceptance['control_matches_historical_masks_and_metrics']
        assert acceptance['tasks_sha256'] == c['tasks_sha256']
        assert acceptance['checkpoint_sha256'] == c['checkpoints']['H']['sha256']
        for mode in MODES:
            for filename, key in [('rows.json', 'rows_sha256'), ('masks.npy', 'masks_sha256')]:
                assert sha(out/mode/filename) == acceptance['modes'][mode][key]
            for row in json.loads((out/mode/'rows.json').read_text()):
                assert row['index'] not in all_metrics[mode]
                all_metrics[mode][row['index']] = row['metrics']
    reports = {}
    for mode, metrics in all_metrics.items():
        assert sorted(metrics) == list(range(11000))
        cuts = {'all': list(range(11000))}
        for lower, upper, name in [(0, .01, 'lt1pct'), (.01, .05, '1to5pct'),
                                   (.05, .2, '5to20pct'), (.2, 1.01, 'large_ge20pct')]:
            cuts[name] = [i for i, m in metrics.items() if m['positive'] and lower <= m['positive']/m['valid'] < upper]
        reports[mode] = {name: summarize(indices, tasks, metrics) for name, indices in cuts.items()}
    differences = {mode: {scope: {
        key: reports[mode][scope][key]-reports['control'][scope][key]
        for key in ['macro_dice', 'macro_FPA', 'macro_bf1', 'macro_precision', 'macro_recall']
        if reports[mode][scope][key] is not None and reports['control'][scope][key] is not None}
        for scope in reports[mode]} for mode in MODES if mode != 'control'}
    dump(root/'report.json', dict(status='complete_accepted_inference_diagnostic', reports=reports,
                                  differences_from_control=differences, training_updates=0,
                                  threshold=.5, query_human_clicks=0, selection_used=False,
                                  limitations='Fixed H weights; interventions change inference feature distribution. '
                                  'Sensitivity only, not matched retraining or causal proof of training benefit. '
                                  'No_anchor retains other click-conditioned paths; no_coarse_prior retains coarse transport channel. '
                                  'Repeatedly exposed within-cohort same-slide spatial validation, not independent generalization.'))
    dump(root/'acceptance.json', dict(status='complete_accepted_inference_diagnostic',
                                      report_sha256=sha(root/'report.json'), tasks_sha256=c['tasks_sha256'],
                                      tasks=11000, variants=4, predictions=44000, no_training=True))


if __name__ == '__main__':
    main()
