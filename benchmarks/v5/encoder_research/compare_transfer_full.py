"""Compare all four accepted fixed-checkpoint transfer evaluations without selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.v5.transfer_comparison.data import TransferData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_change(control, mixed, data, metric, present):
    groups = defaultdict(list)
    differences = []
    for left, right in zip(control, mixed):
        identity = ('budget', 'pair_id', 'support_id', 'query_id', 'target')
        assert all(left[key] == right[key] for key in identity)
        assert left['positive'] == right['positive']
        if left['budget'] != '3+2' or (left['positive'] > 0) != present:
            continue
        if left[metric] is None or right[metric] is None:
            assert left[metric] is None and right[metric] is None
            continue
        delta = right[metric] - left[metric]
        differences.append(delta)
        groups[data.rows[left['query_id']]['space_key']].append(delta)
    if not differences:
        return dict(pairs=0, mixed_minus_control=None, conditional_CI95=None)
    interval = None
    if len(groups) >= 2:
        sums = np.array([sum(v) for v in groups.values()])
        counts = np.array([len(v) for v in groups.values()])
        sample = np.random.default_rng(20261031).integers(0, len(groups), (2000, len(groups)))
        interval = np.quantile(sums[sample].sum(1) / counts[sample].sum(1), [.025, .975]).tolist()
    return dict(pairs=len(differences), mixed_minus_control=float(np.mean(differences)),
                query_space_clusters=len(groups), conditional_CI95=interval,
                qualification='fixed support bank; support reuse and same-patient dependence not fully accounted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.run / 'contract.json').read_text())
    data = TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    receipts, rows, evidence, original = {}, {}, {}, {}
    for name, reference in contract['references'].items():
        directory = args.run / name
        receipt = json.loads((directory / 'independent_acceptance.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        assert receipt['status'] == 'accepted_full935_transfer'
        assert receipt['rows_replayed'] == 2805 and receipt['reference240_masks_exact']
        assert receipt['checkpoint_sha256'] == report['checkpoint_sha256'] == contract['checkpoint_sha256'][name]
        assert receipt['frozen_tasks_sha256'] == report['frozen_tasks_sha256'] == sha(data.tasks / 'validation_maximum.json')
        assert sha(Path(reference) / 'checkpoint.pt') == receipt['checkpoint_sha256']
        assert report['source_sha256'] == contract['sources']['benchmarks/v5/encoder_research/evaluate_transfer_full.py']
        assert receipt['source_sha256'] == contract['sources']['benchmarks/v5/encoder_research/accept_transfer_full.py']
        prior = json.loads((Path(reference) / 'independent_acceptance.json').read_text())
        assert prior['status'] == 'accepted_fixed_image_pilot' and prior['rows_replayed'] == 1146
        original[name] = prior['summary']
        receipts[name] = receipt
        rows[name] = json.loads((directory / 'rows.json').read_text())
        evidence[name] = {file: sha(directory / file) for file in
                          ['independent_acceptance.json', 'report.json', 'rows.json', 'packed_masks.npz']}
    assert set(rows) == {'seed0_local_control', 'seed0_mixed', 'seed1_local_control', 'seed1_mixed'}
    target_counts = next(iter(receipts.values()))['by_target']['3+2']
    coverage = dict(target_groups=len(target_counts),
                    present_target_groups=sum(v['present'] > 0 for v in target_counts.values()),
                    no_present_targets=[k for k, v in target_counts.items() if not v['present']],
                    present_pairs=470, absent_pairs=465, unique_query=330, unique_support=212)
    paired = {}
    targets = {}
    for seed in [0, 1]:
        control, mixed = (rows[f'seed{seed}_{arm}'] for arm in ['local_control', 'mixed'])
        paired[str(seed)] = {
            metric: paired_change(control, mixed, data, metric, present)
            for metric, present in [('dice', True), ('bf1_um2', True), ('false_positive_fraction', False)]}
        targets[str(seed)] = {}
        for dataset, target in sorted({(r['dataset'], r['target']) for r in control}):
            left = [r for r in control if (r['dataset'], r['target']) == (dataset, target)]
            right = [r for r in mixed if (r['dataset'], r['target']) == (dataset, target)]
            targets[str(seed)][f'{dataset}|{target}'] = {
                metric: paired_change(left, right, data, metric, present)
                for metric, present in [('dice', True), ('bf1_um2', True), ('false_positive_fraction', False)]}
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='accepted_full935_paired_comparison', primary_budget='3+2', evidence=evidence,
                  by_arm={k: v['by_budget'] for k, v in receipts.items()}, paired_changes=paired,
                  original_local_and80=original,
                  coverage=coverage,
                  by_target_paired_changes=targets, source_sha256=sha(__file__),
                  scope='expanded previously exposed development cohort, not a blind test',
                  guard_policy='Original80 guards and secondseed BF1 failure retained; no thresholds refitted',
                  decision='No automatic fullWSI promotion; complete P1 cell contribution controls next')
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 完整935对固定模型验证', '',
             '开发集扩大验证；935对共享330张query和212张support。旧80对门槛保留。', '',
             '16组dataset/target中15组有present任务；TIGER target3仅2组absent，检出质量为NA。', '',
             '| seed | 训练任务 | present Dice | present BF1 | absent FPA |',
             '|---|---|---:|---:|---:|']
    for name, receipt in receipts.items():
        v = receipt['by_budget']['3+2']
        lines.append(f"| {name.split('_')[0]} | {name.split('_', 1)[1]} | {v['dice']:.4f} | {v['BF1']:.4f} | {v['absent_FPA']:.4f} |")
    def display(value):
        return 'NA' if value is None else f'{value:.4f}'
    for seed in [0, 1]:
        lines += ['', f'## seed{seed}逐组织结果', '',
                  '| dataset / target | present n | absent n | control Dice | mixed Dice | control FPA | mixed FPA |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        left = receipts[f'seed{seed}_local_control']['by_target']['3+2']
        right = receipts[f'seed{seed}_mixed']['by_target']['3+2']
        assert left.keys() == right.keys()
        for target in left:
            a, b = left[target], right[target]
            assert (a['present'], a['absent']) == (b['present'], b['absent'])
            label = target.replace('|', ' / ')
            lines.append(f"| {label} | {a['present']} | {a['absent']} | {display(a['dice'])} | {display(b['dice'])} | {display(a['absent_FPA'])} | {display(b['absent_FPA'])} |")
    lines += ['', '各组织样本数、NA、逐类差异和条件区间见comparison.json及每模型independent_acceptance.json。',
              '条件区间只对固定support库下的query空间来源重采样，不作为独立患者泛化置信区间。',
              '两个seed分别报告，不将seed数、任务数或点击预算当作独立样本数。']
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], paired_changes=paired)), flush=True)


if __name__ == '__main__':
    main()
