"""Compare the single frozen-common-decoder cell adaptation revision after acceptance."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .accept_cell_only_pilot import summarize
from .cell_only_checks import ParentMaskReference
from .continuous_data import ContinuousData, sha
from .continuous_prompt import ARMS


def summary_values(receipt):
    remote, local = receipt['summary']['remote|3+2'], receipt['summary']['local|3+2']
    return dict(remote_macro_Dice=remote['class_macro_Dice'], remote_Dice=remote['dice'], remote_BF1=remote['BF1'],
                absent_FPA=remote['absent_FPA'], local_Dice=local['dice'], local_BF1=local['BF1'],
                small_Dice=receipt['local_strata']['small']['dice'], fat_Dice=receipt['local_strata']['fat']['dice'])


def resource_guards(reference, candidate, bounds):
    assert all(value is not None for value in reference.values())
    assert all(value is not None for value in candidate.values())
    tests = dict(primary_Dice_gain=candidate['remote_macro_Dice'] >= reference['remote_macro_Dice'] + bounds['primary_Dice_gain'],
                 remote_BF1=candidate['remote_BF1'] >= reference['remote_BF1'] - bounds['remote_BF1_max_drop'],
                 absent_FPA=candidate['absent_FPA'] <= reference['absent_FPA'] + bounds['absent_FPA_max_increase'],
                 local_Dice=candidate['local_Dice'] >= reference['local_Dice'] - bounds['local_Dice_max_drop'],
                 small_Dice=candidate['small_Dice'] >= reference['small_Dice'] - bounds['small_Dice_max_drop'],
                 fat_Dice=candidate['fat_Dice'] >= reference['fat_Dice'] - bounds['fat_Dice_max_drop'])
    return dict(tests=tests, pass_all=all(tests.values()),
                candidate_minus_reference={k: candidate[k] - reference[k] for k in reference})


def paired_interval(reference, candidate, inventory, scope, metric, present, macro=False):
    groups, targets = {}, {}
    records = []
    for left, right in zip(reference, candidate):
        if left['scope'] != scope or left['budget'] != '3+2' or (left['positive'] > 0) != present:
            continue
        if left[metric] is None or right[metric] is None:
            assert left[metric] is None and right[metric] is None
            continue
        cluster = inventory[left['query_id']]['space_key']
        target = (left['dataset'], left['target']) if macro else ('all', 0)
        groups.setdefault(cluster, len(groups))
        targets.setdefault(target, len(targets))
        records.append((groups[cluster], targets[target], right[metric] - left[metric]))
    sums = np.zeros((len(groups), len(targets)))
    counts = np.zeros_like(sums)
    for group, target, difference in records:
        sums[group, target] += difference
        counts[group, target] += 1
    point = float(np.mean(sums.sum(0) / counts.sum(0)))
    interval, valid_draws = None, 0
    if len(groups) >= 2:
        draws = np.random.default_rng(20261031).integers(0, len(groups), (2000, len(groups)))
        totals, n = sums[draws].sum(1), counts[draws].sum(1)
        valid = (n > 0).all(1)
        valid_draws = int(valid.sum())
        if valid_draws >= 100:
            estimates = (totals[valid] / n[valid]).mean(1)
            interval = np.quantile(estimates, [.025, .975]).tolist()
    return dict(pairs=len(records), query_space_clusters=len(groups), target_groups=len(targets),
                candidate_minus_reference=point, conditional_interval95=interval,
                bootstrap_draws=2000, valid_class_coverage_draws=valid_draws,
                limitation='fixed support bank only; same-patient/support reuse not fully accounted; draws missing a required class excluded and counted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-transfer-reference', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.run / 'contract.json').read_text())
    assert (args.run / 'exit_code').read_text() == '0'
    assert contract['updates_per_arm'] == 1000 and contract['cumulative_optimizer_calls'] == 3000
    assert contract['shared_steps'] == 2000 and contract['cell_steps'] == 1000
    assert str(args.parent_transfer_reference) == contract['parent_transfer_reference']
    addendum = args.run / 'contract_mask_count_addendum.json'
    if addendum.exists():
        assert json.loads(addendum.read_text())['formal_expected_masks_per_arm'] == {arm: 3711 for arm in ARMS}
    for path, value in contract['inputs_sha256'].items():
        assert sha(path) == value
    for path, value in contract['sources'].items():
        assert sha(path) == value
    assert sha(contract['parent']) == contract['parent_sha256']
    assert sha(contract['training']) == contract['training_sha256']
    assert sha(contract['gate_acceptance']) == contract['gate_acceptance_sha256']
    receipts, rows, reports, evidence = {}, {}, {}, {}
    for arm in ARMS:
        directory = args.run / arm
        report = json.loads((directory / 'report.json').read_text())
        receipt = json.loads((directory / 'independent_acceptance.json').read_text())
        assert receipt['status'] == 'accepted_cell_only_pilot' and receipt['arm'] == arm
        assert receipt['rows_replayed'] == 3711 and receipt['all_metrics_replayed']
        assert receipt['acceptance_source_sha256'] == contract['sources']['benchmarks/v5/encoder_research/accept_cell_only_pilot.py']
        assert receipt['parent_sha256'] == report['parent_sha256'] == contract['parent_sha256']
        assert receipt['training_sha256'] == report['consumed_sha256'] == contract['training_sha256']
        assert report['total_updates'] == 3000 and report['training_updates'] == 1000
        assert report['seed'] == contract['seed']
        assert report['update_policy'] == receipt['update_policy'] == 'cell_only_frozen_shared'
        assert report['shared_updates'] == receipt['shared_updates'] == 2000
        assert receipt['cell_updates'] == 1000 and receipt['shared_weights_and_optimizer_exact']
        assert receipt['initial_identity']['trainable_parameters'] == 76800
        assert sha(directory / 'checkpoint.pt') == receipt['checkpoint_sha256'] == report['checkpoint_sha256']
        assert sha(directory / 'training_consumed.jsonl') == contract['training_sha256']
        for name, value in receipt['sources'].items():
            assert contract['sources']['benchmarks/v5/encoder_research/' + name] == value
        receipts[arm], reports[arm] = receipt, report
        rows[arm] = json.loads((directory / 'rows.json').read_text())
        grouped = defaultdict(list)
        for row in rows[arm]:
            grouped[row['scope'] + '|' + row['budget']].append(row)
        assert {k: summarize(v) for k, v in grouped.items()} == receipt['summary']
        local = grouped['local|3+2']
        strata = {'small': summarize([r for r in local if r['positive'] / r['valid'] < .05]),
                  'fat': summarize([r for r in local if r['dataset'] == 'crc' and r['target'] == 10])}
        assert strata == receipt['local_strata']
        evidence[arm] = {name: sha(directory / name) for name in
                         ['report.json', 'independent_acceptance.json', 'rows.json', 'packed_masks.npz', 'checkpoint.pt']}
    data = ContinuousData(contract['images'], contract['grids'])
    keys = ['scope', 'budget', 'pair_id', 'dataset', 'target', 'support_id', 'query_id', 'kind', 'positive', 'valid', 'true_boundary_pixels']
    for arm in ['xcell', 'lemon']:
        assert receipts[arm]['inputs'] == receipts['image_only']['inputs'] == data.identity
        assert receipts[arm]['initial_identity'] == receipts['image_only']['initial_identity']
        assert len(rows[arm]) == len(rows['image_only']) == 3711
        for left, right in zip(rows['image_only'], rows[arm]):
            assert all(left[k] == right[k] for k in keys)
    assert receipts['image_only']['parent_masks_checked'] == 3711
    parent_masks = ParentMaskReference(contract['parent'], args.parent_transfer_reference)
    with np.load(args.run / 'image_only/packed_masks.npz', allow_pickle=False) as file:
        for row, packed in zip(rows['image_only'], file['masks']):
            parent_masks.assert_equal(row, packed)
    values = {arm: summary_values(value) for arm, value in receipts.items()}
    changes = {}
    for label in contract['comparison_plan']['pairs']:
        candidate, reference = label.split('_vs_')
        result = resource_guards(values[reference], values[candidate], contract['guard'])
        result['paired_uncertainty'] = {
            'remote_macro_Dice': paired_interval(rows[reference], rows[candidate], data.rows, 'remote', 'dice', True, True),
            'remote_Dice': paired_interval(rows[reference], rows[candidate], data.rows, 'remote', 'dice', True),
            'remote_BF1': paired_interval(rows[reference], rows[candidate], data.rows, 'remote', 'bf1_um2', True),
            'absent_FPA': paired_interval(rows[reference], rows[candidate], data.rows, 'remote', 'false_positive_fraction', False),
            'local_Dice': paired_interval(rows[reference], rows[candidate], data.rows, 'local', 'dice', True)}
        assert abs(result['candidate_minus_reference']['remote_macro_Dice'] - result['paired_uncertainty']['remote_macro_Dice']['candidate_minus_reference']) < 1e-12
        changes[label] = result
    eligible = [arm for arm in ['xcell', 'lemon'] if changes[arm + '_vs_image_only']['pass_all']]
    parent_context = None
    if args.parent_transfer_reference:
        parent_remote = json.loads((args.parent_transfer_reference / 'independent_acceptance.json').read_text())
        assert parent_remote['status'] == 'accepted_full935_transfer'
        assert parent_remote['checkpoint_sha256'] == contract['parent_sha256']
        parent_local = json.loads((Path(contract['parent']).parent / 'independent_acceptance.json').read_text())
        assert parent_local['status'] == 'accepted_fixed_image_pilot'
        r, l = parent_remote['by_budget']['3+2'], parent_local['summary']['local|3+2']
        parent_context = dict(updates=2000, remote_macro_Dice=r['class_macro_Dice'], remote_Dice=r['dice'],
                              remote_BF1=r['BF1'], absent_FPA=r['absent_FPA'], local_Dice=l['dice'],
                              checkpoint_sha256=contract['parent_sha256'],
                              remote_acceptance_sha256=sha(args.parent_transfer_reference / 'independent_acceptance.json'),
                              qualification='frozen common state and exact image-only prediction reference; cell-only conditional ablation, not optimized strong baseline')
        for key in ['remote_macro_Dice', 'remote_Dice', 'remote_BF1', 'absent_FPA', 'local_Dice']:
            assert abs(values['image_only'][key] - parent_context[key]) < 1e-12
    result = dict(status='accepted_preregistered_cell_only_comparison', worker_pid=os.getpid(),
                  source_sha256=sha(__file__), contract_sha256=sha(args.run / 'contract.json'), evidence=evidence,
                  values=values, comparisons=changes, eligible_for_second_seed=eligible,
                  parent_context=parent_context, update_policy='cell_only_frozen_shared',
                  shared_updates=2000, cell_updates=1000, global_optimizer_calls=3000, image_only_parent_masks_exact=3711,
                  parent_reference_identity=parent_masks.identity,
                  contract_mask_addendum_sha256=sha(addendum) if addendum.exists() else None,
                  summaries={arm: r['summary'] for arm, r in receipts.items()},
                  local_strata={arm: r['local_strata'] for arm, r in receipts.items()},
                  original80_guards_unchanged=True, fullWSI_promoted=False,
                  next_action='independent second-seed confirmation required for eligible arms' if eligible else 'cell residual adaptation recipe budget exhausted; stop and finish remaining P1 evidence; preserve all negatives',
                  limitations=['previously exposed development set, not blind/external confirmation',
                               'one training seed; conditional source-space intervals do not replace seed replication',
                               'shared weights and optimizer state fixed; only76800 cell parameters train; image-only zero input has no effective adaptation',
                               'XCell context/mask prior differs from LEMON center crop; retained cap255 unchanged',
                               'TIGER target3 has only2 absent tasks, no present detection estimate'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 冻结公共decoder：细胞单独适配', '',
             f"固定seed{contract['seed']}、共同mixed2000步起点被冻结，仅cell投影1000×4。开发集，无整片晋升。", '',
             '| arm | local Dice | remote Dice | remote macroDice | remote BF1 | absent FPA | small Dice | fat Dice |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in values.items():
        columns = ['local_Dice', 'remote_Dice', 'remote_macro_Dice', 'remote_BF1', 'absent_FPA', 'small_Dice', 'fat_Dice']
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{row[key]:.4f}' for key in columns) + ' |')
    lines += ['', '| 比较 | 主macroDice变化pp | 全守门通过 | 未通过项 |', '|---|---:|---|---|']
    for label, row in changes.items():
        failed = ', '.join(k for k, v in row['tests'].items() if not v) or '无'
        lines.append(f"| {label} | {row['candidate_minus_reference']['remote_macro_Dice'] * 100:+.3f} | {row['pass_all']} | {failed} |")
    if parent_context:
        lines += ['', f"原2000步起点的remote macroDice={parent_context['remote_macro_Dice']:.4f}、Dice={parent_context['remote_Dice']:.4f}、FPA={parent_context['absent_FPA']:.4f}、local Dice={parent_context['local_Dice']:.4f}。",
                  '本轮A全部3711个mask与该父模型逐位一致，共享参数/optimizer保持2000步；评价三臂同样遵守原门槛。']
    for scope in ['local', 'remote']:
        lines += ['', f'## {scope} 3+2逐组织结果', '',
                  '| dataset / target | present n | absent n | A Dice | XCell Dice | LEMON Dice | A FPA | XCell FPA | LEMON FPA |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        tables = {arm: r['summary'][scope + '|3+2']['per_target'] for arm, r in receipts.items()}
        def display(value):
            return 'NA' if value is None else f'{value:.4f}'
        for target, counts in tables['image_only'].items():
            for arm in ARMS:
                assert (tables[arm][target]['present'], tables[arm][target]['absent']) == (counts['present'], counts['absent'])
            cells = [display(tables[arm][target]['dice']) for arm in ARMS] + [display(tables[arm][target]['FPA']) for arm in ARMS]
            lines.append('| ' + target.replace('|', ' / ') + f" | {counts['present']} | {counts['absent']} | " + ' | '.join(cells) + ' |')
    lines += ['', '所有次预算、逐类NA、条件区间及有效bootstrap次数见comparison.json。',
              'A同样消费4000任务，只能训练零输入cell投影；共享decoder被冻结，并非充分优化的强基线。',
              '保留原80对BF1守门失败和历史local代价，不能用本次相对守门洗掉。']
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], values=values, eligible_for_second_seed=eligible)), flush=True)


if __name__ == '__main__':
    main()
