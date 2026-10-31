"""Combine accepted independent seeds without pooling tasks as independent samples."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .continuous_data import sha
from .continuous_prompt import ARMS


def read_run(run, comparison):
    assert (run / 'exit_code').read_text() == '0'
    contract = json.loads((run / 'contract.json').read_text())
    result = json.loads(comparison.read_text())
    assert result['status'] == 'accepted_preregistered_continuous_cell_comparison'
    assert result['contract_sha256'] == sha(run / 'contract.json')
    assert result['source_sha256'] == sha(Path(__file__).with_name('compare_continuous_pilots.py'))
    for arm in ARMS:
        for name, expected in result['evidence'][arm].items():
            assert sha(run / arm / name) == expected
        receipt = json.loads((run / arm / 'independent_acceptance.json').read_text())
        report = json.loads((run / arm / 'report.json').read_text())
        assert receipt['status'] == 'accepted_continuous_pilot' and receipt['rows_replayed'] == 3711
        assert report['seed'] == contract['seed'] and report['total_updates'] == 3000
    return contract, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['first-run', 'first-comparison', 'second-run', 'second-comparison', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    first, a = read_run(args.first_run, args.first_comparison)
    second, b = read_run(args.second_run, args.second_comparison)
    assert (first['seed'], second['seed']) == (20261031, 20261101)
    assert first['parent_sha256'] != second['parent_sha256']
    for key in ['training_sha256', 'images', 'grids', 'guard', 'comparison_plan', 'updates_per_arm', 'tasks_per_update']:
        assert first[key] == second[key]
    for arm in ARMS:
        left = json.loads((args.first_run / arm / 'independent_acceptance.json').read_text())
        right = json.loads((args.second_run / arm / 'independent_acceptance.json').read_text())
        assert left['inputs'] == right['inputs']
        assert left['sources'] == right['sources']
        assert left['acceptance_source_sha256'] == right['acceptance_source_sha256']
        assert left['initial_identity']['start_model_sha256'] != right['initial_identity']['start_model_sha256']
    replication = {}
    for name in first['comparison_plan']['pairs']:
        changes = [result['comparisons'][name] for result in [a, b]]
        primary = [r['candidate_minus_reference']['remote_macro_Dice'] for r in changes]
        replication[name] = dict(primary_changes=primary, positive_direction_in_both=all(v > 0 for v in primary),
                                 resource_guard_pass_by_seed=[r['pass_all'] for r in changes],
                                 all_guards_pass_both=all(r['pass_all'] for r in changes),
                                 failed_tests_by_seed=[[k for k, v in r['tests'].items() if not v] for r in changes],
                                 primary_mean=float(np.mean(primary)), primary_sample_std=float(np.std(primary, ddof=1)))
    parent_comparison = {}
    for arm in ARMS:
        changes = [result['values'][arm]['remote_macro_Dice'] - result['parent_context']['remote_macro_Dice'] for result in [a, b]]
        parent_comparison[arm] = dict(changes=changes, improves_parent_in_both=all(v > 0 for v in changes),
                                     qualification='parent has2000 updates; descriptive, never replaces matched3000 control')
    result = dict(status='accepted_two_seed_continuous_cell_summary', worker_pid=os.getpid(), source_sha256=sha(__file__),
                  seeds=[first['seed'], second['seed']], comparisons_sha256=[sha(args.first_comparison), sha(args.second_comparison)],
                  by_seed={str(first['seed']): a['values'], str(second['seed']): b['values']},
                  parent_by_seed={str(first['seed']): a['parent_context'], str(second['seed']): b['parent_context']},
                  replication=replication, descriptive_parent_changes=parent_comparison, model_count=6, masks_replayed=22266,
                  original80_guards_unchanged=True, fullWSI_promoted=False, P1_complete=False,
                  limitations=['two seeds share the same development samples; no increased cohort coverage',
                               'sample std across2 training seeds is descriptive, not a population confidence interval',
                               'per-seed query-space intervals are conditional on fixed support banks',
                               'joint image adaptation, stain sensitivity, vocabulary/spatial/WSI/independent confirmation remain unfinished'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 连续细胞两独立seed复验', '',
             '同训练任务与输入、不同独立父模型。两seed共享开发集，不把22266个mask当独立样本。', '',
             '| seed | arm | local Dice | remote macroDice | remote BF1 | absent FPA | fat Dice |',
             '|---|---|---:|---:|---:|---:|---:|']
    for seed, values in result['by_seed'].items():
        for arm, value in values.items():
            columns = ['local_Dice', 'remote_macro_Dice', 'remote_BF1', 'absent_FPA', 'fat_Dice']
            lines.append('| ' + seed + ' | ' + arm + ' | ' + ' | '.join(f'{value[k]:.4f}' for k in columns) + ' |')
    lines += ['', '| comparison | seed0变化pp | seed1变化pp | 两seed均过原守门 |', '|---|---:|---:|---|']
    for key, value in replication.items():
        x, y = value['primary_changes']
        lines.append(f"| {key} | {x * 100:+.3f} | {y * 100:+.3f} | {value['all_guards_pass_both']} |")
    lines += ['', '父2000步参考、局部边界代价及逐项失败见summary.json和各seed完整逐类报告。',
              'P1尚未完成；不据此宣称LEMON独有优势、全片提升或独立泛化。']
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], replication=replication, descriptive_parent_changes=parent_comparison)), flush=True)


if __name__ == '__main__':
    main()
