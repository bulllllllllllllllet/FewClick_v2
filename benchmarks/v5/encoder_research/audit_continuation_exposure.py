"""Audit repeated supervision, optimizer policy and starting-point regressions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .accept_continuous_pilot import summarize
from .continuous_data import sha
from .continuous_prompt import ARMS


def episode_key(plan):
    return json.dumps({key: plan[key] for key in
                       ['dataset', 'kind', 'support_id', 'query_id', 'target', 'positive_xy', 'negative_xy']}, sort_keys=True)


def task_inventory(plans):
    remote = [p for p in plans if p['support_id'] != p['query_id']]
    return dict(tasks=len(plans), unique_episodes=len({episode_key(p) for p in plans}),
                query_images=len({p['query_id'] for p in plans}), support_images=len({p['support_id'] for p in plans}),
                kinds=dict(Counter(p['kind'] for p in plans)),
                dataset_targets=dict(Counter(p['dataset'] + '|' + str(p['target']) for p in plans)),
                positive_click_counts=dict(Counter(str(len(p['positive_xy'])) for p in plans)),
                remote_tasks=len(remote), remote_same_space=sum(p['kind'].startswith('same_') for p in remote),
                remote_cross_space=sum(p['kind'].startswith('cross_') for p in remote))


def strata(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['budget'] == '3+2':
            groups[row['scope'] + '|' + row['kind']].append(row)
    return {key: summarize(value) for key, value in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-transfer-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.run / 'contract.json').read_text())
    assert (args.run / 'exit_code').read_text() == '0'
    training_source = 'benchmarks/v5/encoder_research/train_continuous_pilot.py'
    assert sha(training_source) == contract['sources'][training_source]
    parent = Path(contract['parent'])
    parent_directory = parent.parent
    old_receipt = json.loads((parent_directory / 'independent_acceptance.json').read_text())
    remote_receipt = json.loads((args.parent_transfer_reference / 'independent_acceptance.json').read_text())
    assert old_receipt['status'] == 'accepted_fixed_image_pilot'
    assert remote_receipt['status'] == 'accepted_full935_transfer'
    assert sha(parent) == remote_receipt['checkpoint_sha256'] == contract['parent_sha256']
    parent_lines = (parent_directory / 'training_consumed.jsonl').read_text().splitlines(keepends=True)
    current_lines = Path(contract['training']).read_text().splitlines(keepends=True)
    assert len(parent_lines) == len(current_lines) == 4000
    assert sha(parent_directory / 'training_consumed.jsonl') == contract['training_sha256']
    assert parent_lines == current_lines
    old_plans, plans = [json.loads(s) for s in parent_lines], [json.loads(s) for s in current_lines]
    old_keys = {episode_key(p) for p in old_plans}
    duplicate_count = sum(episode_key(p) in old_keys for p in plans)
    local = [r for r in json.loads((parent_directory / 'rows.json').read_text()) if r['scope'] == 'local']
    remote = [dict(r, scope='remote') for r in json.loads((args.parent_transfer_reference / 'rows.json').read_text())]
    parent_rows = local + remote
    assert len(parent_rows) == 3711
    parent_by_stratum = strata(parent_rows)
    arms, changed_rows, evidence = {}, {}, {}
    identity = ['scope', 'budget', 'dataset', 'target', 'support_id', 'query_id', 'kind', 'positive', 'valid']
    for arm in ARMS:
        directory = args.run / arm
        accepted = json.loads((directory / 'independent_acceptance.json').read_text())
        assert accepted['status'] == 'accepted_continuous_pilot' and accepted['rows_replayed'] == 3711
        assert (directory / 'training_consumed.jsonl').read_text().splitlines(keepends=True) == parent_lines
        checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
        assert checkpoint['updates'] == 3000 and checkpoint['cell_updates'] == 1000
        groups = checkpoint['optimizer']['param_groups']
        rows = json.loads((directory / 'rows.json').read_text())
        assert len(rows) == len(parent_rows)
        for old, new in zip(parent_rows, rows):
            assert all(old[k] == new[k] for k in identity)
        by_stratum = strata(rows)
        changes = {}
        for key, old in parent_by_stratum.items():
            new = by_stratum[key]
            assert old['rows'] == new['rows']
            changes[key] = {metric: None if old[metric] is None else new[metric] - old[metric]
                            for metric in ['dice', 'BF1', 'absent_FPA', 'class_macro_Dice']}
        arms[arm] = dict(optimizer_groups=[{key: g[key] for key in ['lr', 'weight_decay', 'betas', 'eps']} for g in groups],
                         shared_steps=sorted({int(checkpoint['optimizer']['state'][p]['step']) for p in groups[0]['params']}),
                         cell_steps=sorted({int(checkpoint['optimizer']['state'][p]['step']) for p in groups[1]['params']}),
                         strata=by_stratum, change_from_parent=changes)
        changed_rows[arm] = len(rows)
        evidence[arm] = dict(acceptance_sha256=sha(directory / 'independent_acceptance.json'),
                             checkpoint_sha256=sha(directory / 'checkpoint.pt'), rows_sha256=sha(directory / 'rows.json'))
    parent_state = torch.load(parent, map_location='cpu', weights_only=True)
    parent_rates = [g['lr'] for g in parent_state['optimizer']['param_groups']]
    assert parent_rates == [.001] and all(v['optimizer_groups'][0]['lr'] == .001 for v in arms.values())
    primary_remote = [r for r in parent_rows if r['scope'] == 'remote' and r['budget'] == '3+2']
    training_inventory = task_inventory(plans)
    evaluation_inventory = dict(pairs=len(primary_remote), kinds=dict(Counter(r['kind'] for r in primary_remote)),
                                remote_same_space=sum(r['kind'].startswith('same_') for r in primary_remote),
                                remote_cross_space=sum(r['kind'].startswith('cross_') for r in primary_remote))
    result = dict(status='accepted_continuation_exposure_audit', worker_pid=os.getpid(), source_sha256=sha(__file__),
                  seed=contract['seed'], parent_checkpoint_sha256=contract['parent_sha256'],
                  training_source_sha256=contract['sources'][training_source],
                  parent_consumed_sha256=sha(parent_directory / 'training_consumed.jsonl'),
                  continuation_consumed_sha256=contract['training_sha256'], exact_repeated_tasks=duplicate_count,
                  novel_supervision_episodes=sum(episode_key(p) not in old_keys for p in plans),
                  training=training_inventory, remote_evaluation=evaluation_inventory, parent_strata=parent_by_stratum,
                  arms=arms, evidence=evidence, fixed_start_and_endpoint_only=True, training_updates=0,
                  conclusions=['additional1000 updates replay exactly the same4000 supervision tasks in the same order',
                               'shared decoder continues at the same0.001 learning rate; no scheduler state stored or LR decay applied by the frozen source',
                               'same/cross space-key task mixture differs between train and transfer evaluation; not a patient-disjoint split issue',
                               'these are verified risk factors, not a causal proof of overfitting or the LR being wrong',
                               'new cell parameters start at0 optimizer steps; repeated supervision does not mean the new cell branch was previously trained',
                               'retain fixed matched-control guards; parent2000 is a descriptive reference with a different update budget'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 继续训练回落：监督暴露与优化器审计', '',
             f"seed{contract['seed']}：新增4000/4000任务与父模型mixed阶段逐行相同，无新增监督episode，三臂完全匹配。", '',
             'shared AdamW维持lr0.001，新细胞组从0步训练1000步；这不是“细胞参数已经训练过4000任务”。', '',
             '| 来源 | same space-key remote | cross space-key remote |', '|---|---:|---:|',
             f"| train | {training_inventory['remote_same_space']} | {training_inventory['remote_cross_space']} |",
             f"| eval935 | {evaluation_inventory['remote_same_space']} | {evaluation_inventory['remote_cross_space']} |", '',
             '| scope / kind | n | A Dice变化pp | XCell Dice变化pp | LEMON Dice变化pp |', '|---|---:|---:|---:|---:|']
    for key, values in parent_by_stratum.items():
        columns = [arms[arm]['change_from_parent'][key]['dice'] for arm in ARMS]
        lines.append('| ' + key.replace('|', ' / ') + f" | {values['rows']} | " + ' | '.join('NA' if v is None else f'{v * 100:+.3f}' for v in columns) + ' |')
    lines += ['', '来源分布与重复监督是风险线索；本审计没有训练新模型、挑阈值、删除类别或确认过拟合原因。',
              '完整BF1/FPA变化、逐类n/NA及输入SHA见audit.json。']
    (args.output / 'audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], exact_repeated_tasks=duplicate_count,
                         train_same_cross=[training_inventory['remote_same_space'], training_inventory['remote_cross_space']],
                         eval_same_cross=[evaluation_inventory['remote_same_space'], evaluation_inventory['remote_cross_space']])), flush=True)


if __name__ == '__main__':
    main()
