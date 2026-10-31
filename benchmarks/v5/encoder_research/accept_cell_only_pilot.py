"""Audit the cell-only policy, invariant common state and all saved masks."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from .continuous_data import ContinuousData, sha
from .continuous_prompt import ARMS, ContinuousPromptDecoder, fresh_optimizer, tree_sha
from .train_cell_only_pilot import gate_indices, source_identity
from .cell_only_checks import freeze_shared, assert_shared_frozen, ParentMaskReference
from benchmarks.v5.fair_comparison.metrics import score


def summarize(rows):
    present = [r for r in rows if r['positive'] > 0]
    absent = [r for r in rows if r['positive'] == 0]
    def mean(values):
        return float(np.mean(values)) if values else None
    targets = defaultdict(list)
    for row in rows:
        targets[row['dataset'] + '|' + str(row['target'])].append(row)
    per_target = {key: dict(present=sum(r['positive'] > 0 for r in values),
                            absent=sum(r['positive'] == 0 for r in values),
                            dice=mean([r['dice'] for r in values if r['positive'] > 0]),
                            FPA=mean([r['false_positive_fraction'] for r in values if r['positive'] == 0]))
                  for key, values in targets.items()}
    return dict(rows=len(rows), present=len(present), absent=len(absent),
                dice=mean([r['dice'] for r in present]),
                BF1=mean([r['bf1_um2'] for r in present if r['bf1_um2'] is not None]),
                absent_FPA=mean([r['false_positive_fraction'] for r in absent]),
                class_macro_Dice=mean([v['dice'] for v in per_target.values() if v['dice'] is not None]),
                per_target=per_target)


def accept_pilot(directory):
    report = json.loads((directory / 'report.json').read_text())
    expected_status = 'complete_cell_only_gate_unaccepted' if report['gate'] else 'complete_cell_only_pilot_unaccepted'
    assert report['status'] == expected_status and report['sources'] == source_identity()
    assert sha(report['parent']) == report['parent_sha256']
    assert sha(report['training_path']) == report['training_sha256']
    lines = Path(report['training_path']).read_text().splitlines(keepends=True)
    assert len(lines) == 4000 and gate_indices(lines) == report['gate_indices']
    expected_lines = [lines[i] for i in report['gate_indices']] if report['gate'] else lines
    assert (directory / 'training_consumed.jsonl').read_text() == ''.join(expected_lines)
    assert sha(directory / 'training_consumed.jsonl') == report['consumed_sha256']
    updates = 2 if report['gate'] else 1000
    assert report['training_updates'] == updates and report['consumed_tasks'] == updates * 4
    assert sha(directory / 'checkpoint.pt') == report['checkpoint_sha256']
    parent = torch.load(report['parent'], map_location='cpu', weights_only=True)
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['updates'] == report['total_updates'] == 2000 + updates
    assert checkpoint['cell_updates'] == updates and checkpoint['arm'] == report['arm']
    assert checkpoint['seed'] == report['seed'] == parent.get('seed', 20261031)
    assert tree_sha(parent['model']) == report['initial_identity']['parent_model_sha256']
    assert tree_sha(parent['optimizer']) == report['initial_identity']['parent_optimizer_sha256']
    initial = dict(parent['model'], **{'cell_project.weight': torch.zeros(1024, 75, 1, 1)})
    assert tree_sha(initial) == report['initial_identity']['start_model_sha256']
    assert checkpoint['shared_updates'] == report['shared_updates'] == 2000
    assert checkpoint['update_policy'] == report['update_policy'] == 'cell_only_frozen_shared'
    assert report['shared_weights_and_optimizer_exact']
    model = ContinuousPromptDecoder(report['arm'])
    model.load_state_dict(checkpoint['model'], strict=True)
    freeze_shared(model)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == report['initial_identity']['trainable_parameters'] == 76800
    optimizer = fresh_optimizer(model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    assert tree_sha(optimizer.state_dict()) == tree_sha(checkpoint['optimizer'])
    for parameter in optimizer.param_groups[0]['params']:
        assert int(optimizer.state[parameter]['step']) == 2000
    assert_shared_frozen(model, optimizer, parent)
    assert len(optimizer.param_groups[1]['params']) == 1
    assert int(optimizer.state[model.cell_project.weight]['step']) == updates
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    changed = bool(torch.count_nonzero(model.cell_project.weight))
    assert changed == report['cell_parameters_changed'] == (report['arm'] != 'image_only')
    assert report['initial_identity']['head_parameters'] == sum(v.numel() for v in model.parameters()) == 508705
    assert report['initial_parent_logits_exact'] and report['exact_disk_reload'] and report['exact_optimizer_reload']
    assert report['perturbation_max_abs'] == 0 if report['arm'] == 'image_only' else report['perturbation_max_abs'] > 0
    if report['gate']:
        assert report['exact_optimizer_continuation']
    else:
        gate = json.loads(Path(report['gate_acceptance_path']).read_text())
        assert sha(report['gate_acceptance_path']) == report['gate_acceptance_sha256']
        assert gate['status'] == 'accepted_all_three_cell_only_gates' and gate['sources'] == report['sources']
        assert gate['acceptance_source_sha256'] == sha(__file__)
    data = ContinuousData(report['images'], report['grids'])
    assert data.identity == report['inputs']
    expected = [('gate_train', 'recorded', json.loads(line)) for line in expected_lines] if report['gate'] and report['arm'] != 'image_only' else list(data.validation())
    parent_reference = ParentMaskReference(report['parent'], report['parent_transfer_reference']) if report['arm'] == 'image_only' else None
    if parent_reference is not None:
        assert report['parent_reference_identity'] == parent_reference.identity
        assert report['parent_masks_checked'] == 3711
    rows = json.loads((directory / 'rows.json').read_text())
    assert len(rows) == len(expected) == report['scoring_rows'] == (8 if report['gate'] and report['arm'] != 'image_only' else 3711)
    with np.load(directory / 'packed_masks.npz', allow_pickle=False) as file:
        masks = file['masks'].copy()
    assert masks.shape == (len(rows), 32768)
    for index, (row, (scope, budget, plan)) in enumerate(zip(rows, expected)):
        assert row['scope'] == scope and row['budget'] == budget and row['mask_index'] == index
        for key in ['dataset', 'target', 'support_id', 'query_id', 'kind']:
            assert row[key] == plan[key]
        assert row['pair_id'] == plan.get('pair_id')
        if parent_reference is not None:
            parent_reference.assert_equal(row, masks[index])
        gt, mpp = data.labels(plan['query_id'])
        predicted = np.unpackbits(masks[index], count=512 * 512).reshape(512, 512).astype(bool)
        metrics = score(predicted, gt == plan['target'], gt != 255, mpp)
        for key, value in metrics.items():
            if value is None:
                assert row[key] is None
            elif isinstance(value, (int, bool)):
                assert row[key] == value
            else:
                assert abs(row[key] - value) < 1e-12
        if (index + 1) % 200 == 0:
            (directory / 'accept_status.json').write_text(json.dumps(dict(status='replaying', worker_pid=os.getpid(), masks=index + 1)) + '\n')
    groups = defaultdict(list)
    for row in rows:
        groups[row['scope'] + '|' + row['budget']].append(row)
    local = groups.get('local|3+2', [])
    small = [r for r in local if r['positive'] / r['valid'] < .05]
    fat = [r for r in local if r['dataset'] == 'crc' and r['target'] == 10]
    result = dict(status='accepted_cell_only_gate' if report['gate'] else 'accepted_cell_only_pilot',
                  worker_pid=os.getpid(), arm=report['arm'], rows_replayed=len(rows), all_metrics_replayed=True,
                  summary={key: summarize(value) for key, value in groups.items()},
                  local_strata={'small': summarize(small), 'fat': summarize(fat)},
                  sources=report['sources'], parent_sha256=report['parent_sha256'], training_sha256=report['training_sha256'],
                  inputs=report['inputs'], initial_identity=report['initial_identity'], checkpoint_sha256=report['checkpoint_sha256'],
                  shared_weights_and_optimizer_exact=True, shared_updates=2000, cell_updates=updates,
                  update_policy='cell_only_frozen_shared', parent_masks_checked=report['parent_masks_checked'],
                  gate_indices=report['gate_indices'], acceptance_source_sha256=sha(__file__),
                  quality_outcome='not_compared_yet', scope='development only; original80 guards unchanged')
    (directory / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    (directory / 'accept_status.json').write_text(json.dumps(dict(status='complete', worker_pid=os.getpid(), masks=len(rows))) + '\n')
    print(json.dumps(dict(status=result['status'], arm=report['arm'], rows_replayed=len(rows))), flush=True)


def aggregate_gates(root):
    rows = {arm: json.loads((root / arm / 'independent_acceptance.json').read_text()) for arm in ARMS}
    reference = rows['image_only']
    for arm, row in rows.items():
        assert row['status'] == 'accepted_cell_only_gate' and row['arm'] == arm
        for key in ['sources', 'parent_sha256', 'training_sha256', 'inputs', 'initial_identity', 'gate_indices']:
            assert row[key] == reference[key]
        assert row['acceptance_source_sha256'] == sha(__file__)
    result = {key: reference[key] for key in ['sources', 'parent_sha256', 'training_sha256', 'inputs', 'initial_identity', 'gate_indices']}
    result.update(status='accepted_all_three_cell_only_gates', worker_pid=os.getpid(), acceptance_source_sha256=sha(__file__),
                  gate_receipts={arm: sha(root / arm / 'independent_acceptance.json') for arm in ARMS}, quality_claim=False)
    (root / 'independent_gate_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', type=Path)
    group.add_argument('--gate-root', type=Path)
    arguments = parser.parse_args()
    if arguments.pilot:
        accept_pilot(arguments.pilot)
    else:
        aggregate_gates(arguments.gate_root)
