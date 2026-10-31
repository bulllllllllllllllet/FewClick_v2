"""Build fixed train-only novel episodes for the one P1I joint-adaptation study."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .cell_catalog import load_cell_catalog


BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
TASKS = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
WARM = BASE / 'image_pilot_1031_015107/phikon/pilot'
PARENT = BASE / 'task_diagnostic_run_1031_023222/mixed'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(plan):
    return json.dumps({key: plan[key] for key in
                       ['dataset', 'kind', 'support_id', 'query_id', 'target', 'positive_xy', 'negative_xy']}, sort_keys=True)


def overlap(x, y):
    return min(x[2], y[2]) > max(x[0], y[0]) and min(x[3], y[3]) > max(x[1], y[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    p0_path = BASE / 'p0_tissue_contract_1031_011514/contract.json'
    p0 = json.loads(p0_path.read_text())
    source = Path(p0['frozen_training']['original_path'])
    assert sha(source) == p0['frozen_training']['original_sha256']
    old = []
    for directory in [WARM, PARENT]:
        assert json.loads((directory / 'independent_acceptance.json').read_text())['status'] == 'accepted_fixed_image_pilot'
        lines = (directory / 'training_consumed.jsonl').read_text().splitlines()
        assert len(lines) == 4000
        old.extend(json.loads(s) for s in lines)
    parent_proof = json.loads((PARENT / 'independent_acceptance.json').read_text())
    assert parent_proof['initial_checkpoint_sha256'] == sha(WARM / 'checkpoint.pt')
    assert sha(WARM / 'training_consumed.jsonl') == p0['frozen_training']['sha256']
    old_keys = {episode_key(plan) for plan in old}
    catalog, inventory = load_cell_catalog(TASKS)
    pools = {dataset: sorted([r for r in inventory.values() if r['dataset'] == dataset and r['split'] == 'train'],
                              key=lambda r: r['tile_id']) for dataset in ['crc', 'tiger']}
    counts = {key: json.loads(row['counts_json']) for key, row in inventory.items()}
    boxes = {key: np.asarray(json.loads(row['multiscale_json'])['context_um'], float) for key, row in inventory.items()}
    selected, skips, seen = [], [], set(old_keys)
    total_source = 0
    with source.open() as stream:
        for source_index, line in enumerate(stream):
            total_source = source_index + 1
            if source_index < 4000 or len(selected) == 4000:
                continue
            original = json.loads(line)
            assert original['kind'] == 'local' and original['split'] == 'train'
            support = original['support_id']
            assert support == original['query_id'] and inventory[support]['split'] == 'train'
            slot = len(selected)
            plan = dict(original, source_training_line=source_index, training_recipe='P1I_novel_mixed4000_v1',
                        source_space=inventory[support]['space_key'], query_has_human_clicks=slot % 2 == 0)
            if slot % 2:
                present = slot % 4 == 1
                eligible = []
                for row in pools[original['dataset']]:
                    query = row['tile_id']
                    n = counts[query].get(str(original['target']), 0)
                    valid = sum(v for key, v in counts[query].items() if key != '255')
                    if query == support or valid < 64 or (n < 64 if present else n != 0):
                        continue
                    if row['space_key'] == inventory[support]['space_key'] and overlap(boxes[query], boxes[support]):
                        continue
                    eligible.append(row)
                if not eligible:
                    skips.append(dict(source_line=source_index, reason='no_eligible_disjoint_train_query'))
                    continue
                seed = int(hashlib.sha256(f'20261031|P1I|{source_index}|{slot}|{support}'.encode()).hexdigest()[:16], 16)
                query = eligible[seed % len(eligible)]
                kind = ('same_' if query['space_key'] == inventory[support]['space_key'] else 'cross_') + ('present' if present else 'absent')
                plan.update(protocol='transfer', query_id=query['tile_id'], kind=kind, query_organ=query['organ'],
                            query_presence='present' if present else 'absent',
                            query_GT_role='train episode stratification and loss only; never model input')
            plan['query_space'] = inventory[plan['query_id']]['space_key']
            key = episode_key(plan)
            if key in seen:
                skips.append(dict(source_line=source_index, reason='repeated_supervision_episode'))
                continue
            seen.add(key)
            selected.append(plan)
            if len(selected) % 200 == 0:
                (args.output / 'status.json').write_text(json.dumps(dict(status='preparing', worker_pid=os.getpid(), selected=len(selected), source_line=source_index)) + '\n')
    assert total_source == 40000
    if len(selected) != 4000:
        (args.output / 'failure.json').write_text(json.dumps(dict(status='insufficient_novel_train_episodes', selected=len(selected), skipped=len(skips))) + '\n')
        raise ValueError('No fallback to repeats, overlapping regions or validation')
    with (args.output / 'training4000.jsonl').open('w') as file:
        for plan in selected:
            file.write(json.dumps(plan, sort_keys=True) + '\n')
    with (args.output / 'skipped_sources.jsonl').open('w') as file:
        for row in skips:
            file.write(json.dumps(row) + '\n')
    inputs = [p0_path, source, WARM / 'training_consumed.jsonl', PARENT / 'training_consumed.jsonl',
              WARM / 'checkpoint.pt', PARENT / 'checkpoint.pt', catalog / 'tiles.parquet']
    result = dict(status='complete_novel_joint_tasks_unaccepted', worker_pid=os.getpid(), source_sha256=sha(__file__),
                  inputs_sha256={str(p): sha(p) for p in inputs}, original_training=str(source),
                  parent_warm=str(WARM), parent_mixed=str(PARENT), source_tasks=total_source,
                  parent_checkpoint_sha256=sha(PARENT / 'checkpoint.pt'), training_sha256=sha(args.output / 'training4000.jsonl'),
                  selected_tasks=4000, source_line_first=selected[0]['source_training_line'],
                  source_line_last=selected[-1]['source_training_line'], skipped_source_tasks=len(skips),
                  kinds=dict(Counter(p['kind'] for p in selected)),
                  dataset_targets=dict(Counter(p['dataset'] + '|' + str(p['target']) for p in selected)),
                  novel_relative_to_actual_parent_consumption=True, validation_GT_read=False,
                  qualification='new episode tuples within historically exposed train pool; not new independent data')
    (args.output / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'status.json').write_text(json.dumps(dict(status='complete_unaccepted', worker_pid=os.getpid(), selected=4000)) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
