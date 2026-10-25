"""Independently replay masked catalogs, task membership and actual prompt signs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit import FOLDS
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest, write


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, realized, base_tasks, remote, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(root/'manifest.json')
    original_contract = read(base_tasks/'contract.json')
    original_rows = pd.read_parquet(Path(original_contract['catalog'])/'tiles.parquet').to_dict('records')
    original_by_id = {r['tile_id']:r for r in original_rows}
    results = {}
    for fold, item in manifest.items():
        tasks = Path(item['tasks']); contract = read(tasks/'contract.json')
        require(contract['parent_protocol_sha256'] == digest(base_tasks/'contract.json'), 'parent protocol SHA')
        rows = pd.read_parquet(Path(contract['catalog'])/'tiles.parquet').to_dict('records')
        require(digest(Path(contract['catalog'])/'tiles.parquet') == contract['catalog_sha256'], 'catalog SHA')
        by_id = {r['tile_id']:r for r in rows}; labels = {}
        require(set(by_id) == set(original_by_id), 'all original fields preserved')
        for row in rows:
            changed_columns = {'labels', 'labels_array_sha256', 'counts_json', 'eligible_json', 'interior_counts_json'}
            for key, value in original_by_id[row['tile_id']].items():
                if key not in changed_columns:
                    require(row[key] == value or (pd.isna(row[key]) and pd.isna(value)), 'unchanged RGB/cells/spatial provenance: '+key)
            original = np.asarray(Image.open(row['heldout_original_labels']))
            require(hashlib.sha256(original.tobytes()).hexdigest() == row['heldout_original_labels_sha256'], 'original GT preserved')
            masked = np.asarray(Image.open(row['labels']))
            expected = np.where(np.isin(original, FOLDS[fold]['exclude'][row['dataset']]), 255, original)
            require(np.array_equal(masked, expected), 'exact exclusion and preserved seen labels')
            require(hashlib.sha256(masked.tobytes()).hexdigest() == row['labels_array_sha256'], 'masked GT SHA')
            labels[row['tile_id']] = masked
        recipes = [json.loads(line) for line in (tasks/'training_recipe.jsonl').open()]
        require(len(recipes) == contract['training_tasks'] == 40000, 'complete recipe')
        require(digest(tasks/'training_recipe.jsonl') == contract['training_recipe_sha256'], 'recipe SHA')
        counts = Counter()
        for index, plan in enumerate(recipes):
            row = by_id[plan['support_id']]
            require(plan['number'] == index and row['split'] == 'train', 'recipe order/split')
            require(plan['dataset'] == row['dataset'] and plan['query_id'] == plan['support_id'], 'local/domain')
            require(plan['target'] not in FOLDS[fold]['exclude'][row['dataset']], 'no excluded positive target')
            require(plan['target'] in json.loads(row['eligible_json']), 'eligible seen target')
            counts[plan['dataset']] += 1
        require(set(counts) == {'crc', 'tiger'}, 'both domains')
        val = read(tasks/'validation_maximum.json')
        require(digest(tasks/'validation_maximum.json') == contract['validation_sha256'], 'seen val SHA')
        path = realized/fold
        actual_summary = read(path/'summary.json')
        require(actual_summary['status'] == 'complete' and actual_summary['direct_replay_passed'], 'actual resolver replay')
        require(actual_summary['protocol_sha256'] == digest(tasks/'contract.json'), 'actual protocol binding')
        require(actual_summary['training_plans_sha256'] == digest(path/'training_plans.jsonl'), 'actual coordinates SHA')
        actual = [json.loads(line) for line in (path/'training_plans.jsonl').open()]
        require(len(actual) == actual_summary['tasks'] and len(actual) <= len(recipes), 'actual task count')
        for expected, plan in zip(recipes, actual):
            require(all(plan[k] == v for k,v in expected.items()), 'actual recipe prefix unchanged')
        for split, plans in [('val', val), ('train', actual)]:
            for plan in plans:
                row = by_id[plan['support_id']]; gt = labels[row['tile_id']]
                require(row['split'] == split and plan['target'] not in FOLDS[fold]['exclude'][row['dataset']], 'seen selection/training membership')
                for sign in ('positive', 'negative'):
                    xy = np.asarray(plan[sign+'_xy'])
                    require(np.isfinite(xy).all() and ((xy >= 0)&(xy < 1)).all(), 'normalized coordinates')
                    pixel = np.floor(xy*np.array(gt.shape[::-1])).astype(int)
                    values = gt[pixel[:,1],pixel[:,0]]
                    require((values != 255).all(), 'no ignored prompt')
                    require((values == plan['target']).all() if sign == 'positive' else (values != plan['target']).all(), 'actual prompt signs')
        tests = {}
        for name in ('heldout_local', 'heldout_remote'):
            folder = Path(item[name]); test_contract = read(folder/'contract.json')
            require(test_contract['checkpoint_selection_allowed'] is False, 'terminal scores only')
            require(test_contract['training_protocol_sha256'] == digest(tasks/'contract.json'), 'test/train binding')
            test_plans = read(folder/'validation_maximum.json')
            original_plans = read((base_tasks if name == 'heldout_local' else remote)/'validation_maximum.json')
            expected_plans = [p for p in original_plans if p['target'] in FOLDS[fold]['test'][p['dataset']]]
            require(test_plans == expected_plans, 'original frozen test task/point sequence')
            test_sha_key = 'validation_sha256' if name == 'heldout_local' else 'tasks_sha256'
            require(test_contract[test_sha_key] == digest(folder/'validation_maximum.json'), 'test task SHA')
            require(all(p['target'] in FOLDS[fold]['test'][p['dataset']] for p in test_plans), 'heldout target membership')
            require(test_contract['catalog'] != contract['catalog'], 'original test GT catalog separate')
            tests[name] = dict(tasks=len(test_plans), pairs=len({p['pair_id'] for p in test_plans}))
        results[fold] = dict(fields=len(rows), training_recipes=len(recipes), recipe_domains=dict(counts),
            seen_point_groups=len(val), actual_tasks=len(actual), complete_actual_protocol=actual_summary['complete_protocol'],
            actual_plans_sha256=actual_summary['training_plans_sha256'], tests=tests)
        print(json.dumps(dict(fold=fold, **results[fold])), flush=True)
    write(output/'summary.json', dict(status='passed_prepared_protocol_audit', worker_pid=os.getpid(),
        source_sha256=digest(__file__), manifest_sha256=digest(root/'manifest.json'), folds=results,
        scope='Exact GT masking, full recipe membership and available actual-point signs; no model training or accuracy'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'realized', 'base-tasks', 'remote', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.root, a.realized, a.base_tasks, a.remote, a.output)
