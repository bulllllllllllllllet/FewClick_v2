"""Check novel P1I episodes against original points, actual labels and footprints."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .build_joint_task_manifest import TASKS, episode_key, overlap, sha
from .cell_catalog import load_cell_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.manifest / 'contract.json').read_text())
    assert report['status'] == 'complete_novel_joint_tasks_unaccepted'
    assert report['source_sha256'] == sha(Path(__file__).with_name('build_joint_task_manifest.py'))
    for path, value in report['inputs_sha256'].items():
        assert sha(path) == value
    plans = [json.loads(s) for s in (args.manifest / 'training4000.jsonl').read_text().splitlines()]
    assert len(plans) == 4000 and sha(args.manifest / 'training4000.jsonl') == report['training_sha256']
    indices = [p['source_training_line'] for p in plans]
    assert indices == sorted(set(indices)) and min(indices) >= 4000 and max(indices) < 40000
    wanted, originals = set(indices), {}
    with Path(report['original_training']).open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                originals[index] = json.loads(line)
    old = []
    for directory in [Path(report['parent_warm']), Path(report['parent_mixed'])]:
        old.extend(json.loads(s) for s in (directory / 'training_consumed.jsonl').read_text().splitlines())
    old_keys = {episode_key(p) for p in old}
    keys = [episode_key(p) for p in plans]
    assert len(set(keys)) == 4000 and not set(keys) & old_keys
    _, inventory = load_cell_catalog(TASKS)
    visited_labels = set()
    @lru_cache(maxsize=64)
    def labels(identity):
        row = inventory[identity]
        assert row['split'] == 'train'
        visited_labels.add(identity)
        with Image.open(row['labels']) as im:
            value = np.asarray(im).copy()
        assert hashlib.sha256(value.tobytes()).hexdigest() == row['labels_array_sha256']
        assert value.shape == (512, 512)
        return value
    classes = Counter()
    remote_present = remote_absent = local_count = 0
    for index, plan in enumerate(plans):
        original = originals[plan['source_training_line']]
        for key in ['support_id', 'dataset', 'target', 'positive_xy', 'negative_xy', 'click_audit']:
            assert plan[key] == original[key]
        support, query = inventory[plan['support_id']], inventory[plan['query_id']]
        assert support['split'] == query['split'] == plan['split'] == 'train'
        assert support['dataset'] == query['dataset'] == plan['dataset']
        truth = labels(plan['support_id'])
        for sign in ['positive', 'negative']:
            xy = np.asarray(plan[sign + '_xy'])
            assert len(xy) and ((xy > 0) & (xy < 1)).all()
            pixel = np.floor(xy * np.array(truth.shape[::-1])).astype(int)
            value = truth[pixel[:, 1], pixel[:, 0]]
            assert (value == plan['target']).all() if sign == 'positive' else ((value != plan['target']) & (value != 255)).all()
        gt = labels(plan['query_id'])
        positive, valid = int((gt == plan['target']).sum()), int((gt != 255).sum())
        assert valid >= 64
        if index % 2 == 0:
            assert plan['kind'] == 'local' and plan['support_id'] == plan['query_id']
            local_count += 1
        else:
            assert plan['support_id'] != plan['query_id'] and not plan['query_has_human_clicks']
            present = index % 4 == 1
            assert positive >= 64 if present else positive == 0
            assert plan['kind'].endswith('present' if present else 'absent')
            if support['space_key'] == query['space_key']:
                a = json.loads(support['multiscale_json'])['context_um']
                b = json.loads(query['multiscale_json'])['context_um']
                assert not overlap(a, b) and plan['kind'].startswith('same_')
            else:
                assert plan['kind'].startswith('cross_')
            remote_present += int(present)
            remote_absent += int(not present)
        classes[plan['dataset'] + '|' + str(plan['target'])] += 1
        if (index + 1) % 200 == 0:
            (args.manifest / 'accept_status.json').write_text(json.dumps(dict(status='replaying_train_GT', worker_pid=os.getpid(), tasks=index + 1)) + '\n')
    assert (local_count, remote_present, remote_absent) == (2000, 1000, 1000)
    assert classes['crc|10'] > 0
    result = dict(status='accepted_novel_train_only4000_joint_tasks', worker_pid=os.getpid(), tasks_replayed=4000,
                  local=local_count, remote_present=remote_present, remote_absent=remote_absent,
                  source_support_clicks_exact=True, actual_GT_signs_and_query_presence_checked=True,
                  remote_footprints_checked=2000, novel_episode_count=4000, unique_queried_train_GT_files=len(visited_labels),
                  dataset_targets=dict(classes), training_sha256=report['training_sha256'],
                  contract_sha256=sha(args.manifest / 'contract.json'), source_sha256=sha(__file__),
                  validation_GT_read=False, patient_disjoint_split_imposed=False,
                  scope='train task manifest only; no model update or independent confirmation claim')
    (args.manifest / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.manifest / 'accept_status.json').write_text(json.dumps(dict(status='complete', worker_pid=os.getpid(), tasks=4000)) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
