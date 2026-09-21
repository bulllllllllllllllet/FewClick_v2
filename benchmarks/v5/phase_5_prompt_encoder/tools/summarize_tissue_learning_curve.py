#!/usr/bin/env python3
"""Read-only summary and optional saved-mask verification of learning curves."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import pixel_metrics, task_macro


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['scope'] == 'unprompted':
            groups[(row['method'], row['step'], row['seed'])].append(row)
    per_method = defaultdict(list)
    for (method, step, seed), values in groups.items():
        metrics = task_macro(values)
        per_method[(method, step)].append(dict(seed=seed, dice=metrics['task_macro_dice'], iou=metrics['task_macro_iou']))
    return [dict(method=method, step=step, seeds=values,
                 dice_mean=float(np.mean([v['dice'] for v in values])),
                 dice_seed_sd=float(np.std([v['dice'] for v in values], ddof=1)) if len(values)>1 else None,
                 iou_mean=float(np.mean([v['iou'] for v in values])))
            for (method, step), values in sorted(per_method.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--verify-masks', action='store_true')
    args = p.parse_args()
    rows = json.loads((args.run/'episode_metrics.json').read_text())
    if args.verify_masks:
        groups = defaultdict(list)
        for row in rows:
            groups[row['episode_id']].append(row)
        for episode, values in groups.items():
            with np.load(args.run/f'val_{episode:04d}.npz') as masks:
                for row in values:
                    valid = masks['valid'] if row['scope']=='all' else masks['valid'] & ~masks['prompt_excluded']
                    key = f"{row['method']}_{row['seed']}_step{row['step']}"
                    actual = pixel_metrics(masks[key], masks['gt'], valid)
                    if any(row[k] != value for k,value in actual.items()):
                        raise ValueError(f'metric mismatch: {episode}, {key}')
        print(json.dumps(dict(verified_metric_rows=len(rows))))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == '__main__':
    main()
