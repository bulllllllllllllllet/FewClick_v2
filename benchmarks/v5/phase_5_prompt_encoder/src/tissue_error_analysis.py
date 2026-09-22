"""Aggregations for paired tissue-prompt error analysis."""
from collections import defaultdict
import numpy as np


def aggregate_task_errors(rows):
    """Aggregate pixel counts by task and report Dice, recall, FPR, and FNR."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['task_id'])].append(row)
    result = {}
    for task, values in sorted(groups.items(), key=lambda x: int(x[0])):
        tp = sum(int(x['tp']) for x in values)
        fp = sum(int(x['fp']) for x in values)
        fn = sum(int(x['fn']) for x in values)
        valid = sum(int(x['valid_pixels']) for x in values)
        target = tp + fn
        result[task] = dict(episodes=len(values), tp=tp, fp=fp, fn=fn, valid_pixels=valid,
            dice=2*tp/(2*tp+fp+fn) if target else None,
            recall=tp/target if target else None,
            false_negative_rate=fn/target if target else None,
            false_positive_rate=fp/(valid-target) if valid>target else None)
    return result


def paired_delta(local, cross):
    """Match task rows, preserving absent fields rather than inventing values."""
    result = {}
    for task in sorted(set(local) & set(cross), key=int):
        left, right = local[task], cross[task]
        result[task] = dict(local=left, cross=right,
            dice_delta=None if left['dice'] is None or right['dice'] is None else right['dice']-left['dice'],
            fnr_delta=None if left['false_negative_rate'] is None or right['false_negative_rate'] is None else right['false_negative_rate']-left['false_negative_rate'],
            fpr_delta=None if left['false_positive_rate'] is None or right['false_positive_rate'] is None else right['false_positive_rate']-left['false_positive_rate'])
    return result


def seed_task_mean(rows):
    """Mean per-seed Dice across seed-specific pixel aggregates, no pseudo-N."""
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row['seed'])].append(row)
    return {seed: aggregate_task_errors(items) for seed, items in by_seed.items()}
