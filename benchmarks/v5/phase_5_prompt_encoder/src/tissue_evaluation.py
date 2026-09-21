"""Binary pixel metrics and support-only prototype baselines for tissue tasks."""
from collections import defaultdict

import numpy as np
import torch

from .multi_prototype import PositiveNegativePrototypeMatcher


def pixel_metrics(prediction, target, valid):
    prediction, target, valid = (np.asarray(x) for x in (prediction, target, valid))
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('prediction, target and valid must share shape')
    if any(x.dtype != np.bool_ for x in (prediction, target, valid)):
        raise ValueError('explicit boolean masks required')
    p, y = prediction[valid], target[valid]
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    # Empty targets are evaluated separately; empty-empty cannot inflate macro Dice.
    present = bool(y.any())
    return dict(tp=tp, fp=fp, fn=fn, valid_pixels=len(y), target_present=present,
                dice=2*tp/(2*tp+fp+fn) if present else None,
                iou=tp/(tp+fp+fn) if present else None,
                false_positive_area=fp/len(y) if len(y) else None)


def task_macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['task_id'])].append(row)
    by_task = {}
    for task, values in sorted(groups.items()):
        by_task[task] = {}
        for key in ('dice', 'iou'):
            scores = [v[key] for v in values if v[key] is not None]
            by_task[task][key] = float(np.mean(scores)) if scores else None
        by_task[task]['episodes'] = len(values)
    result = {'by_task': by_task}
    for key in ('dice', 'iou'):
        values = [v[key] for v in by_task.values() if v[key] is not None]
        result['task_macro_'+key] = float(np.mean(values)) if values else None
    return result


def prototype_logits(batch, mode, temperature=.1):
    """Use only provided support tokens. No query labels or query clustering."""
    if mode not in ('single', 'multi'):
        raise ValueError('mode must be single or multi')
    sets = []
    for sign in ('positive', 'negative'):
        values, valid = batch[sign+'_tokens'], batch[sign+'_mask']
        if not valid.any(-1).all():
            raise ValueError('both signs require support')
        if mode == 'single':
            values = (values*valid[..., None]).sum(1, keepdim=True)/valid.sum(1)[:, None, None]
            valid = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        sets.extend((values, valid))
    matcher = PositiveNegativePrototypeMatcher(temperature).to(batch['fine_tokens'].device)
    return matcher(batch['fine_tokens'], *sets)['logits']
