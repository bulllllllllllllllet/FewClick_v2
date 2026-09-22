"""Explicit common feature contract, fixed spatial sampling and pixel clicks."""
from collections import defaultdict
import hashlib
import json
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def common_cells(cells):
    if cells.ndim != 2 or cells.shape[1] != 74 or not torch.isfinite(cells).all():
        raise ValueError('finite [N,74] cells required')
    result = cells.clone()
    result[:, 2:10] = 0  # unavailable area/category; never encode missing as category 0
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def spread(rows, count, keys, seed):
    """Round-robin across explicit strata, without replacing any row."""
    if len(rows) < count or count < 1:
        raise ValueError(f'requested {count} but only {len(rows)} candidates')
    rng = np.random.default_rng(seed)

    def interleave(values, remaining_keys):
        if not remaining_keys:
            ordered = sorted(values, key=lambda r: str(r.get('sample_id', r.get('patch_id'))))
            for i in rng.permutation(len(ordered)):
                yield ordered[i]
            return
        groups = defaultdict(list)
        for row in values:
            groups[str(row[remaining_keys[0]])].append(row)
        queues = [iter(interleave(groups[k], remaining_keys[1:])) for k in sorted(groups)]
        while queues:
            active = []
            for queue in queues:
                value = next(queue, None)
                if value is not None:
                    yield value
                    active.append(queue)
            queues = active

    iterator = interleave(rows, keys)
    return [next(iterator) for _ in range(count)]



def overlaps(a, b, buffer=128):
    if str(a['wsi_id']) != str(b['wsi_id']):
        return False
    x,y,w,h = [float(a[k]) for k in ('x_level0','y_level0','width_level0','height_level0')]
    X,Y,W,H = [float(b[k]) for k in ('x_level0','y_level0','width_level0','height_level0')]
    return x < X+W+buffer and X < x+w+buffer and y < Y+H+buffer and Y < y+h+buffer


def interior_click(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('no candidate click pixels')
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1,1:-1]
    return tuple(int(v) for v in np.unravel_index(distance.argmax(), mask.shape))


def next_click(prediction, target, valid, clicks):
    available = valid.copy()
    for y,x,_ in clicks:
        available[y,x] = False
    candidates = []
    for sign, error in ((1,target & ~prediction),(0,~target & prediction)):
        error = error & available
        if error.any():
            y,x = interior_click(error)
            depth = distance_transform_edt(np.pad(error,1))[y+1,x+1]
            candidates.append((depth, int(error.sum()), sign, y, x))
    if not candidates:
        return None
    _,_,sign,y,x = max(candidates)
    return y,x,sign


def confusion_scores(confusion):
    matrix = np.asarray(confusion, dtype=np.float64)
    tp = matrix.diagonal(); gt = matrix.sum(1); pred = matrix.sum(0)
    present = gt > 0
    dice = np.divide(2*tp,gt+pred,out=np.zeros_like(tp),where=gt+pred>0)
    iou = np.divide(tp,gt+pred-tp,out=np.zeros_like(tp),where=gt+pred-tp>0)
    return {'macro_dice':float(dice[present].mean()),'macro_iou':float(iou[present].mean()),
            'pixel_accuracy':float(tp.sum()/matrix.sum()),
            'by_class': {str(k):{'dice':float(dice[k]),'iou':float(iou[k]),'gt_pixels':int(gt[k]),'pred_pixels':int(pred[k])} for k in range(len(tp))},
            'absent_gt_classes':np.flatnonzero(~present).tolist()}
