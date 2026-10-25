"""Model-independent boundary eligibility and Euclidean physical tolerance."""
import numpy as np
from scipy.ndimage import distance_transform_edt

VERSION = 'physical_euclidean_gt_boundary_v1_1025'


def boundary(mask, valid):
    mask = np.asarray(mask, bool); valid = np.asarray(valid, bool)
    if mask.ndim != 2 or mask.shape != valid.shape:
        raise ValueError('expected equal [H,W] masks')
    edge = np.zeros_like(mask)
    vertical = (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    horizontal = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[1:] |= vertical; edge[:-1] |= vertical
    edge[:, 1:] |= horizontal; edge[:, :-1] |= horizontal
    return edge & valid


def boundary_score(predicted_edge, true_edge, mpp, tolerance_um=2.):
    predicted_edge = np.asarray(predicted_edge, bool); true_edge = np.asarray(true_edge, bool)
    mpp = np.asarray(mpp, float)
    if (predicted_edge.shape != true_edge.shape or predicted_edge.ndim != 2
            or mpp.shape != (2,) or not np.isfinite(mpp).all() or (mpp <= 0).any()
            or not np.isfinite(tolerance_um) or tolerance_um < 0):
        raise ValueError('invalid physical boundary geometry')
    npred = int(predicted_edge.sum()); ntrue = int(true_edge.sum())
    result = dict(predicted_boundary_pixels=npred, true_boundary_pixels=ntrue,
                  boundary_evaluable=ntrue > 0, bf1_um2=None)
    if not ntrue:
        return result
    if not npred:
        return result | dict(bf1_um2=0.)
    dp = distance_transform_edt(~predicted_edge, sampling=mpp[::-1])
    dt = distance_transform_edt(~true_edge, sampling=mpp[::-1])
    precision = float((dt[predicted_edge] <= tolerance_um + 1e-9).mean())
    recall = float((dp[true_edge] <= tolerance_um + 1e-9).mean())
    return result | dict(bf1_um2=2*precision*recall/(precision+recall) if precision+recall else 0.)


def score(prediction, truth, valid, mpp):
    prediction = np.asarray(prediction, bool); truth = np.asarray(truth, bool); valid = np.asarray(valid, bool)
    if prediction.shape != truth.shape or truth.shape != valid.shape or not valid.any():
        raise ValueError('invalid scoring masks')
    tp = int((prediction & truth & valid).sum()); fp = int((prediction & ~truth & valid).sum())
    fn = int((~prediction & truth & valid).sum()); positive = int((truth & valid).sum())
    return dict(tp=tp, fp=fp, fn=fn, positive=positive, valid=int(valid.sum()),
                dice=2*tp/(2*tp+fp+fn) if positive else None,
                false_positive_fraction=fp/int(valid.sum()),
                **boundary_score(boundary(prediction, valid), boundary(truth, valid), mpp))
