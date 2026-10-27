"""Scoring-only remote retrieval counters on original GT cell centres.

Full-image GT components must be computed before any query crop. No component
size filter is applied. Predictions and source prompts must already be frozen.
"""
import numpy as np


def binary_counts(truth, prediction, valid):
    positive = int(np.count_nonzero(truth & valid))
    negative = int(np.count_nonzero(~truth & valid))
    tp = int(np.count_nonzero(truth & prediction & valid))
    fp = int(np.count_nonzero(~truth & prediction & valid))
    return dict(tp=tp, fp=fp, fn=positive-tp, positive=positive,
                negative=negative, valid=positive+negative)


def summarize(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    positive, negative = counts['positive'], counts['negative']
    return dict(counts, dice=2*tp/(2*tp+fp+fn) if positive else None,
                recall=tp/positive if positive else None,
                precision=tp/(tp+fp) if tp+fp else 0.,
                false_positive_fraction=fp/counts['valid'] if counts['valid'] else None,
                false_positive_rate=fp/negative if negative else None)


def score_domain(labels, components, prediction, target, prompted_components,
                 native_x_centres, native_y_centres, native_mpp, positive_native_xy):
    """Score one frozen task/budget in a GT-aligned rectangular domain.

    ``prediction`` is a Boolean mask sampled at original GT cell centres.
    Distances are pixelwise nearest-positive-point physical distances, never
    distances between component centroids. Negative points do not censor GT.
    A component's denominator is its area inside this scoring domain.
    """
    labels = np.asarray(labels)
    components = np.asarray(components)
    prediction = np.asarray(prediction)
    if labels.ndim != 2 or components.shape != labels.shape or prediction.shape != labels.shape:
        raise ValueError('aligned two-dimensional GT/component/prediction arrays required')
    if prediction.dtype != np.bool_ or target not in range(12) or target == 2:
        raise ValueError('Boolean prediction and foreground tissue target required')
    if not np.array_equal(components == 0, labels == 255):
        raise ValueError('void255 alone must have component ID0')
    prompted = np.asarray(sorted(set(prompted_components)), dtype=components.dtype)
    if np.any(prompted == 0):
        raise ValueError('void cannot be a prompted component')
    removed = np.isin(components, prompted)
    if np.any(removed & (labels != target)):
        raise ValueError('only positive-point target components can be censored')
    valid = labels != 255
    truth = labels == target
    eligible = valid & ~removed
    ids, areas = np.unique(components[truth & eligible], return_counts=True)
    hits = np.zeros(len(ids), dtype=np.int64)
    if len(ids):
        hit_ids, hit_counts = np.unique(components[truth & prediction & eligible], return_counts=True)
        hits[np.searchsorted(ids, hit_ids)] = hit_counts
    points = np.asarray(positive_native_xy, dtype=np.float64)
    x, y = np.asarray(native_x_centres), np.asarray(native_y_centres)
    mx, my = native_mpp
    if x.shape != (labels.shape[1],) or y.shape != (labels.shape[0],) or points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError('native GT cell centres and positive support points required')
    if not all(np.isfinite(a).all() for a in (points, x, y, np.asarray(native_mpp))) or min(mx, my) <= 0:
        raise ValueError('finite coordinates and positive physical pixel spacing required')
    distance2 = np.full(labels.shape, np.inf)
    for px, py in points:
        np.minimum(distance2, ((x[None, :]-px)*mx)**2 + ((y[:, None]-py)*my)**2, out=distance2)
    strata = ((distance2 < 1000**2),
              (distance2 >= 1000**2) & (distance2 < 5000**2),
              (distance2 >= 5000**2))
    return dict(standard=summarize(binary_counts(truth, prediction, valid)),
                unprompted=summarize(binary_counts(truth, prediction, eligible)),
                censored_target_pixels=int(np.count_nonzero(removed)),
                components=dict(ids=ids.tolist(), areas=areas.tolist(), hits=hits.tolist(),
                    count=len(ids), mean_area_recall=float(np.mean(hits/areas)) if len(ids) else None,
                    detected_at_50=int(np.count_nonzero(2*hits >= areas)),
                    detected_at_10=int(np.count_nonzero(10*hits >= areas))),
                distance=[dict(bin=name, **summarize(binary_counts(truth, prediction, eligible & mask)))
                          for name, mask in zip(('<1mm', '1-5mm', '>=5mm'), strata)])
