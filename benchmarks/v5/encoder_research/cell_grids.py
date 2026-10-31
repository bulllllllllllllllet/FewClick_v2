"""Fixed cell normalization and mean aggregation; no labels or fitted state creation."""
import numpy as np


def cell_vectors(raw, transform, arm):
    if arm == 'xcell':
        value = (raw['old_reg64'].astype(np.float64) - transform['xcell_mean']) / transform['xcell_std']
    elif arm == 'lemon':
        value = ((raw['lemon384'].astype(np.float64) - transform['lemon_mean']) / transform['lemon_std']).astype(np.float32)
        # sklearn's live Vt can be Fortran-strided while NPZ reload is C-strided.
        # Canonicalize before GEMV/GEMM so singleton-cell views replay exactly.
        components = np.ascontiguousarray(transform['components'])
        value = (value - transform['pca_mean']) @ components.T
        value = (value.astype(np.float64) - transform['projected_mean']) / transform['projected_std']
    else:
        raise ValueError(arm)
    return np.concatenate([raw['metadata10'], value], axis=1).astype(np.float32)


def cell_grid(positions, vectors):
    assert positions.shape == (len(vectors), 2) and vectors.shape[1:] == (74,)
    assert np.isfinite(positions).all() and ((positions >= 0) & (positions < 1)).all()
    xy = np.floor(positions * 32).astype(int)
    bins = xy[:, 1] * 32 + xy[:, 0]
    counts = np.bincount(bins, minlength=1024)
    sums = np.zeros((1024, 74), np.float64)
    np.add.at(sums, bins, vectors)
    average = sums / np.maximum(counts[:, None], 1)
    grid = np.concatenate([average, np.log1p(counts[:, None])], axis=1).T.reshape(75, 32, 32).astype(np.float32)
    assert np.isfinite(grid).all() and counts.sum() == len(vectors)
    return grid, counts.reshape(32, 32)
