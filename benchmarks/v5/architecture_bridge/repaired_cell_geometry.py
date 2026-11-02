"""Versioned, GT-free uniform hash selection and pixel-center geometry."""
import hashlib
import numpy as np
from scipy import ndimage

POLICY = 'uniform_hash_cap255_pixel_centers_v1_20261102'


def cell_geometry(instances, field_key, mpp_xy, cap=255):
    if instances.ndim != 2 or not np.issubdtype(instances.dtype, np.integer) or (instances < 0).any():
        raise ValueError('nonnegative integer instance raster required')
    if not 0 < cap <= 255:
        raise ValueError('XCell supports at most 255 cells')
    mpp = np.asarray(mpp_xy, dtype=float)
    if mpp.shape != (2,) or not np.isfinite(mpp).all() or (mpp <= 0).any():
        raise ValueError('positive finite xy MPP required')
    ids = np.unique(instances); ids = ids[ids > 0]
    total = len(ids)
    rank = sorted(ids.tolist(), key=lambda label: hashlib.sha256(
        f'{POLICY}|{field_key}|{label}'.encode()).digest())
    selected = np.array(rank[:cap], dtype=instances.dtype)
    if not len(selected):
        return np.empty((0, 4), np.float32), selected, total
    yx = np.asarray(ndimage.center_of_mass(np.ones_like(instances, np.uint8), instances, selected))
    order = np.lexsort((selected, yx[:, 1], yx[:, 0]))
    selected, yx = selected[order], yx[order]
    area = np.bincount(instances.ravel())[selected]
    xy = (yx[:, ::-1] + .5) / np.array(instances.shape[::-1])
    meta = np.column_stack((xy, np.log1p(area * mpp.prod() / .25**2), np.zeros(len(selected)))).astype(np.float32)
    assert np.isfinite(meta).all() and ((meta[:, :2] > 0) & (meta[:, :2] < 1)).all()
    return meta, selected, total
