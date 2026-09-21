"""Project Phase-2 soft token assignments into a declared physical grid."""
from __future__ import annotations

import numpy as np
import torch


def patch_pixel_grid_indices(height, width, origin_xy, source_to_common, mpp_um, grid_um, source_pixel_stride=1.):
    """Return common-coordinate grid IDs for centres of an H&E patch's pixels.

    ``origin_xy`` is the level-0 source coordinate of the registered patch.  It
    is deliberately explicit because a local H&E/DAPI registration is not an
    exact whole-slide coordinate calibration.
    """
    if height < 1 or width < 1 or mpp_um <= 0 or grid_um <= 0 or source_pixel_stride <= 0:
        raise ValueError("positive patch dimensions and physical scales required")
    matrix = np.asarray(source_to_common, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("finite 3x3 source-to-common matrix required")
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64) + .5, np.arange(width, dtype=np.float64) + .5, indexing="ij")
    points = np.stack((xx * float(source_pixel_stride) + float(origin_xy[0]), yy * float(source_pixel_stride) + float(origin_xy[1]), np.ones_like(xx)), axis=-1)
    common = points @ matrix.T
    if not np.isfinite(common).all():
        raise ValueError("non-finite common coordinates")
    return np.floor(common[..., :2] * float(mpp_um) / float(grid_um)).astype(np.int64)


def project_assignment_to_grid(assignment, pixel_grid):
    """Aggregate [K,H,W] soft assignment mass into physical grid cells."""
    if assignment.ndim != 3 or not torch.isfinite(assignment).all() or (assignment < 0).any():
        raise ValueError("finite non-negative assignment [K,H,W] required")
    k, h, w = assignment.shape
    pixel_grid = np.asarray(pixel_grid, dtype=np.int64)
    if pixel_grid.shape != (h, w, 2):
        raise ValueError("pixel_grid must be [H,W,2]")
    coordinates, inverse = np.unique(pixel_grid.reshape(-1, 2), axis=0, return_inverse=True)
    index = torch.from_numpy(inverse).to(assignment.device)
    mass = assignment.reshape(k, -1).new_zeros((k, len(coordinates)))
    mass.scatter_add_(1, index.unsqueeze(0).expand(k, -1), assignment.reshape(k, -1))
    fraction = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    return {"grid_xy": coordinates, "token_grid_mass": mass, "token_grid_fraction": fraction}


def project_observed_targets(projection, targets, classes=5, min_token_grid_fraction=.80):
    """Attach panel-specific observed counts only where physical overlap is high."""
    if not 0 < min_token_grid_fraction <= 1:
        raise ValueError("min_token_grid_fraction must be in (0,1]")
    mass, fraction = projection["token_grid_mass"], projection["token_grid_fraction"]
    if fraction.shape != mass.shape:
        raise ValueError("projection mass/fraction shapes disagree")
    k = mass.shape[0]
    counts = mass.new_zeros((k, classes)); observed = mass.new_zeros(k); coverage = mass.new_zeros(k); valid = torch.zeros(k, dtype=torch.bool, device=mass.device)
    for grid_index, xy in enumerate(np.asarray(projection["grid_xy"], dtype=np.int64)):
        value = targets.get((int(xy[0]), int(xy[1])))
        if value is None:
            continue
        raw_counts, raw_observed, raw_coverage = value
        candidate = fraction[:, grid_index] >= min_token_grid_fraction
        if not candidate.any():
            continue
        target_counts = torch.as_tensor(raw_counts, dtype=mass.dtype, device=mass.device)
        if target_counts.shape != (classes,) or not torch.isfinite(target_counts).all() or (target_counts < 0).any() or raw_observed < 0 or not 0 <= raw_coverage <= 1:
            raise ValueError("invalid observed target values")
        counts[candidate] = target_counts
        observed[candidate] = float(raw_observed)
        coverage[candidate] = float(raw_coverage)
        valid[candidate] = True
    return {"immune_counts": counts, "immune_observed_cells": observed, "immune_coverage": coverage, "immune_target_valid": valid}
