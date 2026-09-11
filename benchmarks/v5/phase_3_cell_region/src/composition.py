"""Area-calibrated statistics; no RNA/protein ground truth enters this path."""
from __future__ import annotations

import torch


def pool_composition(mass, valid, probabilities, hidden, uncertainty, area_um2):
    """Complete supplied detections only, not a claim about undetected cells.

    All supplied valid cells must have features. Subsampled inventories are rejected
    by the manifest loader: inverse-probability estimation is not implemented yet.
    """
    if mass.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("expected [B,K,N] assignment and boolean valid")
    b, k, n = mass.shape
    if valid.shape != (b, n) or area_um2.shape != (b, k):
        raise ValueError("assignment/valid/area shapes disagree")
    if probabilities.shape[:2] != (b, n) or hidden.shape[:2] != (b, n) or uncertainty.shape != (b, n):
        raise ValueError("cell evidence shape mismatch")
    for value in (mass, probabilities, hidden, uncertainty, area_um2):
        if not torch.isfinite(value).all():
            raise ValueError("non-finite composition input")
    if (mass < 0).any() or (area_um2 < 0).any() or (probabilities < 0).any():
        raise ValueError("negative mass, area or probability")
    if not torch.allclose(mass.sum(1)[valid], torch.ones_like(mass.sum(1)[valid]), atol=1e-4):
        raise ValueError("cell assignment must sum to one over regions")
    if not torch.allclose(probabilities.sum(-1)[valid], torch.ones_like(uncertainty[valid]), atol=1e-4):
        raise ValueError("identity probabilities must sum to one")
    weights = mass.float() * valid[:, None, :]
    counts = weights.sum(-1)
    normalized = weights / counts.clamp_min(1e-8)[..., None]
    composition = torch.einsum("bkn,bnc->bkc", normalized, probabilities.float())
    pooled = torch.einsum("bkn,bnh->bkh", normalized, hidden.float())
    entropy = torch.einsum("bkn,bn->bk", normalized, uncertainty.float())
    evidence_valid = (counts > 1e-8) & (area_um2 >= 1.0)
    density = torch.where(evidence_valid, counts / (area_um2 / 1e6).clamp_min(1e-6), 0.0)
    return {"composition": composition, "pooled_hidden": pooled,
            "region_uncertainty": entropy, "detected_cell_count": counts,
            "density_per_mm2": density, "evidence_valid": evidence_valid}


def physical_region_area(assignment, pixel_area_um2):
    """pixel_area refers to one assignment-grid pixel, NOT an original WSI pixel."""
    if assignment.ndim != 4 or not torch.isfinite(assignment).all() or (assignment < 0).any():
        raise ValueError("invalid spatial assignment")
    scale = torch.as_tensor(pixel_area_um2, device=assignment.device, dtype=torch.float32)
    if scale.ndim == 0:
        scale = scale.expand(assignment.shape[0])
    if scale.shape != (assignment.shape[0],) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("positive assignment-grid pixel area required per sample")
    if not torch.allclose(assignment.sum(1), torch.ones_like(assignment[:, 0]), atol=1e-4):
        raise ValueError("spatial assignment must sum to one")
    return assignment.float().sum((-1, -2)) * scale[:, None]
