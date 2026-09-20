"""Losses distinguish unlabeled entries from measured negatives."""
from __future__ import annotations

import torch


def partial_identity_loss(logits, allowed, valid, class_weights=None):
    if allowed.shape != logits.shape or valid.shape != logits.shape[:-1]:
        raise ValueError("identity supervision shapes disagree")
    if allowed.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError("allowed/valid must be boolean")
    # The full vocabulary is an uninformative label, not a supervised example.
    selected = valid & allowed.any(-1) & ~allowed.all(-1)
    if not selected.any():
        return logits.sum() * 0, 0
    logp = logits[selected].float().log_softmax(-1)
    per_cell = -logp.masked_fill(~allowed[selected], -torch.inf).logsumexp(-1)
    if class_weights is not None:
        if class_weights.shape != (logits.shape[-1],) or not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError("class_weights must be finite positive values for every identity class")
        # A partial label contributes the mean weight of its permitted classes.
        weight = (allowed[selected].float() * class_weights.to(logits)).sum(-1) / allowed[selected].sum(-1)
        loss = (per_cell * weight).sum() / weight.sum().clamp_min(1e-8)
    else:
        loss = per_cell.mean()
    return loss, int(selected.sum())


def balanced_softmax_identity_loss(logits, allowed, valid, train_class_counts):
    """Balanced-Softmax identity loss using only positive train-split class counts."""
    if allowed.shape != logits.shape or valid.shape != logits.shape[:-1]:
        raise ValueError("identity supervision shapes disagree")
    if allowed.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError("allowed/valid must be boolean")
    counts = torch.as_tensor(train_class_counts, dtype=torch.float32, device=logits.device)
    if counts.shape != (logits.shape[-1],) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Balanced Softmax requires finite positive train-only counts for every class")
    selected = valid & allowed.any(-1) & ~allowed.all(-1)
    if not selected.any():
        return logits.sum() * 0, 0
    # log(n_c) participates in both numerator and denominator during training;
    # ordinary logits are retained for inference.
    adjusted = logits[selected].float() + counts.log()
    logp = adjusted.log_softmax(-1)
    per_cell = -logp.masked_fill(~allowed[selected], -torch.inf).logsumexp(-1)
    return per_cell.mean(), int(selected.sum())


def distribution_loss(logits, target, valid):
    """Soft CE for independently verified, full-distribution region targets."""
    if logits.shape != target.shape or valid.shape != logits.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("distribution target shapes/mask disagree")
    if not valid.any():
        return logits.sum() * 0, 0
    y = target[valid].float()
    if not torch.isfinite(y).all() or (y < 0).any() or not torch.allclose(y.sum(-1), torch.ones_like(y[:, 0]), atol=1e-4):
        raise ValueError("valid distribution targets must be probabilities")
    return -(y * logits[valid].float().log_softmax(-1)).sum(-1).mean(), int(valid.sum())

def observed_composition_loss(logits, counts, observed_cells, coverage, min_observed_cells=1):
    """CE over mapped panel classes only; low coverage and absent observations are ignored/weighted."""
    if logits.shape != counts.shape or observed_cells.shape != logits.shape[:-1] or coverage.shape != logits.shape[:-1]:
        raise ValueError("observed composition shapes disagree")
    if min_observed_cells < 1 or not torch.isfinite(counts).all() or not torch.isfinite(coverage).all() or (counts < 0).any() or (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("invalid observed composition evidence")
    mapped = counts.sum(-1); valid = (observed_cells >= min_observed_cells) & (mapped > 0) & (coverage > 0)
    if not valid.any(): return logits.sum() * 0, 0
    target = counts[valid].float() / mapped[valid, None]
    per = -(target * logits[valid].float().log_softmax(-1)).sum(-1)
    weight = coverage[valid].float()
    return (per * weight).sum() / weight.sum().clamp_min(1e-8), int(valid.sum())
