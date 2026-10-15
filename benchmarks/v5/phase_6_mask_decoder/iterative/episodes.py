"""Build one correction step from a real previous prediction and pixel truth."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from .iterative_decoder import click_values


def select_correction_click(previous_probability: torch.Tensor, truth: torch.Tensor,
                            valid: torch.Tensor, prefer_sign: int | None = None,
                            threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest FP/FN component, then its most interior pixel; no GT enters model input.

    This is an oracle click simulator for training and paired evaluation only.
    Returns normalized xy and 1 for a missed target or 0 for a false positive.
    """
    if previous_probability.shape != truth.shape or truth.shape != valid.shape or truth.ndim != 3:
        raise ValueError("prediction, truth and valid must be [B,H,W]")
    coords: list[list[float]] = []
    signs: list[int] = []
    for probability, target, included in zip(previous_probability, truth, valid):
        predicted = probability.detach().float().cpu().numpy() >= threshold
        target_np = target.detach().bool().cpu().numpy()
        valid_np = included.detach().bool().cpu().numpy()
        false_positive = predicted & ~target_np & valid_np
        false_negative = ~predicted & target_np & valid_np
        count_positive = int(false_positive.sum())
        count_negative = int(false_negative.sum())
        if prefer_sign is None:
            sign = 0 if count_positive >= count_negative else 1
        else:
            sign = int(prefer_sign)
        error = false_negative if sign else false_positive
        if not error.any():
            raise ValueError(f"no {'false negative' if sign else 'false positive'} pixel for correction")
        labels, components = ndimage.label(error)
        sizes = np.bincount(labels.ravel())[1:]
        largest = labels == (int(sizes.argmax()) + 1)
        distance = ndimage.distance_transform_edt(largest)
        y, x = np.unravel_index(int(distance.argmax()), distance.shape)
        coords.append([(x + 0.5) / error.shape[1], (y + 0.5) / error.shape[0]])
        signs.append(sign)
    return previous_probability.new_tensor(coords), torch.tensor(signs, device=truth.device, dtype=torch.long)


def tokens_at_clicks(region_tokens: torch.Tensor, assignment: torch.Tensor,
                     xy: torch.Tensor) -> torch.Tensor:
    """Sample actual soft assignment at clicked pixels, then pool region tokens."""
    if assignment.shape[:2] != region_tokens.shape[:2]:
        raise ValueError("assignment and region token slots differ")
    weights = assignment.float() / assignment.float().sum(1, keepdim=True).clamp_min(1e-8)
    height, width = assignment.shape[-2:]
    x = (xy[..., 0].clamp(0, 1) * width).long().clamp(max=width - 1)
    y = (xy[..., 1].clamp(0, 1) * height).long().clamp(max=height - 1)
    click_weights = weights.flatten(2).gather(2, (y * width + x)[:, None].expand(-1, weights.shape[1], -1))
    return torch.einsum("bkp,bkd->bpd", click_weights, region_tokens.float())


def append_click(old_xy: torch.Tensor, old_sign: torch.Tensor, old_valid: torch.Tensor,
                 new_xy: torch.Tensor, new_sign: torch.Tensor,
                 region_tokens: torch.Tensor, assignment: torch.Tensor
                 ) -> dict[str, torch.Tensor]:
    """Pack unpadded valid prompt history plus one new click for one batch."""
    if old_xy.shape[:2] != old_sign.shape or old_valid.shape != old_sign.shape:
        raise ValueError("old prompt history shape mismatch")
    if new_xy.shape != (old_xy.shape[0], 2) or new_sign.shape != (old_xy.shape[0],):
        raise ValueError("new click shape mismatch")
    batch, old_points = old_sign.shape
    new_index = old_valid.sum(-1).long()
    if not torch.equal(new_index, torch.full_like(new_index, old_points)):
        raise ValueError("append_click expects compact unpadded prompt history")
    xy = torch.cat((old_xy, new_xy[:, None]), 1)
    sign = torch.cat((old_sign, new_sign[:, None]), 1)
    valid = torch.cat((old_valid, torch.ones(batch, 1, dtype=torch.bool, device=old_valid.device)), 1)
    return {"prompt_xy": xy, "prompt_sign": sign, "prompt_valid": valid,
            "prompt_tokens": tokens_at_clicks(region_tokens, assignment, xy),
            "new_index": new_index}


def correction_loss(output: dict[str, torch.Tensor], previous_probability: torch.Tensor,
                    truth: torch.Tensor, valid: torch.Tensor,
                    prompt_xy: torch.Tensor, prompt_sign: torch.Tensor,
                    prompt_valid: torch.Tensor, new_index: torch.Tensor,
                    click_weight: float = 2.0, remote_weight: float = 0.5,
                    keep_weight: float = 2.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pixel supervision plus signed monotonic correction and preservation."""
    current = output["pixel_probability"].clamp(1e-6, 1 - 1e-6)
    if current.shape != truth.shape or current.shape != valid.shape:
        raise ValueError("pixel target shape mismatch")
    target = truth.float()
    included = valid.bool()
    bce = F.binary_cross_entropy(current[included], target[included])
    overlap = (current * target * included).flatten(1).sum(-1)
    denominator = ((current + target) * included).flatten(1).sum(-1)
    dice = (1 - (2 * overlap + 1) / (denominator + 1)).mean()
    click = click_values(current, prompt_xy)
    signed_click = torch.where(prompt_sign.bool(), F.relu(0.7 - click), F.relu(click - 0.3))
    click_term = signed_click[prompt_valid].mean()
    sign = prompt_sign.gather(1, new_index[:, None]).squeeze(1).bool()
    fp = (previous_probability >= 0.5) & ~truth.bool() & included
    fn = (previous_probability < 0.5) & truth.bool() & included
    change = current - previous_probability.detach()
    remote_mask = torch.where(sign[:, None, None], fn, fp)
    remote_error = torch.where(sign[:, None, None], F.relu(0.05 - change), F.relu(0.05 + change))
    remote = remote_error[remote_mask].mean() if remote_mask.any() else current.sum() * 0
    correct = torch.where(sign[:, None, None], ~truth.bool() & included & (previous_probability < 0.5),
                          truth.bool() & included & (previous_probability >= 0.5))
    keep_error = torch.where(sign[:, None, None], F.relu(change - 0.02), F.relu(-change - 0.02))
    keep = keep_error[correct].mean() if correct.any() else current.sum() * 0
    total = bce + dice + click_weight * click_term + remote_weight * remote + keep_weight * keep
    return total, {"bce": bce.detach(), "dice": dice.detach(), "click": click_term.detach(),
                   "remote": remote.detach(), "keep": keep.detach()}
