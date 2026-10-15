"""Image-conditioned residual refinement; inference gate never uses labels."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask[:, None].float(), 2 * radius + 1, 1, radius)[:, 0].bool()


class BoundaryPixelRefiner(nn.Module):
    def __init__(self, width: int = 24, radius: int = 8, residual_limit: float = 2.0):
        super().__init__()
        self.radius = radius
        self.residual_limit = residual_limit
        self.net = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor, probability: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3 or probability.shape != image.shape[:1] + image.shape[-2:]:
            raise ValueError('expected image [B,3,H,W], probability [B,H,W]')
        if not torch.isfinite(image).all() or not torch.isfinite(probability).all():
            raise ValueError('nonfinite refiner input')
        if (probability < 0).any() or (probability > 1).any():
            raise ValueError('probability outside [0,1]')
        probability = probability.detach().float()
        edge = binary_boundary(probability >= .5, torch.ones_like(probability, dtype=torch.bool))
        gate = dilate(edge, self.radius) | ((probability >= .2) & (probability <= .8))
        uncertainty = 4 * probability * (1 - probability)
        raw = self.net(torch.cat((image.float(), probability[:, None], uncertainty[:, None]), 1))[:, 0]
        residual = self.residual_limit * raw.tanh() * gate
        logits = torch.logit(probability.clamp(1e-6, 1-1e-6)) + residual
        refined = torch.where(gate, logits.sigmoid(), probability)
        return {'pixel_probability': refined, 'pixel_logits': logits, 'gate': gate, 'residual': residual}


def refinement_loss(output: dict, truth: torch.Tensor, valid: torch.Tensor, boundary_weight: float) -> torch.Tensor:
    from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import balanced_pixel_bce, soft_pixel_dice, soft_boundary_loss
    logits, probability = output['pixel_logits'], output['pixel_probability']
    total = balanced_pixel_bce(logits, truth, valid) + soft_pixel_dice(probability, truth, valid)
    if boundary_weight:
        # Do not train at ignore interfaces; each supervised pixel has a valid neighbourhood.
        safe = ~dilate(~valid, 4) & valid
        band = dilate(binary_boundary(truth, valid), 4) & safe
        band_loss = F.binary_cross_entropy_with_logits(logits[band], truth[band].float()) if band.any() else logits.sum() * 0
        total = total + boundary_weight * band_loss + .1 * soft_boundary_loss(probability, truth, valid)
    return total
