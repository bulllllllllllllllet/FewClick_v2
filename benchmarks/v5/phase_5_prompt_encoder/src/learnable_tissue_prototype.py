"""Shared train-only metric projection; dynamic support-defined classes."""
import torch
from torch import nn
from torch.nn import functional as F
from .tissue_evaluation import prototype_logits


class LearnableTissuePrototype(nn.Module):
    def __init__(self, mode='single', dim=256):
        super().__init__()
        if mode not in ('single', 'multi'):
            raise ValueError('unknown prototype mode')
        self.mode = mode
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, batch):
        projected = dict(batch)
        for key in ('fine_tokens', 'positive_tokens', 'negative_tokens'):
            projected[key] = self.projection(batch[key])
        return {'logits': prototype_logits(projected, self.mode)}


def unprompted_bce(logits, batch):
    target = batch['binary_target']
    valid = (target != 255) & batch['fine_active'].bool() & ~batch['all_prompted_regions'].bool()
    counts = valid.sum(1)
    if not (counts > 0).all():
        raise ValueError('episode has no unprompted valid regions')
    losses = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1).float(), reduction='none')
    return ((losses * valid).sum(1) / counts).mean()
