"""Positive/negative multi-prototype matching for open-ended prompt tasks."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def normalized_lme_similarity(
    query: torch.Tensor, prototypes: torch.Tensor, prototype_valid: torch.Tensor, temperature: float | torch.Tensor = 1.0
) -> torch.Tensor:
    """Return normalized log-mean-exp cosine similarity without prototype-count bias."""
    if query.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("query must be [B,N,D] and prototypes [B,P,D]")
    if query.shape[0] != prototypes.shape[0] or query.shape[-1] != prototypes.shape[-1]:
        raise ValueError("query/prototype batch or feature dimensions disagree")
    if prototype_valid.shape != prototypes.shape[:2] or prototype_valid.dtype != torch.bool:
        raise ValueError("prototype_valid must be boolean [B,P]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not prototype_valid.any(dim=1).all():
        raise ValueError("each item requires at least one valid prototype")
    q = F.normalize(query, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    similarity = torch.einsum("bnd,bpd->bnp", q, p) / temperature
    similarity = similarity.masked_fill(~prototype_valid[:, None], float("-inf"))
    count = prototype_valid.sum(-1).to(similarity.dtype)
    return torch.logsumexp(similarity, dim=-1) - count.log()[:, None]


class PositiveNegativePrototypeMatcher(nn.Module):
    """Task-conditioned binary logit from dynamic positive and negative prototypes."""

    def __init__(self, temperature: float = 1.0, learnable_temperature: bool = False):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(float(temperature)), persistent=True)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(torch.finfo(torch.float32).eps) if self.learnable_temperature else self.fixed_temperature

    def forward(
        self, query: torch.Tensor, positive: torch.Tensor, positive_valid: torch.Tensor,
        negative: torch.Tensor, negative_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temperature = self.temperature()
        positive_score = normalized_lme_similarity(query, positive, positive_valid, temperature)
        negative_score = normalized_lme_similarity(query, negative, negative_valid, temperature)
        return {"logits": positive_score - negative_score, "positive_similarity": positive_score,
                "negative_similarity": negative_score, "temperature": self.temperature()}
