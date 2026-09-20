"""Panel-specific regional evidence heads; not a replacement for complete composition targets."""
from __future__ import annotations
import torch
from torch import nn

class ObservedRegionEvidenceHeads(nn.Module):
    def __init__(self, region_dim: int = 256, structure_classes: int = 5, immune_classes: int = 5):
        super().__init__()
        self.structure_head = nn.Linear(region_dim, structure_classes)
        self.immune_head = nn.Linear(region_dim, immune_classes)
    def forward(self, region_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if region_tokens.ndim != 3 or not torch.isfinite(region_tokens).all(): raise ValueError("finite region tokens [B,K,D] required")
        return {"structure_observed_logits": self.structure_head(region_tokens), "immune_observed_logits": self.immune_head(region_tokens)}
