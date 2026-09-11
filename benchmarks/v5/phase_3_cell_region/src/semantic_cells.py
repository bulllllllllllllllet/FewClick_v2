"""H&E-only identity adapter. Metadata and teacher labels are not inputs."""
from __future__ import annotations

import torch
from torch import nn


class CellIdentity(nn.Module):
    def __init__(self, classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        if classes < 2:
            raise ValueError("identity vocabulary requires at least two classes")
        self.encoder = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, classes)
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, reg_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if reg_features.shape[-1] != 64 or not torch.isfinite(reg_features).all():
            raise ValueError("identity input must be finite XCell reg-64")
        if not torch.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        hidden = self.encoder(reg_features)
        logits = self.head(hidden)
        probabilities = (logits.float() / self.temperature).softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / torch.log(logits.new_tensor(logits.shape[-1]).float())
        return {"hidden": hidden, "identity_logits": logits,
                "identity_probabilities": probabilities, "uncertainty": entropy}
