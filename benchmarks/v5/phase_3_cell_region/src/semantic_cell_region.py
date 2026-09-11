"""First-stage semantic residual, preserving the frozen 256-D region contract."""
from __future__ import annotations

import torch
from torch import nn

from .composition import pool_composition
from .frozen_cell_region import CellToRegionAttention
from .semantic_cells import CellIdentity


class SemanticCellToRegion(nn.Module):
    schema_version = "semantic_phase3_v1"

    def __init__(self, identity_classes, tissue_classes, region_dim=256, hidden=128):
        super().__init__()
        self.baseline = CellToRegionAttention(region_dim, 74)
        self.baseline.requires_grad_(False)
        self.identity = CellIdentity(identity_classes, hidden)
        self.evidence = nn.Sequential(nn.Linear(hidden + identity_classes + 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Linear(region_dim + hidden, 1)
        self.delta = nn.Linear(hidden, region_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.tissue_head = nn.Linear(region_dim, tissue_classes)
        self.composition_head = nn.Linear(region_dim, identity_classes)
        self.stage = "fusion"
        self.set_stage("fusion")

    def set_stage(self, stage):
        if stage not in ("identity", "fusion"):
            raise ValueError("stage must be identity or fusion")
        self.stage = stage
        self.requires_grad_(False)
        if stage == "identity":
            self.identity.requires_grad_(True)
        else:
            for module in (self.evidence, self.gate, self.delta, self.tissue_head, self.composition_head):
                module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        if self.stage != "identity":
            self.identity.eval()
        return self

    def forward(self, region_tokens, cells, cell_valid, assignment_at_cells, region_area_um2):
        if cells.ndim != 3 or cells.shape[-1] != 74 or cell_valid.dtype != torch.bool:
            raise ValueError("expected cells [B,N,74] and boolean valid")
        if region_tokens.ndim != 3 or region_tokens.shape[:2] != assignment_at_cells.shape[:2]:
            raise ValueError("region tokens/assignment shapes disagree")
        if not torch.isfinite(region_tokens).all() or not torch.isfinite(cells).all():
            raise ValueError("non-finite input")
        if cell_valid.shape != cells.shape[:2]:
            raise ValueError("cell validity shape mismatch")
        xy = cells[..., :2][cell_valid]
        if (xy < 0).any() or (xy > 1).any():
            raise ValueError("valid cell positions must be normalized in [0,1]")
        identity = self.identity(cells[..., 10:74])
        stats = pool_composition(assignment_at_cells, cell_valid, identity["identity_probabilities"],
                                 identity["hidden"], identity["uncertainty"], region_area_um2)
        base = self.baseline(region_tokens, assignment_at_cells, cells, cell_valid)
        features = torch.cat((stats["pooled_hidden"], stats["composition"],
                              stats["density_per_mm2"].log1p()[..., None],
                              stats["region_uncertainty"][..., None]), -1)
        evidence = self.evidence(features)
        gate = self.gate(torch.cat((region_tokens, evidence), -1)).sigmoid()
        residual = self.delta(evidence) * gate * stats["evidence_valid"][..., None]
        fused = base["fused_tokens"] + residual
        return {**base, **identity, **stats, "baseline_tokens": base["fused_tokens"],
                "fused_tokens": fused, "residual": residual, "gate": gate,
                "tissue_logits": self.tissue_head(fused),
                "composition_logits": self.composition_head(fused)}
