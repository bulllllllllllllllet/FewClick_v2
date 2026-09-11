"""Checkpoint-compatible copy of v4 CellToRegionAttention (see migration record)."""
from __future__ import annotations

import torch


class CellToRegionAttention(torch.nn.Module):
    def __init__(self, region_dim=256, cell_dim=74, hidden_dim=128):
        super().__init__()
        self.cell = torch.nn.Sequential(torch.nn.LayerNorm(cell_dim), torch.nn.Linear(cell_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, region_dim))
        self.query = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.key = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.value = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.fuse = torch.nn.Sequential(torch.nn.Linear(region_dim * 2 + 1, region_dim), torch.nn.ReLU(), torch.nn.LayerNorm(region_dim))

    def forward(self, region_tokens, assignment_at_cells, cells, cell_valid, total_cell_count=None):
        encoded = self.cell(cells)
        logits = torch.einsum("bkd,bnd->bkn", self.query(region_tokens), self.key(encoded)) * region_tokens.shape[-1] ** -0.5
        logits = logits + assignment_at_cells.clamp_min(1e-8).log()
        logits = logits.masked_fill(~cell_valid[:, None, :], float("-inf"))
        has_cells = cell_valid.any(1)
        safe_logits = torch.where(has_cells[:, None, None], logits, torch.zeros_like(logits))
        weights = safe_logits.softmax(-1) * cell_valid[:, None, :]
        context = torch.einsum("bkn,bnd->bkd", weights, self.value(encoded))
        selected = cell_valid.float().sum(1)
        if total_cell_count is None:
            total_cell_count = selected
        scale = torch.where(selected > 0, total_cell_count.to(selected).clamp_min(0) / selected.clamp_min(1), torch.zeros_like(selected))
        normalized = assignment_at_cells / assignment_at_cells.sum(1, keepdim=True).clamp_min(1e-8)
        count = (normalized * cell_valid[:, None, :]).sum(-1) * scale[:, None]
        fused = self.fuse(torch.cat((region_tokens, context, count.log1p().unsqueeze(-1)), -1))
        return {"fused_tokens": fused, "cell_context": context, "attention": weights,
                "has_cells": has_cells, "region_cell_count": count}
