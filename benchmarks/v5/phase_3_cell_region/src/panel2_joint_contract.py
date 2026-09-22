"""Named input contract for Phase-3 panel-2 joint supervision."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Panel2JointInputs:
    """One CRC patch in the argument order required by CellToRegionAttention."""

    region_tokens: torch.Tensor
    assignment_at_cells: torch.Tensor
    cells: torch.Tensor
    cell_valid: torch.Tensor
    total_cell_count: torch.Tensor

    def validate(self) -> None:
        if self.region_tokens.ndim != 3 or self.assignment_at_cells.ndim != 3 or self.cells.ndim != 3:
            raise ValueError("region tokens, assignment, and cells must be batched tensors")
        if self.region_tokens.shape[:2] != self.assignment_at_cells.shape[:2]:
            raise ValueError("region token and assignment shapes disagree")
        if self.cells.shape[:2] != self.cell_valid.shape:
            raise ValueError("cell features and validity mask shapes disagree")
        if self.assignment_at_cells.shape[-1] != self.cells.shape[1]:
            raise ValueError("assignment and cell counts disagree")
        if self.total_cell_count.shape != (self.cells.shape[0],):
            raise ValueError("total_cell_count must have one entry per patch")

    def fuse(self, cell_to_region: torch.nn.Module) -> dict[str, torch.Tensor]:
        self.validate()
        return cell_to_region(
            self.region_tokens,
            self.assignment_at_cells,
            self.cells,
            self.cell_valid,
            self.total_cell_count,
        )
