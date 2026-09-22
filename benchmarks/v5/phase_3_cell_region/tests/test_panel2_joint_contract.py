import unittest

import torch

from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v5.phase_3_cell_region.src.panel2_joint_contract import Panel2JointInputs


class Panel2JointContractTest(unittest.TestCase):
    def test_named_contract_matches_direct_cell_fusion(self):
        torch.manual_seed(4)
        model = CellToRegionAttention(region_dim=8, cell_dim=3, hidden_dim=4)
        region_tokens = torch.randn(1, 2, 8)
        cells = torch.randn(1, 3, 3)
        assignment = torch.rand(1, 2, 3)
        valid = torch.tensor([[True, True, False]])
        total = torch.tensor([5.])
        value = Panel2JointInputs(region_tokens, assignment, cells, valid, total)
        torch.testing.assert_close(
            value.fuse(model)["fused_tokens"],
            model(region_tokens, assignment, cells, valid, total)["fused_tokens"],
        )

    def test_rejects_positional_order_shape_mismatch(self):
        value = Panel2JointInputs(
            torch.randn(1, 2, 8),
            torch.randn(1, 3, 3),
            torch.rand(1, 2, 3),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([3.]),
        )
        with self.assertRaisesRegex(ValueError, "region token and assignment"):
            value.validate()


if __name__ == "__main__":
    unittest.main()
