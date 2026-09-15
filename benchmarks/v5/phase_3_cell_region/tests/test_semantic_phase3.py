"""Numerical/contract tests. Synthetic fixtures do not establish biological accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.v5.phase_3_cell_region.src.semantic_cell_region import SemanticCellToRegion
from benchmarks.v5.phase_3_cell_region.src.composition import pool_composition, physical_region_area
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import partial_identity_loss, distribution_loss
from benchmarks.v5.phase_3_cell_region.src.semantic_training import summarize_confusion
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention as FrozenV4


class SemanticTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.model = SemanticCellToRegion(3, 4)

    def inputs(self, n=5):
        cells = torch.randn(2, n, 74)
        cells[..., :2] = torch.rand(2, n, 2)
        return {"region_tokens": torch.randn(2, 4, 256), "cells": cells,
                "cell_valid": torch.ones(2, n, dtype=torch.bool),
                "assignment_at_cells": torch.rand(2, 4, n).softmax(1),
                "region_area_um2": torch.full((2, 4), 1000.)}

    def test_initialization_exactly_matches_frozen_v4(self):
        data = self.inputs()
        old = FrozenV4(256, 74)
        old.load_state_dict(self.model.baseline.state_dict())
        expected = old(data["region_tokens"], data["assignment_at_cells"], data["cells"], data["cell_valid"])
        actual = self.model(**data)
        torch.testing.assert_close(actual["fused_tokens"], expected["fused_tokens"], rtol=0, atol=0)

    def test_empty_cells_and_padding(self):
        for n in (0, 5):
            data = self.inputs(n)
            data["cell_valid"][:] = False
            with torch.no_grad():
                self.model.delta.bias.fill_(1.)
            result = self.model(**data)
            self.assertTrue(torch.isfinite(result["fused_tokens"]).all())
            self.assertEqual(float(result["residual"].detach().abs().sum()), 0.)
            self.assertEqual(float(result["detected_cell_count"].sum()), 0.)

    def test_identity_does_not_read_nuclear_classes_or_position(self):
        self.model.eval()
        original = self.inputs()
        changed = copy.deepcopy(original)
        changed["cells"][..., :2] = 0.5
        changed["cells"][..., 2:10] = 123.
        a, b = self.model(**original), self.model(**changed)
        torch.testing.assert_close(a["identity_logits"], b["identity_logits"], rtol=0, atol=0)

    def test_physical_density_and_known_composition(self):
        mass = torch.tensor([[[1., 1.], [0., 0.]]])
        p = torch.tensor([[[1., 0.], [0., 1.]]])
        result = pool_composition(mass, torch.ones(1, 2, dtype=torch.bool), p,
                                  torch.ones(1, 2, 3), torch.zeros(1, 2), torch.tensor([[1000., 0.]]))
        torch.testing.assert_close(result["composition"][0, 0], torch.tensor([.5, .5]))
        self.assertAlmostEqual(float(result["density_per_mm2"][0, 0]), 2000., places=2)
        self.assertFalse(result["evidence_valid"][0, 1])
        area = physical_region_area(torch.ones(1, 2, 4, 4) / 2, 100.)
        torch.testing.assert_close(area, torch.tensor([[800., 800.]]))

    def test_invalid_coordinate_and_assignment_rejected(self):
        data = self.inputs()
        data["cells"][0, 0, 0] = 1.1
        with self.assertRaises(ValueError):
            self.model(**data)
        data = self.inputs()
        data["assignment_at_cells"] *= 2
        with self.assertRaises(ValueError):
            self.model(**data)

    def test_partial_and_missing_labels(self):
        logits = torch.zeros(1, 2, 3, requires_grad=True)
        allowed = torch.tensor([[[True, True, False], [False, False, False]]])
        loss, count = partial_identity_loss(logits, allowed, torch.tensor([[True, False]]))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss.detach()), -float(torch.log(torch.tensor(2 / 3))), places=6)
        loss.backward()
        loss, count = distribution_loss(logits, torch.zeros_like(logits), torch.zeros(1, 2, dtype=torch.bool))
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.)

    def test_class_weighted_identity_loss_emphasizes_rare_class(self):
        logits = torch.tensor([[[2., 0.], [2., 0.]]], requires_grad=True)
        allowed = torch.tensor([[[True, False], [False, True]]])
        plain, _ = partial_identity_loss(logits, allowed, torch.tensor([[True, True]]))
        weighted, _ = partial_identity_loss(logits, allowed, torch.tensor([[True, True]]), torch.tensor([1., 5.]))
        self.assertGreater(float(weighted.detach()), float(plain.detach()))
        weighted.backward()

    def test_fusion_gradient_and_checkpoint_reload(self):
        data = self.inputs()
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.001)
        before = self.model.delta.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            result = self.model(**data)
            loss = torch.nn.functional.cross_entropy(result["tissue_logits"].reshape(-1, 4), torch.zeros(8, dtype=torch.long))
            loss.backward()
            self.assertTrue(all(p.grad is None for p in self.model.identity.parameters()))
            self.assertTrue(all(p.grad is None for p in self.model.baseline.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(before, self.model.delta.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pth"
            torch.save(self.model.state_dict(), path)
            other = SemanticCellToRegion(3, 4)
            other.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
            other.eval()
            torch.testing.assert_close(other(**data)["fused_tokens"], self.model(**data)["fused_tokens"], rtol=0, atol=0)

    def test_identity_stage_only_updates_identity(self):
        self.model.set_stage("identity")
        self.model.train()
        result = self.model(**self.inputs())
        result["identity_logits"].square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in self.model.identity.parameters()))
        self.assertFalse(self.model.delta.weight.requires_grad)
        self.assertFalse(self.model.baseline.training)

    def test_supported_class_metrics_retain_false_positives(self):
        value = summarize_confusion(torch.tensor([[2, 1, 1], [0, 0, 0], [0, 0, 0]]))
        self.assertAlmostEqual(value["macro_f1"], 2 / 3)
        self.assertEqual(value["support"], [4., 0., 0.])


if __name__ == "__main__":
    unittest.main()
