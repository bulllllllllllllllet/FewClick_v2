import importlib.util
from pathlib import Path
import unittest

import torch

PATH = Path(__file__).resolve().parents[1] / "tools" / "train_semantic_phase3.py"
SPEC = importlib.util.spec_from_file_location("identity_training", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IdentityBalancingTest(unittest.TestCase):
    def test_effective_number_weights_upweight_rare_observed_class(self):
        weights = MODULE.effective_number_weights(torch.tensor([2806, 927, 12, 1356, 2717]), .999, 5.)
        self.assertEqual(tuple(weights.shape), (5,))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertLessEqual(float(weights.max()), 5.)

    def test_absent_class_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.effective_number_weights(torch.tensor([1, 0]), .999, 5.)

    def test_inverse_frequency_equalizes_total_class_loss_mass(self):
        counts = torch.tensor([44210, 39492, 1731, 31214, 282920])
        weights = MODULE.inverse_frequency_weights(counts)
        torch.testing.assert_close(weights * counts, torch.full((5,), float((weights * counts).mean())), rtol=1e-5, atol=1e-3)
        self.assertGreater(float(weights[2]), float(weights[4]))

    def test_square_root_inverse_is_between_uniform_and_strict_inverse(self):
        counts = torch.tensor([44210, 39492, 1731, 31214, 282920])
        tempered = MODULE.power_inverse_frequency_weights(counts, .5)
        strict = MODULE.inverse_frequency_weights(counts)
        self.assertGreater(float(tempered[2]), float(tempered[4]))
        self.assertLess(float(tempered[2] / tempered[4]), float(strict[2] / strict[4]))
        self.assertAlmostEqual(float(tempered.mean()), 1., places=6)

    def test_invalid_power_inverse_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.power_inverse_frequency_weights(torch.tensor([1, 2]), 0.)

    def test_delayed_reweighting_keeps_warmup_uniform(self):
        weights = torch.tensor([.5, 2.])
        self.assertIsNone(MODULE.identity_loss_weights_for_epoch(weights, 3, 2))
        self.assertTrue(torch.equal(MODULE.identity_loss_weights_for_epoch(weights, 3, 3), weights))
        with self.assertRaises(ValueError):
            MODULE.identity_loss_weights_for_epoch(weights, -1, 0)

    def test_patch_sampling_upweights_patch_containing_rare_class(self):
        class Dummy:
            spec = {"identity_classes": ["a", "b"]}
            def __len__(self): return 2
            def __iter__(self):
                for labels in (torch.tensor([[True, False]]), torch.tensor([[False, True]])):
                    yield {"inputs": {"cell_valid": torch.tensor([[True]])}, "targets": {"identity_allowed": labels[None], "identity_valid": torch.tensor([[True]])}}
        weights = MODULE.identity_patch_sampling_weights(Dummy(), torch.tensor([.1, 4.]))
        self.assertGreater(weights[1], weights[0])
