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

    def test_patch_sampling_upweights_patch_containing_rare_class(self):
        class Dummy:
            spec = {"identity_classes": ["a", "b"]}
            def __len__(self): return 2
            def __iter__(self):
                for labels in (torch.tensor([[True, False]]), torch.tensor([[False, True]])):
                    yield {"inputs": {"cell_valid": torch.tensor([[True]])}, "targets": {"identity_allowed": labels[None], "identity_valid": torch.tensor([[True]])}}
        weights = MODULE.identity_patch_sampling_weights(Dummy(), torch.tensor([.1, 4.]))
        self.assertGreater(weights[1], weights[0])
