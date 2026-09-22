import unittest
import torch
from benchmarks.v5.phase_5_prompt_encoder.src.paired_training import shuffled_order, mixed_cross_mask, state_hash, support_counts


class PairedTrainingTests(unittest.TestCase):
    def test_order_is_reproducible_permutation_each_epoch(self):
        order = shuffled_order(36, 77)
        self.assertEqual(order, shuffled_order(36, 77))
        self.assertEqual(sorted(order[:36]), list(range(36)))
        self.assertEqual(sorted(order[36:72]), list(range(36)))
        self.assertNotEqual(order[:36], list(range(36)))

    def test_initialization_hash_detects_change(self):
        model = torch.nn.Linear(2, 2)
        before = state_hash(model)
        with torch.no_grad(): model.weight.add_(1)
        self.assertNotEqual(before, state_hash(model))

    def test_counts_are_actual_not_padded_capacity(self):
        self.assertEqual(support_counts(dict(positive_mask=torch.tensor([True,False,True]),
                                            negative_mask=torch.tensor([True,False]))), (2,1))

    def test_mixed_mask_is_fixed_and_has_requested_fraction(self):
        mask = mixed_cross_mask(100, .25)
        self.assertEqual(int(mask.sum()), 25)
        self.assertTrue(torch.equal(torch.from_numpy(mask), torch.from_numpy(mixed_cross_mask(100,.25))))
