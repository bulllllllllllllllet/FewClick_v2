import unittest
import torch
from benchmarks.v5.phase_5_prompt_encoder.src.paired_training import (shuffled_order, mixed_cross_mask,
    rejection_task_masks, absent_query_target, state_hash, support_counts)


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

    def test_rejection_masks_are_disjoint_and_fixed(self):
        absent, cross = rejection_task_masks(100, .10, .15)
        self.assertEqual(int(absent.sum()), 10)
        self.assertEqual(int(cross.sum()), 15)
        self.assertFalse(bool((absent & cross).any()))
        again = rejection_task_masks(100, .10, .15)
        self.assertTrue(torch.equal(torch.from_numpy(absent), torch.from_numpy(again[0])))
        smoke_absent, smoke_cross = rejection_task_masks(4, .10, .15)
        self.assertEqual(int(smoke_absent.sum()), 1)
        self.assertEqual(int(smoke_cross.sum()), 1)

    def test_absent_target_preserves_ignore_and_removes_prompt_exclusion(self):
        item = dict(binary_target=torch.tensor([1, 0, 255]),
                    all_prompted_regions=torch.tensor([True, False, True]))
        result = absent_query_target(item)
        self.assertTrue(torch.equal(result['binary_target'], torch.tensor([0, 0, 255])))
        self.assertTrue(torch.equal(result['all_prompted_regions'], torch.tensor([False, False, False])))
