import unittest
import torch
from benchmarks.v5.phase_5_prompt_encoder.src.learnable_tissue_prototype import LearnableTissuePrototype, unprompted_bce


class LearnablePrototypeTests(unittest.TestCase):
    def test_backward_and_no_query_label_dependency(self):
        torch.manual_seed(1)
        batch = dict(fine_tokens=torch.randn(2, 6, 8), positive_tokens=torch.randn(2, 2, 8),
                     negative_tokens=torch.randn(2, 2, 8), positive_mask=torch.ones(2, 2, dtype=torch.bool),
                     negative_mask=torch.ones(2, 2, dtype=torch.bool), fine_active=torch.ones(2, 6, dtype=torch.bool),
                     all_prompted_regions=torch.zeros(2, 6, dtype=torch.bool), binary_target=torch.zeros(2, 6))
        for mode in ('single', 'multi'):
            model = LearnableTissuePrototype(mode, 8)
            logits = model(batch)['logits']
            altered = dict(batch, binary_target=torch.ones(2, 6))
            torch.testing.assert_close(logits, model(altered)['logits'])
            unprompted_bce(logits, batch).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_support_and_ignore_excluded(self):
        batch = dict(binary_target=torch.tensor([[1., 255., 0.]]), fine_active=torch.ones(1, 3, dtype=torch.bool),
                     all_prompted_regions=torch.tensor([[True, False, False]]))
        logits = torch.zeros(1, 3, requires_grad=True)
        unprompted_bce(logits, batch).backward()
        self.assertEqual(logits.grad[0, 0], 0)
        self.assertEqual(logits.grad[0, 1], 0)
        self.assertGreater(logits.grad[0, 2], 0)
