import unittest

import torch

from benchmarks.v5.phase_5_prompt_encoder.src.multi_prototype import (
    PositiveNegativePrototypeMatcher,
    normalized_lme_similarity,
)


class MultiPrototypeTest(unittest.TestCase):
    def test_duplicate_prototype_does_not_change_normalized_score(self):
        query = torch.tensor([[[1.0, 0.0]]])
        one = torch.tensor([[[1.0, 0.0]]])
        two = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        first = normalized_lme_similarity(query, one, torch.tensor([[True]]))
        second = normalized_lme_similarity(query, two, torch.tensor([[True, True]]))
        torch.testing.assert_close(first, second)

    def test_positive_negative_matcher_has_finite_gradients(self):
        torch.manual_seed(7)
        query = torch.randn(2, 4, 8, requires_grad=True)
        positive = torch.randn(2, 3, 8, requires_grad=True)
        negative = torch.randn(2, 2, 8, requires_grad=True)
        matcher = PositiveNegativePrototypeMatcher(learnable_temperature=True)
        output = matcher(query, positive, torch.tensor([[True, True, False], [True, False, False]]),
                         negative, torch.tensor([[True, False], [True, True]]))
        self.assertEqual(output["logits"].shape, (2, 4))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        output["logits"].square().mean().backward()
        for value in (query.grad, positive.grad, negative.grad, matcher.log_temperature.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())

    def test_empty_prototype_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalized_lme_similarity(torch.ones(1, 1, 2), torch.ones(1, 1, 2), torch.tensor([[False]]))


if __name__ == "__main__":
    unittest.main()
