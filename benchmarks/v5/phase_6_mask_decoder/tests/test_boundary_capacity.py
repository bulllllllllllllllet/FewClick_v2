import unittest

import torch

from benchmarks.v5.phase_6_mask_decoder.audit_boundary_capacity import fit_coefficients


class CapacityFitTests(unittest.TestCase):
    def test_exact_fit_ignores_unscored_pixels(self):
        assignment = torch.tensor([[[[1., 0., 1.]], [[0., 1., 0.]]]])
        truth = torch.tensor([[[True, False, False]]])
        valid = torch.tensor([[[True, True, False]]])
        probability, report = fit_coefficients(assignment, truth, valid, torch.tensor([[.2, .8]]))
        self.assertTrue(torch.allclose(probability, torch.tensor([[[1., 0., 1.]]]), atol=1e-6))
        self.assertTrue(report['success'])
        self.assertGreater(report['objective_reduction'], 0.)

    def test_unrepresentable_detail_is_not_fabricated(self):
        assignment = torch.ones(1, 1, 2, 2)
        truth = torch.tensor([[[True, False], [False, False]]])
        valid = torch.ones_like(truth)
        probability, report = fit_coefficients(assignment, truth, valid, torch.tensor([[.8]]))
        self.assertTrue(torch.allclose(probability, torch.full((1, 2, 2), .25), atol=1e-6))
        self.assertLess(report['projected_gradient_max'], 1e-6)


if __name__ == '__main__':
    unittest.main()
