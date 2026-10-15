import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.point_objective import inference_composite, objective


class PointObjectiveTests(unittest.TestCase):
    def case(self):
        truth = torch.zeros(1, 32, 32, dtype=torch.bool)
        truth[:, :, 16:] = True
        valid = torch.ones_like(truth)
        valid[:, 4, 4] = False
        # Two training anchors, then GT-independent rendering positions.
        indices = torch.tensor([[2*32+2, 2*32+30, 15*32+15, 15*32+16, 15*32+4, 4*32+4]])
        logits = torch.tensor([[-.2, .3, -.4, .5, .7, -.8]], requires_grad=True)
        output = {'coarse_probability': torch.full((1, 32, 32), .4), 'indices': indices, 'point_logits': logits}
        return output, {'truth': truth, 'valid': valid}

    def test_composite_excludes_training_anchors_and_preserves_outside(self):
        output, sample = self.case()
        probability, _ = inference_composite(output, 2)
        selected = torch.zeros(1, 1024, dtype=torch.bool).scatter(1, output['indices'][:, 2:], True).reshape(1, 32, 32)
        self.assertTrue(torch.equal(probability[~selected], output['coarse_probability'][~selected]))
        probability.sum().backward()
        self.assertTrue(torch.equal(output['point_logits'].grad[:, :2], torch.zeros(1, 2)))
        self.assertTrue((output['point_logits'].grad[:, 2:] > 0).all())

    def test_added_boundary_supervision_has_only_valid_band_gradients(self):
        output, sample = self.case()
        control, _ = objective(output, sample, 2, 'control')
        boundary, parts = objective(output, sample, 2, 'boundary')
        self.assertAlmostEqual(float((boundary-control).detach()), .5*parts['boundary_bce'], places=6)
        (boundary-control).backward()
        grad = output['point_logits'].grad
        self.assertGreater(abs(float(grad[0, 2])), 0)
        self.assertGreater(abs(float(grad[0, 3])), 0)
        self.assertTrue(torch.equal(grad[0, [0, 1, 4, 5]], torch.zeros(4)))

    def test_whole_image_dice_reaches_rendered_nonboundary_and_ignores_invalid(self):
        output, sample = self.case()
        boundary, _ = objective(output, sample, 2, 'boundary')
        global_loss, _ = objective(output, sample, 2, 'global')
        (global_loss-boundary).backward()
        grad = output['point_logits'].grad
        self.assertGreater(abs(float(grad[0, 4])), 0)
        self.assertEqual(float(grad[0, 5]), 0)
        self.assertTrue(torch.isfinite(grad).all())


if __name__ == '__main__':
    unittest.main()
