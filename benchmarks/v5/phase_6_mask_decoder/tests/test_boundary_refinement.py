import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.boundary_refinement import BoundaryPixelRefiner, refinement_loss
from benchmarks.v5.phase_6_mask_decoder.probe_boundary_refinement import metrics, prompt_violations


class BoundaryTests(unittest.TestCase):
    def test_zero_initialization_and_gate_protection(self):
        model = BoundaryPixelRefiner(radius=2)
        image = torch.randn(1,3,32,32)
        p = torch.full((1,32,32),.05); p[:,:,16:] = .95
        out = model(image,p)
        self.assertTrue(torch.allclose(out['pixel_probability'],p,atol=1e-6))
        with torch.no_grad(): model.net[-1].bias.fill_(10)
        out = model(image,p)
        self.assertTrue(torch.equal(out['pixel_probability'][~out['gate']],p[~out['gate']]))
        self.assertLessEqual(float(out['residual'].detach().abs().max()),2.)

    def test_missing_prediction_boundary_is_penalized(self):
        truth = torch.zeros(1,32,32,dtype=torch.bool); truth[:,:,16:]=True
        valid = torch.ones_like(truth)
        row=metrics(torch.zeros(1,32,32),truth,valid)
        self.assertEqual(row['bf1_2'],0.)
        self.assertEqual(row['boundary_band_iou_2'],0.)
        row=metrics(truth.float(),truth,valid)
        self.assertEqual(row['bf1_2'],1.)
        self.assertEqual(row['boundary_band_iou_2'],1.)

    def test_ignore_edges_do_not_create_boundaries(self):
        truth=torch.zeros(1,32,32,dtype=torch.bool); truth[:,:,16:]=True
        valid=torch.ones_like(truth); valid[:,:,14:18]=False
        self.assertIsNone(metrics(truth.float(),truth,valid)['bf1_2'])

    def test_prompt_coordinate_convention(self):
        p=torch.zeros(1,10,10); p[0,2,3]=1
        batch={'positive_xy':torch.tensor([[[.3,.2]]]),'negative_xy':torch.tensor([[[.8,.8]]]),
               'positive_mask':torch.ones(1,1,dtype=torch.bool),'negative_mask':torch.ones(1,1,dtype=torch.bool)}
        self.assertEqual(prompt_violations(p,batch),0)

    def test_real_training_loss_backpropagates(self):
        model=BoundaryPixelRefiner()
        p=torch.full((1,32,32),.4); p[:,:,16:]=.6
        truth=torch.zeros(1,32,32,dtype=torch.bool); truth[:,:,17:]=True
        loss=refinement_loss(model(torch.randn(1,3,32,32),p),truth,torch.ones_like(truth),1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()),0.)


if __name__=='__main__': unittest.main()
