import unittest
import torch
from benchmarks.v5.phase_3_cell_region.src.spatial_identity import SpatialContextIdentity

class SpatialIdentityTest(unittest.TestCase):
    def test_forward_and_gradients_use_region_and_neighbors(self):
        torch.manual_seed(3); model=SpatialContextIdentity(5,neighbors=2)
        cells=torch.randn(1,4,74); cells[...,:2]=torch.tensor([[[.1,.1],[.2,.1],[.8,.8],[.9,.8]]]); cells[...,3]=1
        mass=torch.rand(1,4,4).softmax(1); out=model(torch.randn(1,4,256),cells,torch.tensor([[True,True,True,True]]),mass)
        self.assertEqual(tuple(out['identity_logits'].shape),(1,4,5)); out['identity_logits'].sum().backward(); self.assertIsNotNone(model.region[1].weight.grad)
