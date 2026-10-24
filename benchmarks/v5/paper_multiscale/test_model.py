"""Physical mapping, identity initialization, and sparse topology regression tests."""
import unittest
import pyvips
import torch

from .model import overlap_edges,edge_mask,PaperHierarchy
from .data import intersects


class PhysicalHierarchyTests(unittest.TestCase):
    def test_center_field_ignores_outer_parent_quadrants(self):
        child=torch.ones(1,1,8,8)
        parent=torch.zeros(1,2,32,32);parent[:,1]=1
        parent[:,0,8:24,8:24]=1;parent[:,1,8:24,8:24]=0
        ids,weights=overlap_edges(child,parent,1)
        self.assertEqual(int(ids[0,0,0]),0)
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(1,1)))

    def test_padding_cannot_erase_parent_zero(self):
        mask=edge_mask(torch.tensor([[[0,-1,2]]]),3)
        self.assertEqual(mask.tolist(),[[[True,False,True]]])

    def test_zero_gate_is_identity_but_gate_can_learn(self):
        torch.manual_seed(7);h=PaperHierarchy(8)
        tokens=[torch.randn(1,4,8) for _ in range(3)]
        edges=(torch.arange(4).reshape(1,4,1),torch.ones(1,4,1))
        output=h(tokens,edges,edges)
        self.assertTrue(torch.equal(output[0],tokens[0]))
        output[0].square().sum().backward()
        self.assertIsNotNone(h.gate.grad)
        self.assertGreater(abs(float(h.gate.grad)),0)

    def test_context_touching_is_not_overlap(self):
        self.assertFalse(intersects([0,0,4,4],[4,0,8,4]))
        self.assertTrue(intersects([0,0,4,4],[3,0,8,4]))


if __name__=='__main__':unittest.main()
