import unittest, torch
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss

class ObservedEvidenceTest(unittest.TestCase):
 def test_panel_specific_heads_and_coverage_weighted_loss(self):
  heads=ObservedRegionEvidenceHeads(region_dim=8,structure_classes=3,immune_classes=2); x=torch.randn(1,4,8,requires_grad=True); out=heads(x)
  self.assertEqual(out['structure_observed_logits'].shape,(1,4,3)); self.assertEqual(out['immune_observed_logits'].shape,(1,4,2))
  counts=torch.tensor([[[2.,1.,0.],[0.,0.,0.],[3.,0.,1.],[1.,1.,1.]]]); observed=torch.tensor([[3.,7.,4.,3.]]); coverage=torch.tensor([[1.,1.,.5,0.]])
  loss,n=observed_composition_loss(out['structure_observed_logits'],counts,observed,coverage); self.assertEqual(n,2); loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())
if __name__=='__main__': unittest.main()
