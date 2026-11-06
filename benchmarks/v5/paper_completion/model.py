"""Controlled matcher ablations; identical base and fixed training recipe."""
import torch
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Original, SignedMatcher
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
class DecoderData(SpatialData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ['no_match','no_negative'];super().__init__(cache,'I',seed)
class PositiveMatcher(SignedMatcher):
 def forward(self,support,query,prompts,base):
  se=self.project(support.float());qe=F.normalize(self.project(query.float()),dim=1)
  valid=prompts['positive_mask'].bool();assert valid.any(1).all()
  desc=F.normalize(sample_assignment_at_cells(se,prompts['positive_xy'].float()),dim=1)
  sim=torch.einsum('bcp,bchw->bphw',desc,qe)
  maximum=sim.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True)
  mean=(sim*valid[:,:,None,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None,None]
  zero=torch.zeros_like(maximum)
  prob=F.interpolate(base['pixel_probability'][:,None],size=qe.shape[-2:],mode='bilinear',align_corners=False)
  features=torch.cat([maximum,mean,zero,zero,maximum,prob,4*prob*(1-prob)],1)
  residual=F.interpolate(self.output(self.hidden(features)),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
  logits=base['pixel_logits']+residual
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),residual=residual)
class DecoderModel(Original):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  assert arm in ['no_match','no_negative']
  super().__init__('control' if arm=='no_match' else 'match',seed,device);self.arm=arm
  if arm=='no_negative':
   replacement=PositiveMatcher().to(device);replacement.load_state_dict(self.pixel.state_dict());self.pixel=replacement
