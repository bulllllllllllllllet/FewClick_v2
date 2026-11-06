"""One-variable change: point descriptors versus click-guided predicted-support pooling."""
import torch
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Original,SignedMatcher
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
ARMS=('control','region')
class DecoderData(SpatialData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ARMS
  super().__init__(cache,'I',seed)
class RegionMatcher(SignedMatcher):
 def forward(self,support,query,prompts,base,support_probability):
  se=self.project(support.float());sn=F.normalize(se,dim=1);qe=F.normalize(self.project(query.float()),dim=1)
  p=F.interpolate(support_probability.detach()[:,None],size=se.shape[-2:],mode='bilinear',align_corners=False)[:,0];evidence=[]
  for sign in ['positive','negative']:
   valid=prompts[sign+'_mask'].bool();assert valid.any(1).all()
   points=F.normalize(sample_assignment_at_cells(se,prompts[sign+'_xy'].float()),dim=1)
   affinity=torch.einsum('bcp,bchw->bphw',points,sn)
   weights=(affinity/.1).flatten(2).softmax(-1)*(p if sign=='positive' else 1-p).flatten(1)[:,None]
   pooled=torch.einsum('bpn,bcn->bcp',weights,se.flatten(2))/weights.sum(-1)[:,None].clamp_min(1e-8)
   descriptors=F.normalize(.5*points+.5*F.normalize(pooled,dim=1),dim=1)
   sim=torch.einsum('bcp,bchw->bphw',descriptors,qe)
   evidence.extend([sim.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True),(sim*valid[:,:,None,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None,None]])
  probability=F.interpolate(base['pixel_probability'][:,None],size=qe.shape[-2:],mode='bilinear',align_corners=False)
  features=torch.cat([*evidence,evidence[0]-evidence[2],probability,4*probability*(1-probability)],1)
  residual=F.interpolate(self.output(self.hidden(features)),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0];logits=base['pixel_logits']+residual
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),residual=residual)
class DecoderModel(Original):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  assert arm in ARMS
  super().__init__('match',seed,device);self.study_arm=arm
  if arm=='region':
   replacement=RegionMatcher().to(device);replacement.load_state_dict(self.pixel.state_dict());self.pixel=replacement
 def forward(self,support,query,prompts,local,shared,with_support=True):
  if self.study_arm=='control':return super().forward(support,query,prompts,local,shared,with_support)
  support=self.base.transform(support);query=support if shared else self.base.transform(query)
  se=self.encode(support);qe=se if shared else self.encode(query);task=self.base.core.support_task(se,prompts)
  output,maps=self.base.core.decode(qe,task,prompts,local)
  source=output if shared else self.base.core.decode(se,task,prompts,True)[0]
  support_p=source['pixel_probability']
  output=self.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,output,support_p)
  source=output if shared else self.pixel(se['pixel_embedding'],se['pixel_embedding'],prompts,source,support_p) if with_support else None
  regularization=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
  return dict(query=output,support=source,query_prompt_maps=maps,task_token=task['task_token'],assignment_regularization=regularization)
