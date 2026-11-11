"""A: predicted support-region descriptors; unchanged seven-channel matching head."""
import torch
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Parent,DecoderData as ParentData,SignedMatcher
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state
def pool4(x):
 h,w=x.shape[-2:]
 return torch.stack([x[...,i*h//4:((i+1)*h+3)//4,j*w//4:((j+1)*w+3)//4].mean((-2,-1)) for i in range(4) for j in range(4)],-1).reshape(*x.shape[:-2],4,4)
ARMS=('region_evidence',)
class DecoderData(ParentData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ['region_evidence','match'];super().__init__(cache,'match',seed)
class RegionMatcher(SignedMatcher):
 def forward(self,support,query,prompts,base,source_probability=None):
  if source_probability is None:return super().forward(support,query,prompts,base)
  se=self.project(support.float());qe=F.normalize(self.project(query.float()),dim=1)
  p=F.interpolate(source_probability.detach()[:,None],size=se.shape[-2:],mode='bilinear',align_corners=False)
  evidence=[]
  for sign,weight in [('positive',(2*p-1).clamp_min(0).square()),('negative',(1-2*p).clamp_min(0).square())]:
   mass=pool4(weight).flatten(2)
   regional=pool4(se*weight).flatten(2)/mass.clamp_min(1e-8)
   point=sample_assignment_at_cells(se,prompts[sign+'_xy'].float())
   descriptors=F.normalize(torch.cat([point,regional],2),dim=1)
   valid=torch.cat([prompts[sign+'_mask'].bool(),mass[:,0]>0],1)
   assert valid.any(1).all()
   sim=torch.einsum('bcp,bchw->bphw',descriptors,qe)
   evidence.extend([sim.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True),(sim*valid[:,:,None,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None,None]])
  probability=F.interpolate(base['pixel_probability'][:,None],size=qe.shape[-2:],mode='bilinear',align_corners=False)
  features=torch.cat([*evidence,evidence[0]-evidence[2],probability,4*probability*(1-probability)],1)
  residual=F.interpolate(self.output(self.hidden(features)),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
  logits=base['pixel_logits']+residual
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),residual=residual)
class EvidenceModel(Parent):
 def __init__(self,seed=20261103,device='cuda:0'):
  super().__init__('match',seed,device)
  # No parameters or initialization change: bind same registered weights to extended computation.
  self.pixel.__class__=RegionMatcher
 def cached_decode(self,se,qe,prompts,local=False):
  task=self.base.core.support_task(se,prompts)
  source=self.base.core.decode(se,task,prompts,True)[0]
  source=self.pixel(se['pixel_embedding'],se['pixel_embedding'],prompts,source)
  if local:return source,source,task,None
  out,maps=self.base.core.decode(qe,task,prompts,False)
  out=self.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,out,source['pixel_probability'])
  return out,source,task,maps
 def forward(self,support,query,prompts,local,shared,with_support=True):
  support=self.base.transform(support);query=support if shared else self.base.transform(query)
  se=self.encode(support);qe=se if shared else self.encode(query)
  out,source,task,maps=self.cached_decode(se,qe,prompts,local)
  if maps is None:maps=out['pixel_probability'].new_zeros(out['pixel_probability'].shape[0],2,128,128)
  reg=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
  return dict(query=out,support=source if with_support else None,task_token=task['task_token'],query_prompt_maps=maps,assignment_regularization=reg)
def DecoderModel(arm,seed=20261103,device='cuda:0'):
 assert arm=='region_evidence';return EvidenceModel(seed,device)
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
 return m
