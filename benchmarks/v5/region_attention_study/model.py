"""B spatial signed support memory; C adds query refinement. Public fresh init."""
import math,torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.region_evidence_study.model import EvidenceModel,pool4
from benchmarks.v5.support_match_study.model import DecoderData as ParentData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state
ARMS=('B','C')
class DecoderData(ParentData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ARMS;super().__init__(cache,'match',seed)
class SpatialMemory(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Conv2d(256,64,1);self.sign=nn.Embedding(2,64);self.geometry=nn.Linear(2,64);self.query=nn.Linear(64,64);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64)
  self.head=nn.Sequential(nn.Conv2d(128,64,1),nn.GELU(),nn.Conv2d(64,1,1));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,se,qe,prompts,p):
  sf=self.project(se.float());qf=self.project(qe.float());p=F.interpolate(p.detach()[:,None],size=sf.shape[-2:],mode='bilinear',align_corners=False);tokens=[];valids=[]
  grid=sf.new_tensor([[(x+.5)/4,(y+.5)/4] for y in range(4) for x in range(4)])
  for sign,idx,w in [('positive',0,(2*p-1).clamp_min(0).square()),('negative',1,(1-2*p).clamp_min(0).square())]:
   mass=pool4(w).flatten(2);regional=(pool4(sf*w).flatten(2)/mass.clamp_min(1e-8)).transpose(1,2);point=sample_assignment_at_cells(sf,prompts[sign+'_xy'].float()).transpose(1,2)
   xy=torch.cat([prompts[sign+'_xy'],grid[None].expand(sf.shape[0],-1,-1)],1);tokens.append(torch.cat([point,regional],1)+self.sign.weight[idx]+self.geometry(xy));valids.append(torch.cat([prompts[sign+'_mask'].bool(),mass[:,0]>0],1))
  memory=torch.cat(tokens,1);valid=torch.cat(valids,1);b,_,h,w=qf.shape
  q=self.query(qf.flatten(2).transpose(1,2)).reshape(b,h*w,4,16).transpose(1,2);k=self.key(memory).reshape(b,-1,4,16).transpose(1,2);v=self.value(memory).reshape(b,-1,4,16).transpose(1,2)
  attn=(q@k.transpose(-1,-2)/4).masked_fill(~valid[:,None,None,:],float('-inf')).softmax(-1);context=(attn@v).transpose(1,2).reshape(b,h*w,64).transpose(1,2).reshape(b,64,h,w)
  return self.head(torch.cat([qf,context],1))[:,0],context
class QueryRefiner(nn.Module):
 def __init__(self):
  super().__init__();self.image=nn.Conv2d(256,32,1);self.context=nn.Conv2d(64,32,1);self.head=nn.Sequential(nn.Conv2d(69,32,3,padding=1),nn.GELU(),nn.Conv2d(32,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,encoded,context,prob):
  resize=lambda x:F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
  p=resize(prob[:,None]);x=torch.cat([resize(self.image(encoded['pixel_embedding'].float())),resize(self.context(context)),resize(encoded['pixel_rgb'].float()),p,4*p*(1-p)],1)
  return self.head(x)[:,0]
class AttentionModel(EvidenceModel):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  super().__init__(seed,device);self.arm=arm
  devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(seed+1201);self.memory=SpatialMemory().to(device)
   self.refiner=QueryRefiner().to(device) if arm=='C' else None
 def refine(self,se,qe,prompts,source,base):
  delta,context=self.memory(se['pixel_embedding'],qe['pixel_embedding'],prompts,source['pixel_probability']);logits=base['pixel_logits']+F.interpolate(delta[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0]
  if self.refiner is not None:logits=logits+F.interpolate(self.refiner(qe,context,logits.sigmoid())[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0]
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())
 def cached_decode(self,se,qe,prompts,local=False):
  out,source,task,maps=super().cached_decode(se,qe,prompts,local)
  target=self.refine(se,qe,prompts,source,out)
  supporting=target if local else self.refine(se,se,prompts,source,source)
  return target,supporting,task,maps
 def forward(self,*args,**kwargs):return super().forward(*args,**kwargs)
def DecoderModel(arm,seed=20261103,device='cuda:0'):
 assert arm in ARMS;return AttentionModel(arm,seed,device)
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
 return m
