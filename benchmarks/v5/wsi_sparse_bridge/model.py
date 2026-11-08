import torch
from torch.nn import functional as F
from benchmarks.v5.wsi_author_freeze.model import DecoderModel as DenseOnly,DecoderData,frozen_decoder_state
ARMS=('wsi_sam_ft',)

def auto_points(dense):
 assert dense.shape==(1,1,256,256)
 probability=F.interpolate(dense.detach().float(),size=(64,64),mode='bilinear',align_corners=False)[0,0].sigmoid();available=torch.ones_like(probability,dtype=torch.bool);coords=[];labels=[];scores=[]
 for label,limit in [(1,3),(0,2)]:
  if label==0 and not coords:break
  for _ in range(limit):
   candidate=probability.masked_fill(~available,-1 if label else 2);idx=int(candidate.argmax() if label else candidate.argmin());y,x=divmod(idx,64);value=float(probability[y,x])
   if not bool(available[y,x]) or (value<.7 if label else value>.3):break
   coords.append([(x+.5)/64,(y+.5)/64]);labels.append(label);scores.append(value);available[max(0,y-4):min(64,y+5),max(0,x-4):min(64,x+5)]=False
 if not coords:return None,dict(xy=[],labels=[],scores=[])
 xy=torch.tensor([coords],device=dense.device,dtype=torch.float32)*1024;lab=torch.tensor([labels],device=dense.device,dtype=torch.long);return (xy,lab),dict(xy=coords,labels=labels,scores=scores)

class DecoderModel(DenseOnly):
 def __init__(self,arm='wsi_sam_ft',seed=20261103,device='cuda:0'):
  super().__init__(arm,seed,device);self.auto_enabled=True;self.last_auto_points=None
 def decode(self,se,qe,prompts,local=False):
  if local or not self.auto_enabled:return super().decode(se,qe,prompts,local)
  state=dict(call=0,points=None)
  def inject(module,args,kwargs):
   assert kwargs['points'] is None
   if state['call']==0:state['points'],self.last_auto_points=auto_points(kwargs['masks'])
   p=state['points']
   if p is not None:
    xy,labels=p;kwargs=dict(kwargs,points=(xy if state['call']==0 else xy/2+256,labels))
   state['call']+=1;return args,kwargs
  handle=self.wrapped.sam.prompt_encoder.register_forward_pre_hook(inject,with_kwargs=True)
  try:result=super().decode(se,qe,prompts,local)
  finally:handle.remove()
  assert state['call']==2
  return result

def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True)['model'])
 return m
