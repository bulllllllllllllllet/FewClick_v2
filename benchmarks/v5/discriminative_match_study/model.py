"""Original inference, training-only matching projection features with detached inputs."""
import torch
from benchmarks.v5.support_match_study.model import DecoderModel as Parent,DecoderData as ParentData
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state
ARMS=('D',)
class MatchModel(Parent):
 def __init__(self,seed=20261103,device='cuda:0'):super().__init__('match',seed,device)
 def forward(self,*args,**kwargs):
  if not torch.is_grad_enabled():return super().forward(*args,**kwargs)
  inputs=[];handle=self.pixel.project.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach()))
  try:out=super().forward(*args,**kwargs)
  finally:handle.remove()
  assert len(inputs)>=2
  out['matching_aux_support']=self.pixel.project(inputs[0]);out['matching_aux_query']=self.pixel.project(inputs[1]);return out
class DecoderData(ParentData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm=='D';super().__init__(cache,'match',seed)
def DecoderModel(arm,seed=20261103,device='cuda:0'):
 assert arm=='D';return MatchModel(seed,device)
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
 return m
