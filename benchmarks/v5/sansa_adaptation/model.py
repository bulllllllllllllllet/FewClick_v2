import json,sys,os
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
ROOT=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/reference_baseline_1109_040918')
ARMS=('sansa_ft',)
class DecoderData(SpatialData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ARMS;super().__init__(cache,'I',seed)
class DecoderModel(nn.Module):
 def __init__(self,arm='sansa_ft',seed=20261103,device='cuda:0'):
  super().__init__();assert arm in ARMS and device=='cuda:0';torch.manual_seed(seed);sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'SANSA')]
  from models.sansa.sansa import build_sansa
  from util.path_utils import SAM2_PATHS_CONFIG
  w=json.load(open(ROOT/'weights.json'));SAM2_PATHS_CONFIG['large']=(w['weights']['sam2_hiera_large.pt']['path'],SAM2_PATHS_CONFIG['large'][1]);self.network=build_sansa(channel_factor=.8,device=device).to(device)
  state=torch.load(w['weights']['sansa_universal.pth']['path'],map_location='cpu',weights_only=False)['model'];missing,extra=self.network.load_state_dict(state,strict=False);assert not extra and not any('adapter' in k for k in missing)
  self.arm=arm;self.network.requires_grad_(False)
  for n,p in self.network.named_parameters():p.requires_grad_('adapter' in n)
  assert not self.network.sam.pred_obj_scores
 @property
 def base(self):return self.network.sam
 def train(self,mode=True):
  super().train(mode);self.network.eval();return self
 def encode(self,inputs):
  # Same fine512 RGB reconstruction, PIL640 and GPU normalization as public pilot.
  x=inputs['image'][0].detach().permute(1,2,0).cpu().numpy();rgb=((x*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255).round().clip(0,255).astype('uint8');resized=np.asarray(Image.fromarray(rgb).resize((640,640),Image.Resampling.BILINEAR)).copy();v=torch.from_numpy(resized).permute(2,0,1).to(inputs['image'].device).float()/255;return (v-v.new_tensor([.485,.456,.406])[:,None,None])/v.new_tensor([.229,.224,.225])[:,None,None]
 def _run(self,se,qe,prompts,local):
  coords=[];labels=[]
  for sign,label in [('positive',1),('negative',0)]:
   mask=prompts[sign+'_mask'][0].bool();coords.append(prompts[sign+'_xy'][0][mask]);labels.extend([label]*int(mask.sum()))
  xy=torch.cat(coords)[None]*640;lab=torch.tensor([labels],device=xy.device,dtype=torch.int32);prompt={0:{0:dict(prompt_type='point',prompt=dict(point_coords=xy,point_labels=lab))},'shots':1};video=(se[None] if local else torch.stack([se,qe]))[None]
  logits=self.network(video,prompt)['pred_masks'];assert len(logits)==(1 if local else 2);logits=F.interpolate(logits[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0]
  part=lambda z:dict(pixel_logits=z[None],pixel_probability=z[None].sigmoid());return part(logits[0] if local else logits[1]),part(logits[0])
 def decode(self,se,qe,prompts,local=False):return self._run(se,qe,prompts,local)[0]
 def forward(self,support,query,prompts,local,shared,with_support=True):
  se=self.encode(support);qe=se if shared else self.encode(query);q,s=self._run(se,qe,prompts,local);return dict(query=q,support=s if with_support else None,assignment_regularization=q['pixel_logits'].new_zeros(()),query_prompt_maps=q['pixel_logits'].new_zeros(1,2,128,128),task_token=se.mean((1,2))[None])
def frozen_backbone_state(m):return {n:v for n,v in m.base.image_encoder.state_dict().items() if 'adapter' not in n}
def frozen_decoder_state(m):return m.base.sam_mask_decoder.state_dict()
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True)['model'])
 return m
