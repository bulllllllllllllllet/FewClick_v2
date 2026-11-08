import os,sys,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype
ROOT=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/additional_sam_1108_135227')
def rgb(image):
 v=image[0].permute(1,2,0).cpu().numpy();return ((v*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255).round().clip(0,255).astype('uint8')
class PublicFeatureReference(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='wsi_sam_public':
   from benchmarks.v5.fair_comparison.adapted_models import WSISAMFineTune
   self.wrapped=WSISAMFineTune('cuda:0');self.identity='WSI-SAM public two-view encoder + signed prototype, no target adaptation'
  elif method=='sam2_public':
   sys.path[:0]=[str(ROOT/'isolated_deps'),str(ROOT/'sam2_source')]
   from sam2.build_sam import build_sam2
   from sam2.sam2_image_predictor import SAM2ImagePredictor
   self.network=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'sam2.1_hiera_large.pt'),device='cuda:0',apply_postprocessing=False);self.predictor=SAM2ImagePredictor(self.network);self.identity='SAM2.1 Hiera Large public three-view aligned encoder + signed prototype, no target adaptation'
  else:raise ValueError(method)
  self.requires_grad_(False)
 def encode(self,x):
  if self.method=='wsi_sam_public':
   features=[];native=[]
   for suffix in ['', '_1']:
    image=torch.as_tensor(rgb(x['image'+suffix]),device='cuda:0').permute(2,0,1)[None].float();image=F.interpolate(image,size=(1024,1024),mode='bilinear',align_corners=False).round().clamp(0,255).byte();embedding,intermediate=self.wrapped.sam.image_encoder(self.wrapped.sam.preprocess(image[0])[None]);features.append(embedding.float());native.append((embedding,intermediate))
   fine,low=features;h,w=fine.shape[-2:];yy,xx=torch.meshgrid((torch.arange(h,device=fine.device)+.5)/h,(torch.arange(w,device=fine.device)+.5)/w,indexing='ij');aligned=F.grid_sample(low,torch.stack([xx-.5,yy-.5],-1)[None],align_corners=False);return dict(features=torch.cat([F.normalize(fine,dim=1),F.normalize(aligned,dim=1)],1),native=native)
  values=[]
  for suffix in ['', '_1','_2']:
   self.predictor.set_image(rgb(x['image'+suffix]));values.append(self.predictor.get_image_embedding().clone())
   if suffix=='':fine_features={k:[z.clone() for z in v] if isinstance(v,list) else v.clone() for k,v in self.predictor._features.items()};orig_hw=list(self.predictor._orig_hw)
  assert values[0].shape[-2:]==(64,64)
  fused=(values[0]+F.interpolate(values[1][...,16:48,16:48],size=(64,64),mode='bilinear',align_corners=False)+F.interpolate(values[2][...,24:40,24:40],size=(64,64),mode='bilinear',align_corners=False))/3
  return dict(features=fused,native=fine_features,orig_hw=orig_hw)
 def decode(self,se,qe,prompts):
  prior=dense_prototype(se['features'],qe['features'],prompts,temperature=.1)['pixel_logits']
  dense_mask=F.interpolate(prior[:,None],size=(256,256),mode='bilinear',align_corners=False)
  if self.method=='sam2_public':
   self.predictor._features=qe['native'];self.predictor._orig_hw=qe['orig_hw'];self.predictor._is_image_set=True;self.predictor._is_batch=False
   masks,qualities,_=self.predictor.predict(mask_input=dense_mask[0].cpu().numpy(),multimask_output=True,return_logits=True)
   logits=torch.as_tensor(masks[int(np.argmax(qualities))],device='cuda:0')[None]
  else:
   outputs=[]
   low_mask=torch.zeros_like(dense_mask);low_mask[...,64:192,64:192]=F.interpolate(dense_mask,size=(128,128),mode='bilinear',align_corners=False)
   for index,(embedding,intermediate) in enumerate(qe['native']):
    sparse,dense=self.wrapped.sam.prompt_encoder(points=None,boxes=None,masks=dense_mask if index==0 else low_mask)
    net=self.wrapped.net_high if index==0 else self.wrapped.net_low
    outputs.append(net(image_embeddings=embedding,image_pe=[self.wrapped.sam.prompt_encoder.get_dense_pe()],sparse_prompt_embeddings=[sparse],dense_prompt_embeddings=[dense],multimask_output=False,wsi_token_only=False,interm_embeddings=intermediate))
   mask,_,token,features=outputs[0];lo_token=outputs[1][2];logits=((token+lo_token)@features).view(1,-1,256,256)+mask;logits=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False)[:,0]
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())
 def forward(self,support,query,prompts,local,shared,with_support=False):
  assert not local and not shared
  return dict(query=self.decode(self.encode(support),self.encode(query),prompts))
def build(method):return PublicFeatureReference(method)
