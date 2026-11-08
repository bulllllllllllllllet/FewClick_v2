import torch
from torch.nn import functional as F
from benchmarks.v5.additional_sam.model import PublicFeatureReference
from benchmarks.v5.support_match_study.model import SignedMatcher
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.fair_comparison.adapted_models import points
ARMS=('sam2_ft','wsi_sam_ft')
class DecoderData(SpatialData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ARMS;super().__init__(cache,'I',seed)
class DecoderModel(PublicFeatureReference):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  assert arm in ARMS and device=='cuda:0';torch.manual_seed(seed);super().__init__('sam2_public' if arm=='sam2_ft' else 'wsi_sam_public');self.arm=arm;self.matcher=SignedMatcher().to(device)
  if arm=='sam2_ft':
   self.network.sam_prompt_encoder.requires_grad_(True);self.network.sam_mask_decoder.requires_grad_(True)
   for name in ['conv_s0','conv_s1']:
    if hasattr(self.network.sam_mask_decoder,name):getattr(self.network.sam_mask_decoder,name).requires_grad_(False)
  else:
   self.wrapped.sam.prompt_encoder.requires_grad_(True);self.wrapped.net_high.requires_grad_(True);self.wrapped.net_low.requires_grad_(True)
 @property
 def base(self):return self.network if self.arm=='sam2_ft' else self.wrapped.sam
 def train(self,mode=True):
  super().train(mode)
  if self.arm=='sam2_ft':self.network.image_encoder.eval()
  else:self.wrapped.sam.image_encoder.eval()
  return self
 @torch.no_grad()
 def encode(self,x):
  result=super().encode(x)
  if self.arm=='wsi_sam_ft':
   # Public feature wrapper concatenates fine/aligned2x; matched bridge uses256 channels.
   result['features']=.5*(result['features'][:,:256]+result['features'][:,256:])
  return result
 def decode(self,se,qe,prompts,local=False):
  z=qe['features'].new_zeros(1,512,512);base=dict(pixel_logits=z,pixel_probability=z+.5);prior=self.matcher(se['features'],qe['features'],prompts,base)['pixel_logits'];source=F.normalize(se['features'].float(),dim=1);query=F.normalize(qe['features'].float(),dim=1);evidence=[]
  for sign in ['positive','negative']:
   desc=F.normalize(sample_assignment_at_cells(source,prompts[sign+'_xy']),dim=1);sim=torch.einsum('bcp,bchw->bphw',desc,query);valid=prompts[sign+'_mask'].bool();evidence.append((sim*valid[:,:,None,None]).sum(1)/valid.sum(1)[:,None,None])
  prior=prior+F.interpolate((4*(evidence[0]-evidence[1]))[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0];dense_mask=F.interpolate(prior[:,None],size=(256,256),mode='bilinear',align_corners=False)
  if self.arm=='sam2_ft':
   sparse,dense=self.network.sam_prompt_encoder(points=points(prompts) if local else None,boxes=None,masks=dense_mask)
   masks,_,_,_=self.network.sam_mask_decoder(image_embeddings=qe['native']['image_embed'],image_pe=self.network.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False,repeat_image=False,high_res_features=qe['native']['high_res_feats']);logits=F.interpolate(masks,size=(512,512),mode='bilinear',align_corners=False)[:,0]
  else:
   outputs=[];low=torch.zeros_like(dense_mask);low[...,64:192,64:192]=F.interpolate(dense_mask,size=(128,128),mode='bilinear',align_corners=False)
   for i,(embedding,intermediate) in enumerate(qe['native']):
    sparse,dense=self.wrapped.sam.prompt_encoder(points=points(prompts,low=bool(i)) if local else None,boxes=None,masks=dense_mask if i==0 else low);net=self.wrapped.net_high if i==0 else self.wrapped.net_low;outputs.append(net(image_embeddings=embedding,image_pe=[self.wrapped.sam.prompt_encoder.get_dense_pe()],sparse_prompt_embeddings=[sparse],dense_prompt_embeddings=[dense],multimask_output=False,wsi_token_only=False,interm_embeddings=intermediate))
   mask,_,token,feat=outputs[0];logits=((token+outputs[1][2])@feat).view(1,-1,256,256)+mask;logits=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False)[:,0]
  return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())
 def forward(self,support,query,prompts,local,shared,with_support=True):
  se=self.encode(support);qe=se if shared else self.encode(query);out=self.decode(se,qe,prompts,local);source=out if shared else self.decode(se,se,prompts,True) if with_support else None
  return dict(query=out,support=source,assignment_regularization=out['pixel_logits'].new_zeros(()),query_prompt_maps=out['pixel_logits'].new_zeros(1,2,128,128),task_token=se['features'].mean((2,3)))
def build(arm,checkpoint=None):
 model=DecoderModel(arm)
 if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True)['model'])
 return model
