from benchmarks.v5.sam_matched_adaptation.model import DecoderModel as Original,DecoderData
ARMS=('wsi_sam_ft',)
ADDED=('hf_token','hf_mlp','compress_vit_feat','embedding_encoder','embedding_maskfeature')
class DecoderModel(Original):
 def __init__(self,arm='wsi_sam_ft',seed=20261103,device='cuda:0'):
  assert arm=='wsi_sam_ft';super().__init__(arm,seed,device)
  for net in [self.wrapped.net_high,self.wrapped.net_low]:
   net.requires_grad_(False)
   for name in ADDED:getattr(net,name).requires_grad_(True)
   for name,p in net.named_parameters():assert p.requires_grad==name.startswith(tuple(k+'.' for k in ADDED))
def frozen_decoder_state(model):
 return {prefix+'.'+n:v for prefix,net in [('high',model.wrapped.net_high),('low',model.wrapped.net_low)] for n,v in net.state_dict().items() if not n.startswith(tuple(k+'.' for k in ADDED))}
def build(arm,checkpoint=None):
 import torch
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True)['model'])
 return m
