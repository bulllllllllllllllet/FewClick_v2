"""Only the frozen image encoder changes; fixed norm-preserving channel lift."""
import math,sys
import torch
from torch import nn
from benchmarks.v5.support_match_study.model import DecoderModel as Parent,DecoderData as Data
from benchmarks.v5.additional_sam.model import ROOT,rgb
ARMS=('sam2_encoder',)
class Backbone(nn.Module):
 def __init__(self):
  super().__init__();sys.path[:0]=[str(ROOT/'isolated_deps'),str(ROOT/'sam2_source')]
  from sam2.build_sam import build_sam2
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  network=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'sam2.1_hiera_large.pt'),device='cuda:0',apply_postprocessing=False)
  predictor=SAM2ImagePredictor(network);self.transform=predictor._transforms;self.encoder=network.image_encoder.eval().requires_grad_(False)
 @torch.no_grad()
 def forward(self,image):
  tensor=self.transform(rgb(image)).to(image.device)[None];v=self.encoder(tensor)['vision_features'];assert v.shape[1:]==(256,64,64)
  # No learned candidate-only projection. Match the existing 2048-channel head.
  return {'out':v.repeat(1,8,1,1)/math.sqrt(8)}
class DecoderModel(Parent):
 def __init__(self,arm='sam2_encoder',seed=20261103,device='cuda:0'):
  assert arm in ARMS
  super().__init__('match',seed,device)
  self.arm=arm;backbone=Backbone()
  for encoder in self.base.core.encoders:encoder.backbone=backbone
class DecoderData(Data):
 def __init__(self,cache,arm,seed=20261103):super().__init__(cache,'match',seed)
def frozen_backbone_state(m):return {k:v for k,v in m.state_dict().items() if '.backbone.' in k}
def frozen_decoder_state(m):return {}
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
 return m
