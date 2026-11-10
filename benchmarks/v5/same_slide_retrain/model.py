"""Reuse frozen, previously validated architectures; change task protocol only."""
import torch
from benchmarks.v5.support_match_study.model import DecoderModel as Original,DecoderData as Data
from benchmarks.v5.support_token_repair.model import DecoderModel as Detail
from benchmarks.v5.sansa_adaptation.model import DecoderModel as Sansa
ARMS=('original','detail','sansa')
def DecoderModel(arm,seed=20261103,device='cuda:0'):
 assert arm in ARMS
 return Original('match',seed,device) if arm=='original' else Detail('point_detail',seed,device) if arm=='detail' else Sansa('sansa_ft',seed,device)
class DecoderData(Data):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ARMS;super().__init__(cache,'match',seed)
def frozen_backbone_state(m):
 trainable={n for n,p in m.named_parameters() if p.requires_grad}
 return {n:v for n,v in m.state_dict().items() if n not in trainable}
def frozen_decoder_state(m):return {}
def build(arm,checkpoint=None):
 m=DecoderModel(arm)
 if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
 return m
