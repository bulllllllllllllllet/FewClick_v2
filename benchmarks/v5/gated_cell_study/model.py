"""Coverage-weighted zero-initialized cell residual around the image-only fusion."""
import torch
from benchmarks.v5.matcher_cell_study.model import DecoderData,DecoderModel as Original
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialCell
class GatedCell(SpatialCell):
 def __init__(self,original):
  super().__init__(original,'S11');self.residual_scale=torch.nn.Parameter(torch.zeros(()))
 def forward(self,tokens,mass,cells,valid,total_cell_count=None):
  zero=torch.zeros_like(tokens);counts=tokens.new_zeros(tokens.shape[:2])
  image=self.fuse(torch.cat([tokens,zero,counts[...,None]],-1))
  out=super().forward(tokens,mass,cells,valid,total_cell_count)
  selected_mass=(mass*valid[:,None,:]).sum(-1)
  coverage=selected_mass/(1+selected_mass)
  gate=self.residual_scale.tanh()*coverage
  out['fused_tokens']=image+gate[...,None]*(out['fused_tokens']-image)
  out['cell_residual_gate']=gate
  return out
class DecoderModel(Original):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  super().__init__(arm,seed,device)
  if arm=='cell_on':self.base.core.cell=GatedCell(self.base.core.cell).to(device)
