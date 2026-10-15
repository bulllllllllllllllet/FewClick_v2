"""Cached image features + prompt task -> unconstrained dense pixel residual."""
import torch
from torch import nn
import torch.nn.functional as F


class DensePromptDecoder(nn.Module):
    def __init__(self, high_resolution=False, width=48):
        super().__init__()
        self.high_resolution = high_resolution
        self.deep = nn.Sequential(nn.Conv2d(256, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.task = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 2*width))
        self.context = nn.Sequential(nn.Conv2d(width+16, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        if high_resolution:
            self.shallow = nn.Sequential(nn.Conv2d(32, 24, 1), nn.GroupNorm(8,24), nn.GELU())
            self.fusion = nn.Sequential(nn.Conv2d(width+24, width, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, features):
        base = features['base'].float()
        hidden = self.deep(features['deep'].float())
        scale, shift = self.task(features['task'].float()).chunk(2, dim=1)
        hidden = hidden*(1+.1*scale[:,:,None,None].tanh()) + shift[:,:,None,None]
        size = hidden.shape[-2:]
        probability = F.interpolate(base[:,None],size=size,mode='bilinear',align_corners=False)
        semantic = F.interpolate(features['semantic'].float(),size=size,mode='bilinear',align_corners=False).softmax(1)
        prompts = F.interpolate(features['prompt_maps'].float(),size=size,mode='bilinear',align_corners=False)
        hidden = self.context(torch.cat([hidden,semantic,probability,4*probability*(1-probability),prompts],1))
        if self.high_resolution:
            shallow = self.shallow(features['shallow'].float())
            hidden = self.fusion(torch.cat([F.interpolate(hidden,size=shallow.shape[-2:],mode='bilinear',align_corners=False),shallow],1))
        residual = F.interpolate(self.output(hidden),size=base.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        logits = torch.logit(base.clamp(1e-6,1-1e-6))+residual
        return {'pixel_logits':logits,'pixel_probability':logits.sigmoid(),'residual':residual}


def prompt_maps(batch, size=128):
    device=batch['positive_xy'].device
    coord=(torch.arange(size,device=device)+.5)/size
    yy,xx=torch.meshgrid(coord,coord,indexing='ij')
    maps=[]
    for name in ('positive','negative'):
        xy=batch[name+'_xy']; mask=batch[name+'_mask']
        distances=(xx[None,None]-xy[:,:,0,None,None]).square()+(yy[None,None]-xy[:,:,1,None,None]).square()
        maps.append((torch.exp(-distances/(2*(4/512)**2))*mask[:,:,None,None]).amax(1))
    return torch.stack(maps,1)
