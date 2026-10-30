"""Identical signed-prototype dense decoder for three frozen image encoders."""
import torch
from torch import nn
import torch.nn.functional as F


def bilinear_resize(x,size):
    """align_corners=False separable interpolation using deterministic GEMMs."""
    h,w=size
    def weights(source,target):
        position=((torch.arange(target,device=x.device,dtype=x.dtype)+.5)*source/target-.5).clamp(0,source-1)
        index=torch.arange(source,device=x.device,dtype=x.dtype)
        return (1-(position[:,None]-index[None]).abs()).clamp(min=0)
    wy=weights(x.shape[-2],h);wx=weights(x.shape[-1],w)
    return torch.matmul(torch.matmul(wy,x),wx.T)


def point_prototype(support,points):
    """Equivalent border-padded grid_sample mean, without CUDA grid backward."""
    h,w=support.shape[-2:]
    x=(points[:,:,0]*w-.5).clamp(0,w-1);y=(points[:,:,1]*h-.5).clamp(0,h-1)
    wx=(1-(x[:,:,None]-torch.arange(w,device=x.device,dtype=x.dtype)).abs()).clamp(min=0)
    wy=(1-(y[:,:,None]-torch.arange(h,device=y.device,dtype=y.dtype)).abs()).clamp(min=0)
    weights=(wy[:,:,:,None]*wx[:,:,None,:]).flatten(2).mean(1)
    return torch.bmm(support.flatten(2),weights[:,:,None])[:,:,:,None]


class SharedPromptDecoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.condition=nn.Sequential(nn.Conv2d(388,128,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU())
        self.rgb=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.GELU(),
                               nn.Conv2d(32,32,3,stride=2,padding=1),nn.GELU())
        self.decode=nn.Sequential(nn.Conv2d(160,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,1,1))
        # Initialize common layers before the varying-size adapter.
        self.project=nn.Sequential(nn.Conv2d(input_channels,128,1),nn.GroupNorm(8,128),nn.GELU())

    def forward(self, query_feature, query_rgb, positive_xy, negative_xy,
                support_feature=None, local=True):
        # No query GT, class IDs, presence flags or query clicks in remote path.
        query=self.project(query_feature)
        support=query if support_feature is None else self.project(support_feature)
        prototypes=[];similarities=[];heatmaps=[]
        axis=(torch.arange(32,device=query.device,dtype=query.dtype)+.5)/32
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        for points in (positive_xy,negative_xy):
            prototype=point_prototype(support,points)
            prototypes.append(query*prototype)
            similarities.append(F.cosine_similarity(query,prototype,dim=1)[:,None])
            if local:
                distance=(xx[None,None]-points[:,:,0,None,None])**2+(yy[None,None]-points[:,:,1,None,None])**2
                heatmaps.append(torch.exp(-distance/(2*(1/32)**2)).amax(dim=1)[:,None])
            else:heatmaps.append(query.new_zeros((query.shape[0],1,32,32)))
        fused=self.condition(torch.cat([query,*prototypes,*similarities,*heatmaps],dim=1))
        rgb=self.rgb(query_rgb)
        fused=bilinear_resize(fused,rgb.shape[-2:])
        return bilinear_resize(self.decode(torch.cat([fused,rgb],1)),query_rgb.shape[-2:])[:,0]
