"""Explicit detector adapter and exact cached-center attended crops."""
from pathlib import Path
import hashlib

import pyvips
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


class CachedCenterCropper:
    """Reuse one centroid scan per instance map; retain original crop pixels."""
    def __init__(self):
        self.instances=None;self.centers={}

    def __call__(self,image,inst,label):
        if inst is not self.instances:
            ids=np.unique(inst);ids=ids[ids>0]
            centers=ndimage.center_of_mass(np.ones_like(inst,dtype=np.uint8),inst,ids)
            self.centers={int(i):c for i,c in zip(ids,centers)}
            self.instances=inst
        if int(label) not in self.centers:raise ValueError('crop references missing nucleus')
        y,x=map(int,self.centers[int(label)])
        h,w=inst.shape;ys,ye=max(0,y-64),min(h,y+64);xs,xe=max(0,x-64),min(w,x+64)
        padding=((max(0,64-y),max(0,y+64-h)),(max(0,64-x),max(0,x+64-w)))
        crop=np.pad(image[ys:ye,xs:xe],padding+((0,0),),mode='edge')
        mask=np.pad(inst[ys:ye,xs:xe],padding,mode='edge')==label
        blur=ndimage.gaussian_filter(mask.astype(np.float32),sigma=12,mode='nearest')
        weights=.35+.65*blur/(blur.max()+1e-6)
        return (np.clip(crop.astype(np.float32)/255*weights[...,None],0,1)*255).astype(np.uint8)


class InstanSegDetector:
    """Official released H&E TorchScript, including instance postprocessing.

    Accepts the cell-cache detector's approximately 0.25MPP image, evaluates at
    the model's own MPP and returns categorical instances on the input grid.
    """
    def __init__(self,path,device,input_mpp=.25):
        self.path=Path(path);self.device=torch.device(device);self.input_mpp=float(input_mpp)
        if self.device.type!='cuda':raise ValueError('speed experiment requires explicit CUDA')
        self.model=torch.jit.load(str(path),map_location=self.device).eval()
        self.pixel_size=float(self.model.pixel_size)
        if self.pixel_size!=.5 or self.model.cells_and_nuclei:raise ValueError('unexpected brightfield nuclei checkpoint')
        self.model=self.model.to(memory_format=torch.channels_last)
        self.weight_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        expected='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0'
        if self.weight_sha256!=expected:raise ValueError('official RDF TorchScript SHA256 mismatch')

    @torch.inference_mode()
    def eval(self,image,diameter=None,batch_size=None):
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('InstanSeg expects RGB')
        x=torch.from_numpy(image.copy()).permute(2,0,1).float()[None].to(self.device)
        shape=tuple(max(32,round(n*self.input_mpp/self.pixel_size)) for n in image.shape[:2])
        x=F.interpolate(x,size=shape,mode='bilinear',align_corners=False,antialias=True)
        # Match the official Python API's per-channel 0.1/99.9 percentiles.
        flat=x[0].flatten(1)
        q=torch.quantile(flat,torch.tensor([.001,.999],device=self.device),dim=1)
        x=(x-q[0][None,:,None,None])/(q[1]-q[0]).clamp_min(1e-3)[None,:,None,None]
        x=x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda'):
            labels=self.model(x)
        if labels.ndim!=4 or labels.shape[:2]!=(1,1):raise ValueError('unexpected instance output shape')
        labels=F.interpolate(labels.float(),size=image.shape[:2],mode='nearest')[0,0]
        if not torch.isfinite(labels).all() or (labels<0).any() or not torch.equal(labels,labels.round()):
            raise ValueError('invalid categorical instance map')
        return (labels.cpu().numpy().astype(np.int32),)
