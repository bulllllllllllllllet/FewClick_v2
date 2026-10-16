"""Audited H&E-only cell preparation for external-domain diagnostic inputs."""
from pathlib import Path
import re
import numpy as np
import pyvips
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from cellpose import models
from module.TransPath.ctran import ConvStem, ctranspath
from XCellFormer import XCellFormer
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features

WEIGHTS = {'cpsam':Path('/home/zhaoyh/.cellpose/models/cpsam'),
    'ctranspath':Path('/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth'),
    'xcell':Path('/home/zhaoyh/FewClick/module/checkpoint/he_model_best.pth')}


class GlobalPool(torch.nn.Module):
    def forward(self, x):
        if x.ndim==4:x=F.adaptive_avg_pool2d(x.permute(0,3,1,2),1)
        return x.flatten(1)


def load_cell_models(device):
    detector=models.CellposeModel(gpu=True,pretrained_model=str(WEIGHTS['cpsam']),device=device)
    ctp=ctranspath();ctp.patch_embed=ConvStem(img_size=224,patch_size=4,in_chans=3,embed_dim=ctp.embed_dim)
    state=torch.load(WEIGHTS['ctranspath'],map_location='cpu',weights_only=False)['model'];adapted={}
    for key,value in state.items():
        for i in range(3):
            if f'layers.{i}.downsample' in key:
                key=key.replace(f'layers.{i}.downsample',f'layers.{i+1}.downsample');break
        adapted[key]=value
    mismatch=ctp.load_state_dict(adapted,strict=False)
    allowed=lambda k:k in ('patch_embed.norm.weight','patch_embed.norm.bias') or bool(re.fullmatch(r'layers\.\d+\.blocks\.\d+\.(attn_mask|attn\.relative_position_index)',k))
    if set(mismatch.missing_keys)!={'head.fc.weight','head.fc.bias'} or not all(allowed(k) for k in mismatch.unexpected_keys):
        raise ValueError(f'CTransPath compatibility mismatch: {mismatch}')
    # Identical historical v4 extraction recipe: discarded classifier head,
    # old Swin buffers and unused patch norm explicitly audited above.
    ctp.head=GlobalPool();ctp=ctp.to(device).eval().requires_grad_(False)
    xcell=XCellFormer(768,512,8,4,output_dim=64,max_cells=255,use_large_vit=False,device=str(device))
    xcell.load_state_dict(torch.load(WEIGHTS['xcell'],map_location='cpu',weights_only=False),strict=True)
    xcell=xcell.to(device).eval().requires_grad_(False)
    audit={'ctp_missing':mismatch.missing_keys,'ctp_unexpected':mismatch.unexpected_keys,'xcell_strict':True,
           'cell_class_policy':'0 unknown; no tissue-GT-to-nuclear-class mapping'}
    return detector,ctp,xcell,audit


def cells_from_instances(inst, max_cells=255):
    ids=np.unique(inst);ids=ids[ids>0];total=len(ids)
    if not total:return np.empty((0,4),np.float32),np.empty(0,np.int32),0
    yx=np.asarray(ndimage.center_of_mass(np.ones_like(inst,dtype=np.uint8),inst,ids))
    # Original GDPH legacy policy: ascending instance IDs, then y/x for XCell.
    ids=ids[:max_cells];yx=yx[:max_cells]
    order=np.lexsort((yx[:,1],yx[:,0]));ids=ids[order];yx=yx[order]
    area=np.bincount(inst.ravel())[ids]
    meta=np.stack((yx[:,1]/inst.shape[1],yx[:,0]/inst.shape[0],np.log1p(area),np.zeros(len(ids))),1).astype(np.float32)
    return meta,ids,total


def attended_crop(image, inst, label):
    y,x=ndimage.center_of_mass(np.ones_like(inst,dtype=np.uint8),inst,[label])[0];y,x=int(y),int(x)
    h,w=inst.shape;ys,ye=max(0,y-64),min(h,y+64);xs,xe=max(0,x-64),min(w,x+64)
    padding=((max(0,64-y),max(0,y+64-h)),(max(0,64-x),max(0,x+64-w)))
    crop=np.pad(image[ys:ye,xs:xe],padding+((0,0),),mode='edge')
    mask=np.pad(inst[ys:ye,xs:xe],padding,mode='edge')==label
    blur=ndimage.gaussian_filter(mask.astype(np.float32),sigma=12,mode='nearest')
    weights=.35+.65*blur/(blur.max()+1e-6)
    return (np.clip(crop.astype(np.float32)/255*weights[...,None],0,1)*255).astype(np.uint8)


@torch.inference_mode()
def prepare_cells(image, detector, ctp, xcell, device):
    # Input 0.5MPP, detection/crops at 0.25MPP; interpolation adds no detail.
    high=np.array(Image.fromarray(image).resize((1024,1024),Image.Resampling.BILINEAR))
    inst=np.asarray(detector.eval(high,diameter=18,batch_size=1)[0],dtype=np.int32)
    meta,ids,total=cells_from_instances(inst)
    tensors=[]
    for label in ids:
        crop=attended_crop(high,inst,int(label))
        x=torch.from_numpy(crop.copy()).permute(2,0,1).float()[None]/255
        x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0]
        tensors.append((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
    reg=np.empty((0,64),np.float32)
    if tensors:
        raw=torch.cat([ctp(torch.stack(tensors[i:i+32]).to(device)).float() for i in range(0,len(tensors),32)])
        padded=raw.new_zeros((1,255,768));valid=raw.new_zeros((1,255))
        padded[0,:len(raw)]=raw;valid[0,:len(raw)]=1
        _,r,_,_=xcell(None,padded,valid);reg=r[0,:len(raw)].cpu().numpy()
    features=encode_xcell_features(meta,reg)
    if features.shape!=(len(ids),74) or not np.isfinite(features).all():raise ValueError('invalid external cells')
    return {'cells':torch.from_numpy(features)[None].to(device),
        'cell_valid':torch.ones(1,len(ids),dtype=torch.bool,device=device),
        'total_cell_count':torch.tensor([total],device=device,dtype=torch.float32)}, {'instances':inst,'meta':meta,'reg':reg,'features':features,'total':total}
