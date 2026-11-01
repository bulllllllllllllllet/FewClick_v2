"""Frozen Phikon features routed through unchanged three-view region/graph heads."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoImageProcessor

from benchmarks.v5.fair_comparison.adapted_models import clean_fewclick


class PhikonRegionBackbone(nn.Module):
    def __init__(self,device):
        super().__init__()
        path='/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2'
        self.encoder=AutoModel.from_pretrained(path,local_files_only=True).to(device).eval().requires_grad_(False)
        self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)

    @torch.no_grad()
    def forward(self,image):
        assert tuple(image.shape[1:])==(3,512,512)
        mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
        std=image.new_tensor([.229,.224,.225])[None,:,None,None]
        # The original loader supplies ImageNet-normalized RGB. Recover bytes,
        # then use the exact existing official Phikon normalization contract.
        rgb=((image*std+mean)*255).round().clamp(0,255).byte()
        images=[x.permute(1,2,0).cpu().numpy() for x in rgb]
        pixel=self.processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values'].to(image.device)
        tokens=self.encoder(pixel_values=pixel).last_hidden_state[:,1:]
        assert tokens.shape[1:]==(1024,1024)
        value=tokens.transpose(1,2).reshape(len(image),1024,32,32)
        value=F.interpolate(value,size=(64,64),mode='bilinear',align_corners=False)
        # Fixed isometric channel lift: identical old2048->256 region heads,
        # no candidate-only learned adapter or additional decoder parameters.
        return {'out':torch.cat([value,value],dim=1)/math.sqrt(2)}


def build_bridge_image(arm,device='cuda:0',seed=20261101):
    assert arm in ['resnet','phikon']
    torch.manual_seed(seed)
    model,identity=clean_fewclick(device,'xcell',True,False)
    if arm=='phikon':
        backbone=PhikonRegionBackbone(device)
        for encoder in model.encoders:encoder.backbone=backbone
    for encoder in model.encoders:
        encoder.requires_grad_(False)
        encoder.embedding.requires_grad_(True);encoder.assignment.requires_grad_(True)
    model.eval()
    identity.update(image_arm=arm,backbone_frozen=True,region_heads_trainable=True,
                    Phikon_interface='official fine512;32grid bilinear64grid; fixed [x,x]/sqrt2 channel lift',
                    frozen_weight_sharing='Phikon public weights shared across3 views; identical deterministic mapping',
                    dense=False,J5_loaded=False)
    return model,identity
