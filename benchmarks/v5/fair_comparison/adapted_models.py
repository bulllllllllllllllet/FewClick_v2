"""Differentiable native baselines and FewClick without historical J5 weights."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .audit import digest
from .local_data import fewclick_inputs
from .public_baselines import SharedContextWSISAM
from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
from benchmarks.v4.phase_1_multiscale.src.model import build_model as build_deeplab
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder
from benchmarks.v5.paper_multiscale.model import PaperFewClick


COCO=Path('/nfs-medical3/zyh/v4/phase1/models/deeplabv3_resnet50_coco-cd0a2569.pth')
COCO_SHA='cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06'
METHODS=('sam_ft','wsi_sam_ft','closed_reference','fewclick_m3d_xcell','fewclick_m3d_image')


def generic_config():
    if digest(COCO)!=COCO_SHA:raise ValueError('generic backbone identity mismatch')
    return dict(model=dict(pretrained_weights=str(COCO),spatial_temperature=.08))


class NoCell(nn.Module):
    def forward(self,region_tokens,assignment_at_cells,cells,cell_valid,total_cell_count):
        return dict(fused_tokens=region_tokens)


class ControlledPaper(PaperFewClick):
    def set_stage(self,joint):
        super().set_stage(joint)
        # These 12 channels have no historical tissue semantics. They are fresh
        # latent channels trained only through the binary mask loss in this control.
        if self.use_dense:self.encoders[0].semantic.requires_grad_(True)


def clean_fewclick(device,cell_mode='xcell',multiscale=True,dense=True):
    config=generic_config();phase2=DeepRegionEncoder(config,12,64,256)
    cell=CellToRegionAttention(256,74,128)
    prompt=PromptRegionModel(256,4,2,.1)
    decoder=ContextAwareMaskDecoder(prompt,256,4,2,6,.1,freeze_prompt_encoder=False)
    model=ControlledPaper(SimpleNamespace(phase2=phase2,cell=cell,decoder=decoder),multiscale,dense)
    if cell_mode=='none':model.cell=NoCell()
    elif cell_mode!='xcell':raise ValueError('unknown cell mode')
    return model.to(device),dict(backbone=str(COCO),backbone_sha256=COCO_SHA,
        target_cohort_initialization=False,J5_loaded=False,region_prompt_graph_cell_heads='fresh',
        cell_mode=cell_mode,XCell_overlap='not audited' if cell_mode=='xcell' else 'not used',
        semantic_channels='generic ASPP with fresh 12-channel projection; latent binary-supervised features',
        semantic_training='fine semantic head trainable; BN fixed; no tissue-class labels supplied',
        multiscale=multiscale,dense=dense)


def output_from_logits(logits):
    return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())


def points(prompts,size=1024,low=False):
    xy=torch.cat([prompts['positive_xy'],prompts['negative_xy']],1)*size
    labels=torch.cat([torch.ones_like(prompts['positive_mask'],dtype=torch.long),
                      torch.zeros_like(prompts['negative_mask'],dtype=torch.long)],1)
    return (xy/2+size/4 if low else xy),labels


class SAMFineTune(nn.Module):
    contexts=False;cells=False;closed=False
    def __init__(self,device):
        super().__init__()
        from segment_anything.utils.transforms import ResizeLongestSide
        config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_zero_shot.yaml').read_text())
        adapter=SAMAdapter(config);self.sam=adapter.model.to(device);self.prior=adapter.provenance()
        self.transform=ResizeLongestSide(self.sam.image_encoder.img_size);self.joint=False;self.set_stage(False)

    def set_stage(self,joint):
        self.joint=bool(joint);self.sam.image_encoder.requires_grad_(joint)
        self.sam.prompt_encoder.requires_grad_(True);self.sam.mask_decoder.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode);self.sam.image_encoder.eval();return self

    def parameter_groups(self):
        return [dict(params=list(self.sam.prompt_encoder.parameters())+list(self.sam.mask_decoder.parameters()),name='heads',lr=1e-4),
                dict(params=list(self.sam.image_encoder.parameters()),name='backbone',lr=1e-5)]

    def forward(self,raw,prompts,**kwargs):
        rgb=raw['rgb'][0].permute(1,2,0).cpu().numpy()
        transformed=self.transform.apply_image(rgb)
        image=torch.as_tensor(transformed,device=raw['rgb'].device).permute(2,0,1)
        embedding=self.sam.image_encoder(self.sam.preprocess(image)[None])
        sparse,dense=self.sam.prompt_encoder(points=points(prompts),boxes=None,masks=None)
        masks,quality=self.sam.mask_decoder(image_embeddings=embedding,image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=True)
        masks=self.sam.postprocess_masks(masks,transformed.shape[:2],rgb.shape[:2])
        selected=quality.detach().argmax(1)
        logits=masks[torch.arange(len(masks),device=masks.device),selected]
        return output_from_logits(logits)|dict(candidate_logits=masks,quality=quality,selected=selected)


class WSISAMFineTune(nn.Module):
    contexts=True;cells=False;closed=False
    def __init__(self,device):
        super().__init__();config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/wsi_sam_zero_shot.yaml').read_text())
        config['device']=str(device)
        for value in config['checkpoints'].values():
            if digest(value['path'])!=value['sha256']:raise ValueError('WSI-SAM checkpoint identity')
        backend=SharedContextWSISAM(config)
        self.sam=backend.sam;self.net_high=backend.net_high;self.net_low=backend.net_low
        self.prior=backend.load_report|dict(checkpoints=config['checkpoints']);self.joint=False;self.set_stage(False)

    def set_stage(self,joint):
        self.joint=bool(joint);self.sam.requires_grad_(False);self.sam.image_encoder.requires_grad_(joint)
        self.sam.prompt_encoder.requires_grad_(True);self.net_high.requires_grad_(True);self.net_low.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        # Keep TinyViT attention in training mode so relative-position biases
        # retain gradients (its eval path caches detached biases). Freeze BN only.
        for module in self.sam.image_encoder.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def parameter_groups(self):
        return [dict(params=list(self.sam.prompt_encoder.parameters())+list(self.net_high.parameters())+list(self.net_low.parameters()),name='heads',lr=1e-4),
                dict(params=list(self.sam.image_encoder.parameters()),name='backbone',lr=1e-5)]

    def encode(self,rgb,prompts,low=False):
        image=F.interpolate(rgb.float(),size=(1024,1024),mode='bilinear',align_corners=False).round().clamp(0,255).to(torch.uint8)
        embedding,intermediate=self.sam.image_encoder(self.sam.preprocess(image[0])[None])
        sparse,dense=self.sam.prompt_encoder(points=points(prompts,low=low),boxes=None,masks=None)
        return embedding,intermediate,sparse,dense

    def forward(self,raw,prompts,**kwargs):
        high,hi_inter,hs,hd=self.encode(raw['rgb'],prompts)
        low,lo_inter,ls,ld=self.encode(raw['rgb_1'],prompts,True)
        mask,_,token,features=self.net_high(image_embeddings=high,image_pe=[self.sam.prompt_encoder.get_dense_pe()],
            sparse_prompt_embeddings=[hs],dense_prompt_embeddings=[hd],multimask_output=False,wsi_token_only=False,interm_embeddings=hi_inter)
        _,_,lo_token,_=self.net_low(image_embeddings=low,image_pe=[self.sam.prompt_encoder.get_dense_pe()],
            sparse_prompt_embeddings=[ls],dense_prompt_embeddings=[ld],multimask_output=False,wsi_token_only=False,interm_embeddings=lo_inter)
        logits=((token+lo_token)@features).view(1,-1,256,256)+mask
        logits=F.interpolate(logits,size=raw['rgb'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
        return output_from_logits(logits)


class ClosedReference(nn.Module):
    contexts=False;cells=False;closed=True
    def __init__(self,device):
        super().__init__();config=generic_config()
        self.network=build_deeplab(config,12).to(device)
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        # Generic ASPP initialization shared; different final output projection.
        import copy
        tiger=copy.deepcopy(self.network.classifier);tiger[-1]=nn.Conv2d(256,7,1).to(device)
        self.tiger_head=tiger;self.joint=False;self.set_stage(False)
        self.prior=dict(backbone=str(COCO),backbone_sha256=COCO_SHA,J5_loaded=False,
            class_heads={'crc':12,'tiger':7},class_offsets={'crc':0,'tiger':1},supervision='all valid tissue classes on shared image sequence',
            inference='known dataset and target class ID; categorical argmax; privileged closed-set reference')

    def set_stage(self,joint):
        self.joint=bool(joint);self.network.backbone.requires_grad_(False)
        if joint:
            self.network.backbone.layer3.requires_grad_(True);self.network.backbone.layer4.requires_grad_(True)
        self.network.classifier.requires_grad_(True);self.tiger_head.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def parameter_groups(self):
        return [dict(params=list(self.network.classifier.parameters())+list(self.tiger_head.parameters()),name='heads',lr=1e-4),
                dict(params=list(self.network.backbone.layer3.parameters())+list(self.network.backbone.layer4.parameters()),name='backbone',lr=1e-5)]

    def forward(self,raw,prompts,dataset,target):
        features=self.network.backbone(fewclick_inputs(raw)['image'])['out']
        head=self.network.classifier if dataset=='crc' else self.tiger_head if dataset=='tiger' else None
        if head is None:raise ValueError('unknown reference dataset')
        logits=F.interpolate(head(features),size=raw['rgb'].shape[-2:],mode='bilinear',align_corners=False)
        offset=1 if dataset=='tiger' else 0;index=target-offset
        if not 0<=index<logits.shape[1]:raise ValueError('target outside reference head')
        other=torch.cat([logits[:,:index],logits[:,index+1:]],1)
        return output_from_logits(logits[:,index]-torch.logsumexp(other,1))|dict(class_logits=logits,class_offset=offset,
            categorical_prediction=logits.argmax(1)==index)


class FreshFewClick(nn.Module):
    contexts=True;cells=True;closed=False
    def __init__(self,device,cell_mode):
        super().__init__();self.core,self.prior=clean_fewclick(device,cell_mode)

    def set_stage(self,joint):self.core.set_stage(joint)
    def parameter_groups(self):return self.core.parameter_groups()
    def forward(self,raw,prompts,**kwargs):
        images=fewclick_inputs(raw)
        result=self.core(images,images,prompts,True,True)
        return result['query']|dict(assignment_regularization=result['assignment_regularization'])


def build(method,device):
    if method=='sam_ft':return SAMFineTune(device)
    if method=='wsi_sam_ft':return WSISAMFineTune(device)
    if method=='closed_reference':return ClosedReference(device)
    if method=='fewclick_m3d_xcell':return FreshFewClick(device,'xcell')
    if method=='fewclick_m3d_image':return FreshFewClick(device,'none')
    raise ValueError('unknown adaptation method '+method)
