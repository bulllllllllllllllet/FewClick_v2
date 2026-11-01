"""Common fresh architecture, preserving the original heads and deterministic sampling."""
from types import SimpleNamespace
import torch
from .deterministic_graph import PaperFewClick
from .image_interface import PhikonRegionBackbone
from benchmarks.v5.fair_comparison.adapted_models import generic_config,COCO,COCO_SHA
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder


def build_bridge_image(arm,device='cuda:0',seed=20261101):
    assert arm in ['resnet','phikon'];torch.manual_seed(seed)
    phase2=DeepRegionEncoder(generic_config(),12,64,256)
    cell=CellToRegionAttention(256,74,128)
    prompt=PromptRegionModel(256,4,2,.1)
    decoder=ContextAwareMaskDecoder(prompt,256,4,2,6,.1,freeze_prompt_encoder=False)
    model=PaperFewClick(SimpleNamespace(phase2=phase2,cell=cell,decoder=decoder),True,False).to(device)
    if arm=='phikon':
        backbone=PhikonRegionBackbone(device)
        for encoder in model.encoders:encoder.backbone=backbone
    for encoder in model.encoders:
        encoder.requires_grad_(False);encoder.embedding.requires_grad_(True);encoder.assignment.requires_grad_(True)
    model.eval()
    identity=dict(image_arm=arm,backbone_frozen=True,region_heads_trainable=True,J5_loaded=False,
                  COCO_sha256=COCO_SHA,deterministic_sampling=True,dropout=False,dense=False,
                  Phikon_interface='official512->32grid->bilinear64grid->[x,x]/sqrt2',seed=seed)
    return model,identity
