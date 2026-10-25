"""Explicit source-positive/source-negative feature retrieval; no query prompts or GT."""
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.model import project_region_probabilities


def class_prototype(tokens,valid):
    if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or not valid.any(1).all():raise ValueError('invalid prototype support set')
    tokens=F.normalize(tokens.float(),dim=-1)
    return F.normalize((tokens*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True),dim=-1)


def dense_prototype(source_features,query_features,prompts,output_size=(512,512),temperature=.1):
    if source_features.ndim!=4 or query_features.ndim!=4 or source_features.shape[:2]!=query_features.shape[:2]:raise ValueError('incompatible source/query features')
    if temperature<=0:raise ValueError('temperature must be positive')
    prototypes=[]
    for sign in ('positive','negative'):
        grid=(prompts[sign+'_xy']*2-1)[:,None]
        samples=F.grid_sample(source_features.float(),grid,align_corners=False,padding_mode='border')[:,:,0].transpose(1,2)
        prototypes.append(class_prototype(samples,prompts[sign+'_mask']))
    normalized=F.normalize(query_features.float(),dim=1)
    logits=torch.einsum('bd,bdhw->bhw',prototypes[0]-prototypes[1],normalized)/temperature
    logits=F.interpolate(logits[:,None],size=output_size,mode='bilinear',align_corners=False)[:,0]
    return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())


def region_prototype(source,query,prompts,output_size=(512,512),temperature=.1):
    if temperature<=0:raise ValueError('temperature must be positive')
    prototypes=[]
    for sign in ('positive','negative'):
        weights=sample_assignment_at_cells(source['assignment_low'].float(),prompts[sign+'_xy'])
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        samples=torch.einsum('bkp,bkd->bpd',weights,source['tokens'][0].float())
        prototypes.append(class_prototype(samples,prompts[sign+'_mask']))
    normalized=F.normalize(query['tokens'][0].float(),dim=-1)
    logits=torch.einsum('bd,bkd->bk',prototypes[0]-prototypes[1],normalized)/temperature
    assignment=F.interpolate(query['assignment_low'].float(),size=output_size,mode='bilinear',align_corners=False)
    return project_region_probabilities(assignment,logits)
