"""Original episode objective with mathematically equivalent deterministic point reads."""
import torch
from torch.nn import functional as F
from .deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.multidomain_training.losses import mask_loss


def point_loss(logits,prompts):
    losses=[]
    for sign,label in [('positive',1.),('negative',0.)]:
        values=sample_assignment_at_cells(logits[:,None].float(),prompts[sign+'_xy'])[...,0,:]
        valid=prompts[sign+'_mask'];assert valid.any()
        losses.append(F.binary_cross_entropy_with_logits(values[valid],torch.full_like(values[valid],label)))
    return torch.stack(losses).mean()


def episode_loss(output,data,plan,prompts):
    def supervised(part,identity):
        gt,mpp=data.labels(identity);device=part['pixel_logits'].device
        return mask_loss(part,torch.from_numpy(gt==plan['target'])[None].to(device),torch.from_numpy(gt!=255)[None].to(device),mpp)[0]
    query=supervised(output['query'],plan['query_id']);loss=query
    if plan['query_id']!=plan['support_id']:loss=loss+.25*supervised(output['support'],plan['support_id'])
    signed=point_loss(output['support']['pixel_logits'],prompts)
    loss=loss+.1*signed+output['assignment_regularization'];assert torch.isfinite(loss)
    return loss
