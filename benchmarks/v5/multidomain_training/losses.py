"""Ignore-aware binary objectives covering mixed, absent and full targets."""
import math
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary


def dilate(mask,radius):return F.max_pool2d(mask[:,None].float(),2*radius+1,stride=1,padding=radius)[:,0].bool()


def mask_loss(output,truth,valid,mpp):
    logits=output['pixel_logits'].float();p=logits.sigmoid();valid=valid.bool();truth=truth.bool()
    if truth.shape!=logits.shape or valid.shape!=truth.shape or not valid.flatten(1).any(1).all():raise ValueError('invalid supervised shape/validity')
    pieces=[];parts=[]
    for i in range(len(logits)):
        pos=truth[i]&valid[i];neg=~truth[i]&valid[i]
        if pos.any() and neg.any():bce=.5*(F.softplus(-logits[i][pos]).mean()+F.softplus(logits[i][neg]).mean())
        else:bce=F.binary_cross_entropy_with_logits(logits[i][valid[i]],truth[i][valid[i]].float())
        if pos.any():
            pv=p[i][valid[i]];tv=truth[i][valid[i]].float();dice=1-(2*(pv*tv).sum()+1)/(pv.sum()+tv.sum()+1)
        else:dice=p[i][valid[i]].mean()*.1
        pieces.append(bce+dice);parts.append(dict(bce=float(bce.detach()),dice_or_absent=float(dice.detach())))
    radius=int(math.ceil(2./min(mpp)))
    band=dilate(binary_boundary(truth,valid),radius)&valid&~dilate(~valid,radius)
    boundary=F.binary_cross_entropy_with_logits(logits[band],truth[band].float()) if band.any() else logits.sum()*0.
    return torch.stack(pieces).mean()+.5*boundary,dict(parts=parts,boundary=float(boundary.detach()))


def point_loss(logits,prompts):
    losses=[]
    for sign,label in [('positive',1.),('negative',0.)]:
        grid=(prompts[sign+'_xy']*2-1)[:,None]
        values=F.grid_sample(logits[:,None].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[:,0,0]
        valid=prompts[sign+'_mask']
        if not valid.any():raise ValueError('this training contract requires both prompt signs')
        losses.append(F.binary_cross_entropy_with_logits(values[valid],torch.full_like(values[valid],label)))
    return torch.stack(losses).mean()


def episode_loss(output,supervision,prompts,local):
    query,parts=mask_loss(output['query'],supervision['truth'],supervision['valid'],supervision['mpp'])
    loss=query;source=output['support']
    if source is None:raise ValueError('training requires source supervision')
    if not local:
        support,_=mask_loss(source,supervision['support_truth'],supervision['support_valid'],supervision['support_mpp'])
        loss=loss+.25*support
    signed=point_loss(source['pixel_logits'],prompts);loss=loss+.1*signed
    if not torch.isfinite(loss):raise FloatingPointError('non-finite episode loss')
    return loss,dict(query=float(query.detach()),source_point=float(signed.detach()),query_parts=parts)
