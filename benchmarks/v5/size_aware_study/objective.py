"""Fixed precision penalty, training labels only; inference is unchanged."""
import torch
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss as original_loss


def precision_penalty(logits,gt,target):
    valid=gt!=255;positive=(gt==target)&valid;negative=(gt!=target)&valid
    p=logits[0].sigmoid()
    if not positive.any():return p[valid].mean()
    fp=p[negative].sum();tp=p[positive].sum()
    return fp/(tp+fp+1.)


def episode_loss(output,data,plan,prompts):
    base=original_loss(output,data,plan,prompts)
    if output.get('loss_arm')!='precision':return base
    device=output['query']['pixel_logits'].device
    def penalty(part,identity):
        gt,_=data.labels(identity)
        return precision_penalty(part['pixel_logits'],torch.from_numpy(gt).to(device),plan['target'])
    extra=penalty(output['query'],plan['query_id'])
    if plan['support_id']!=plan['query_id']:extra=extra+.25*penalty(output['support'],plan['support_id'])
    return base+.25*extra
