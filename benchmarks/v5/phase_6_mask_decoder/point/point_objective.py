"""Supervise the point head on its actual inference composite, with matched controls."""
import torch
import torch.nn.functional as F
from .boundary_refinement import dilate
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import balanced_pixel_bce, soft_pixel_dice


def inference_indices(probability, count=65536):
    return (probability.flatten(1)-.5).abs().topk(min(count, probability[0].numel()), largest=False).indices


def inference_composite(output, sampled_count):
    """Only the suffix contains inference points; sampled anchors never widen the gate."""
    coarse = output['coarse_probability']
    indices = output['indices'][:, sampled_count:]
    logits = output['point_logits'][:, sampled_count:]
    if indices.shape[1] == 0:
        raise ValueError('inference point suffix required')
    probability = coarse.flatten(1).scatter(1, indices, logits.sigmoid()).reshape_as(coarse)
    full_logits = torch.logit(coarse.clamp(1e-6, 1-1e-6)).flatten(1).scatter(1, indices, logits).reshape_as(coarse)
    return probability, full_logits


def objective(output, sample, sampled_count, mode):
    if mode not in ('control', 'boundary', 'global'):
        raise ValueError(mode)
    indices = output['indices'][:, :sampled_count]
    logits = output['point_logits'][:, :sampled_count]
    truth = sample['truth'].flatten(1).gather(1, indices)
    valid = sample['valid'].flatten(1).gather(1, indices)
    bce = balanced_pixel_bce(logits, truth, valid)
    probability, full_logits = inference_composite(output, sampled_count)
    dice = (soft_pixel_dice(probability, sample['truth'], sample['valid']) if mode == 'global'
            else soft_pixel_dice(logits.sigmoid(), truth, valid))
    band = dilate(binary_boundary(sample['truth'], sample['valid']), 4) & sample['valid'] & ~dilate(~sample['valid'], 4)
    boundary = F.binary_cross_entropy_with_logits(full_logits[band], sample['truth'][band].float()) if band.any() else full_logits.sum()*0
    loss = bce + dice + (.5*boundary if mode != 'control' else 0)
    return loss, {'bce': float(bce.detach()), 'dice_loss': float(dice.detach()),
                  'boundary_bce': float(boundary.detach()), 'boundary_pixels': int(band.sum())}
