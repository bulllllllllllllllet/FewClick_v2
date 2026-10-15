"""Signed visual click tokens interact with dense queries on cached image features."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .continue_dense_boundary import DenseContextModel


def sample_click_features(feature, xy, valid):
    """Normalized xy use pixel-edge coordinates, like the original prompt cache."""
    safe_xy = torch.where(valid[..., None], xy, torch.zeros_like(xy))
    sampled = F.grid_sample(feature, (safe_xy*2-1)[:, :, None],
                            mode='bilinear', padding_mode='border', align_corners=False)
    return sampled[..., 0].transpose(1, 2)*valid[..., None]


class PixelPromptInteraction(nn.Module):
    def __init__(self, mode, width=32):
        super().__init__()
        if mode not in ('pooled', 'per_click'):
            raise ValueError(mode)
        self.mode = mode
        self.deep = nn.Conv2d(256, width, 1)
        self.shallow = nn.Conv2d(32, width, 1)
        self.context = nn.Conv2d(48, width, 1)
        self.position = nn.Linear(2, width)
        self.sign = nn.Embedding(2, width)
        self.token_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.fusion = nn.Sequential(
            nn.Conv2d(3*width+5, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.output = nn.Conv2d(48, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def attend(self, query, visual, xy, valid, sign_id):
        if xy.shape[1] == 0:
            return torch.zeros_like(query), query.new_zeros(*query.shape[:2], 1)
        safe_xy = torch.where(valid[..., None], xy, torch.zeros_like(xy))
        tokens = (sample_click_features(visual, safe_xy, valid)
                  + self.position(safe_xy) + self.sign.weight[sign_id])
        tokens = tokens*valid[..., None]
        if self.mode == 'pooled':
            count = valid.sum(1, keepdim=True).clamp_min(1)
            tokens = tokens.sum(1, keepdim=True)/count[..., None]
            valid = valid.any(1, keepdim=True)
        tokens = self.token_norm(tokens)
        keys, values = self.key(tokens), self.value(tokens)
        scores = query @ keys.transpose(1, 2)/math.sqrt(query.shape[-1])
        masked = scores.masked_fill(~valid[:, None], torch.finfo(scores.dtype).min)
        weights = masked.softmax(-1)*valid[:, None]
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-9)
        context = weights @ values
        maximum = masked.amax(-1, keepdim=True)
        maximum = torch.where(valid.any(1)[:, None, None], maximum, torch.zeros_like(maximum))
        return context, maximum

    def forward(self, hidden, sample, probability):
        size = hidden.shape[-2:]
        visual = (F.interpolate(self.deep(sample['deep'].float()), size=size,
                                mode='bilinear', align_corners=False)
                  + F.interpolate(self.shallow(sample['shallow'].float()), size=size,
                                  mode='bilinear', align_corners=False))
        context = self.context(hidden.float())+visual
        y = (torch.arange(size[0], device=context.device, dtype=context.dtype)+.5)/size[0]
        x = (torch.arange(size[1], device=context.device, dtype=context.dtype)+.5)/size[1]
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        positions = torch.stack((xx, yy), -1).reshape(1, -1, 2)
        query = self.query(self.query_norm(context.flatten(2).transpose(1, 2)+self.position(positions)))
        positive, pos_score = self.attend(query, visual, sample['positive_xy'].float(), sample['positive_mask'].bool(), 0)
        negative, neg_score = self.attend(query, visual, sample['negative_xy'].float(), sample['negative_mask'].bool(), 1)
        def spatial(value):
            return value.transpose(1, 2).reshape(hidden.shape[0], -1, *size)
        coarse = F.interpolate(probability[:, None], size=size, mode='bilinear', align_corners=False)
        maps = F.interpolate(sample['prompt_maps'].float(), size=size, mode='bilinear', align_corners=False)
        fusion = self.fusion(torch.cat((context, spatial(positive), spatial(negative),
                                       spatial(pos_score), spatial(neg_score), coarse, maps), 1))
        return F.interpolate(self.output(fusion), size=probability.shape[-2:],
                             mode='bilinear', align_corners=False)[:, 0]


class PixelPromptDenseModel(nn.Module):
    def __init__(self, coarse_state, mode):
        super().__init__()
        if mode not in ('control', 'pooled', 'per_click'):
            raise ValueError(mode)
        self.mode = mode
        self.coarse = DenseContextModel()
        self.coarse.load_state_dict(coarse_state, strict=True)
        self.hidden = None
        if mode != 'control':
            self.interaction = PixelPromptInteraction(mode)
            self.coarse.head.fusion.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.hidden = output

    def forward(self, sample):
        self.hidden = None
        coarse = self.coarse(sample)
        residual = (self.interaction(self.hidden, sample, coarse['pixel_probability'])
                    if self.mode != 'control' else torch.zeros_like(coarse['pixel_logits']))
        logits = coarse['pixel_logits']+residual
        return {'pixel_logits': logits, 'pixel_probability': logits.sigmoid(),
                'residual': residual, 'coarse_probability': coarse['pixel_probability']}
