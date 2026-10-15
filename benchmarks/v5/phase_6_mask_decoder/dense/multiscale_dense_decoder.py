"""Progressive native feature fusion on a warm-started prompt-conditioned decoder."""
import torch
from torch import nn
import torch.nn.functional as F
from .continue_dense_boundary import DenseContextModel


def projection(input_channels, output_channels, groups=8):
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 1),
                         nn.GroupNorm(groups, output_channels), nn.GELU())


class NativeFeatureFusion(nn.Module):
    def __init__(self, mode, width=32):
        super().__init__()
        if mode not in ('native', 'middle', 'full256'):
            raise ValueError(mode)
        self.mode = mode
        # Construct common layers before optional ones, keeping seeded common
        # initialization identical across the progressive architecture arms.
        self.context = projection(48, width)
        self.native = projection(256, width)
        self.fusion128 = nn.Sequential(
            nn.Conv2d(2*width+3, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.output128 = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output128.weight)
        nn.init.zeros_(self.output128.bias)
        if mode in ('middle', 'full256'):
            self.middle = projection(512, width)
        if mode == 'full256':
            self.stem = projection(64, 16, groups=4)
            self.fusion256 = nn.Sequential(
                nn.Conv2d(width+16+3, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.GELU())
            self.output256 = nn.Conv2d(24, 1, 1)
            nn.init.zeros_(self.output256.weight)
            nn.init.zeros_(self.output256.bias)

    def forward(self, hidden, sample, coarse_probability):
        context = self.context(hidden.float())
        if self.mode in ('middle', 'full256'):
            middle = self.middle(sample['middle'].float())
            context = context + F.interpolate(middle, size=context.shape[-2:], mode='bilinear', align_corners=False)
        native = self.native(sample['native_shallow'].float())
        size = native.shape[-2:]
        context = F.interpolate(context, size=size, mode='bilinear', align_corners=False)
        prompts = F.interpolate(sample['prompt_maps'].float(), size=size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse_probability[:, None], size=size, mode='bilinear', align_corners=False)
        hidden128 = self.fusion128(torch.cat((context, native, prompts, coarse), 1))
        output_size = coarse_probability.shape[-2:]
        residual = F.interpolate(self.output128(hidden128), size=output_size, mode='bilinear', align_corners=False)[:, 0]
        if self.mode == 'full256':
            stem = self.stem(sample['stem'].float())
            size = stem.shape[-2:]
            hidden256 = self.fusion256(torch.cat((
                F.interpolate(hidden128, size=size, mode='bilinear', align_corners=False), stem,
                F.interpolate(sample['prompt_maps'].float(), size=size, mode='bilinear', align_corners=False),
                F.interpolate(coarse_probability[:, None], size=size, mode='bilinear', align_corners=False)), 1))
            residual = residual + F.interpolate(self.output256(hidden256), size=output_size,
                                                mode='bilinear', align_corners=False)[:, 0]
        return residual


class MultiscaleDenseModel(nn.Module):
    def __init__(self, coarse_state, mode):
        super().__init__()
        if mode not in ('control', 'native', 'middle', 'full256'):
            raise ValueError(mode)
        self.mode = mode
        self.coarse = DenseContextModel()
        self.coarse.load_state_dict(coarse_state, strict=True)
        self.hidden = None
        if mode != 'control':
            self.fusion = NativeFeatureFusion(mode)
            self.coarse.head.fusion.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.hidden = output

    def forward(self, sample):
        self.hidden = None
        coarse = self.coarse(sample)
        residual = (self.fusion(self.hidden, sample, coarse['pixel_probability'])
                    if self.mode != 'control' else torch.zeros_like(coarse['pixel_probability']))
        logits = coarse['pixel_logits'] + residual
        return {'pixel_logits': logits, 'pixel_probability': logits.sigmoid(),
                'residual': residual, 'coarse_probability': coarse['pixel_probability']}
