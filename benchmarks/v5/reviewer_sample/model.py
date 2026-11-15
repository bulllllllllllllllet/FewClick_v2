"""Same seven Phikon encodings: learned prototypes or fixed spatial regions."""
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.hg_combination_study.model import DecoderModel as Full, MixedFineBackbone
from benchmarks.v5.architecture_bridge.image_interface import PhikonRegionBackbone
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.support_match_study.model import DecoderData as Data
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state, frozen_decoder_state

ARMS = ('proto_ms', 'fixed_grid')


class IdentityInput(nn.Module):
    @staticmethod
    def transform(inputs):
        return inputs


class ProtoMS(nn.Module):
    def __init__(self, seed=20261103, device='cuda:0'):
        super().__init__()
        torch.manual_seed(seed)
        self.base = IdentityInput()
        self.backbone = PhikonRegionBackbone(device)
        self.fine_backbone = MixedFineBackbone(self.backbone)
        # Fresh downstream heads, no target-trained representation or region bank.
        torch.manual_seed(seed)
        self.projections = nn.ModuleList([nn.Conv2d(2048, 256, 1) for _ in range(3)])
        self.head = nn.Sequential(nn.Conv2d(3*(4*256+4), 256, 1), nn.GELU(),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(256, 1, 1))
        self.to(device)
        self.study_arm = 'proto_ms'

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def raw_features(self, inputs):
        return [self.fine_backbone(inputs['image'])['out'],
                self.backbone(inputs['image_1'])['out'],
                self.backbone(inputs['image_2'])['out']]

    def encode(self, inputs):
        return [project(raw.float()) for project, raw in zip(self.projections, self.raw_features(inputs))]

    @staticmethod
    def align(value, factor):
        h, w = value.shape[-2:]
        ch, cw = h//factor, w//factor
        y, x = (h-ch)//2, (w-cw)//2
        return F.interpolate(value[..., y:y+ch, x:x+cw], size=(64, 64), mode='bilinear', align_corners=False)

    def prediction(self, se, qe, prompts):
        channels = []
        for sf, qf, factor in zip(se, qe, [1, 2, 4]):
            unit = F.normalize(qf, dim=1)
            prototypes, similarities = [], []
            for sign in ['positive', 'negative']:
                valid = prompts[sign+'_mask'].bool()
                assert valid.any(1).all()
                xy = .5 + (prompts[sign+'_xy'].float()-.5)/factor
                points = sample_assignment_at_cells(sf, xy)
                proto = F.normalize((points*valid[:, None]).sum(-1)/valid.sum(-1)[:, None], dim=1)
                prototypes.append(proto[:, :, None, None].expand(-1, -1, 64, 64))
                sim = (unit*proto[:, :, None, None]).sum(1, keepdim=True)
                maximum = torch.einsum('bcp,bchw->bphw', F.normalize(points, dim=1), unit)
                maximum = maximum.masked_fill(~valid[:, :, None, None], -2).amax(1, keepdim=True)
                similarities.extend([self.align(sim, factor), self.align(maximum, factor)])
            global_context = qf.mean((2, 3), keepdim=True).expand(-1, -1, 64, 64)
            channels.extend([self.align(qf, factor), *prototypes, global_context, *similarities])
        logits = F.interpolate(self.head(torch.cat(channels, 1)), size=(512, 512), mode='bilinear', align_corners=False)[:, 0]
        return dict(pixel_logits=logits, pixel_probability=logits.sigmoid())

    def cached_decode(self, se, qe, prompts, local=False, with_support=True):
        out = self.prediction(se, qe, prompts)
        source = (out if local else self.prediction(se, se, prompts)) if with_support else None
        return out, source, {}, out['pixel_logits'].new_zeros(1, 2, 128, 128)

    def forward(self, support, query, prompts, local, shared, with_support=True):
        assert not local or shared
        se = self.encode(support)
        qe = se if shared else self.encode(query)
        out, source, _, maps = self.cached_decode(se, qe, prompts, local, with_support)
        return dict(query=out, support=source, query_prompt_maps=maps,
                    task_token=se[0].mean((2, 3)), assignment_regularization=out['pixel_logits'].new_zeros(()))


class FixedGrid(Full):
    def __init__(self, seed=20261103, device='cuda:0'):
        super().__init__('HG_mix', seed, device)
        self.study_arm = 'fixed_grid'
        for encoder in self.base.core.encoders:
            assert torch.count_nonzero(encoder.assignment.weight) == 0
            assert torch.count_nonzero(encoder.assignment.bias) == 0
            encoder.assignment.requires_grad_(False)


def DecoderModel(arm, seed=20261103, device='cuda:0'):
    assert arm in ARMS
    return ProtoMS(seed, device) if arm == 'proto_ms' else FixedGrid(seed, device)


class DecoderData(Data):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(cache, 'match', seed)


def build(arm, checkpoint=None):
    model = DecoderModel(arm)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'], strict=True)
    return model
