"""Two bounded interventions in H; all weights and other click paths remain."""
import math
import torch
from torch.nn import functional as F
from benchmarks.v5.parallel_phikon_study.model import (
    DecoderModel, DenseHierarchy, signed_similarity,
)

MODES = ('control', 'no_anchor', 'no_coarse_prior', 'no_anchor_no_coarse_prior')


class DiagnosticHierarchy(DenseHierarchy):
    mode = 'control'

    def forward(self, sf, qf, prompts, source, prob):
        assert self.mode in MODES
        if self.mode == 'control':
            return super().forward(sf, qf, prompts, source, prob)
        assert sf.shape[-2:] == qf.shape[-2:] == (64, 64)
        s32 = F.normalize(F.avg_pool2d(sf, 2), dim=1)
        q32 = F.normalize(F.avg_pool2d(qf, 2), dim=1)
        s16 = F.normalize(F.avg_pool2d(sf, 4), dim=1)
        q16 = F.normalize(F.avg_pool2d(qf, 4), dim=1)
        coarse = torch.bmm(q16.flatten(2).transpose(1, 2), s16.flatten(2)) / .1
        prior = coarse.softmax(-1)
        expanded = prior.reshape(-1, 16, 16, 16, 16)
        for dim in [1, 2, 3, 4]:
            expanded = expanded.repeat_interleave(2, dim=dim)
        expanded = expanded.reshape(-1, 1024, 1024) / 4
        fine = torch.bmm(q32.flatten(2).transpose(1, 2), s32.flatten(2)) / .1
        affinity = (fine.softmax(-1) if 'no_coarse_prior' in self.mode else
                    (fine + .5 * expanded.clamp_min(1e-8).log()).softmax(-1))
        # This removes the extra Gaussian anchor only. Source segmentation,
        # signed point similarities and the original prompt encoder still use clicks.
        support_p = (F.interpolate(source.detach()[:, None], size=(32, 32),
                                   mode='bilinear', align_corners=False)
                     if 'no_anchor' in self.mode else
                     self.clicked_support_probability(source, prompts))
        transport = torch.bmm(affinity, support_p.flatten(2).transpose(1, 2))
        transport = transport.transpose(1, 2).reshape(-1, 1, 32, 32)
        # The coarse transport feature is retained: only its prior injection
        # into fine affinities is removed by no_coarse_prior.
        coarse_p = torch.bmm(prior, F.avg_pool2d(support_p, 2).flatten(2).transpose(1, 2))
        coarse_p = coarse_p.transpose(1, 2).reshape(-1, 1, 16, 16)
        coarse_p = F.interpolate(coarse_p, size=(32, 32), mode='bilinear', align_corners=False)
        certainty = 1 - (-(affinity * affinity.clamp_min(1e-8).log()).sum(-1) /
                         math.log(1024)).reshape(-1, 1, 32, 32)
        pos, neg = signed_similarity(sf, F.avg_pool2d(qf, 2), prompts)
        p = F.interpolate(prob[:, None], size=(32, 32), mode='bilinear', align_corners=False)
        features = torch.cat([transport, coarse_p, certainty, p, 4*p*(1-p), pos, neg, pos-neg], 1)
        delta = 2 * self.head(features).tanh()
        return delta, {'active': delta.new_ones(()), 'transport_std': transport.std().detach()}


def build(checkpoint):
    model = DecoderModel('H')
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'])
    extension = DiagnosticHierarchy().to('cuda:0')
    extension.load_state_dict(model.extension.state_dict(), strict=True)
    model.extension = extension
    return model.eval()


def decode_variants(model, se, qe, prompts, local=False):
    """Share the unchanged base decoding and projection between all four modes."""
    task = model.base.core.support_task(se, prompts)
    base, _ = model.base.core.decode(qe, task, prompts, local)
    base = model.pixel(se['pixel_embedding'], qe['pixel_embedding'], prompts, base)
    if local:
        return {mode: base['pixel_probability'] for mode in MODES}
    source = model.base.core.decode(se, task, prompts, True)[0]
    source = model.pixel(se['pixel_embedding'], se['pixel_embedding'], prompts, source)
    sf = model.pixel.project(se['pixel_embedding'].float())
    qf = model.pixel.project(qe['pixel_embedding'].float())
    result = {}
    try:
        for mode in MODES:
            model.extension.mode = mode
            delta, _ = model.extension(sf, qf, prompts, source['pixel_probability'], base['pixel_probability'])
            residual = F.interpolate(delta, size=(512, 512), mode='bilinear', align_corners=False)[:, 0]
            result[mode] = (base['pixel_logits'] + residual).sigmoid()
    finally:
        model.extension.mode = 'control'
    return result
