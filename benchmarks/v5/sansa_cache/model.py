import hashlib
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(eq=False)
class Frame:
    rgb: torch.Tensor
    backbone: object


class CachedSANSA(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.support_memory = {}

    @torch.no_grad()
    def encode(self, inputs):
        rgb = self.original.encode(inputs)
        net = self.original.network
        # Retain native remote batch size2 to preserve deterministic kernels.
        samples, _, _, sizes = net._preprocess_visual_features(torch.stack([rgb, rgb])[None], net.sam.image_size)
        both = net._forward_backbone(samples, sizes)
        from models.sansa.model_utils import BackboneOutput
        one = BackboneOutput([sizes[0]], [v[:, :1].clone() for v in both.vision_feats],
                             [v[:, :1].clone() for v in both.vision_pos_embeds], both.feat_sizes)
        return Frame(rgb, one)

    @staticmethod
    def key(points):
        h = hashlib.sha256()
        for key in ['positive_xy', 'negative_xy', 'positive_mask', 'negative_mask']:
            h.update(points[key].detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def support(self, frame, points):
        key = (frame, self.key(points))
        if key not in self.support_memory:
            from util.promptable_utils import rescale_prompt
            xy, labels = [], []
            for sign, value in [('positive', 1), ('negative', 0)]:
                valid = points[sign+'_mask'][0].bool()
                xy.append(points[sign+'_xy'][0][valid])
                labels.extend([value]*int(valid.sum()))
            prompt = dict(point_coords=torch.cat(xy)[None]*640,
                          point_labels=torch.tensor([labels], device=xy[0].device, dtype=torch.int32))
            net = self.original.network
            prompt = rescale_prompt(prompt, 'point', frame.backbone.orig_size[0], net.sam.image_size)
            decoded = net._compute_decoder_out_no_mem(frame.backbone, 0, prompt_input=prompt)
            self.support_memory[key] = net._compute_memory_bank_dict(decoded, frame.backbone, 0)
        return self.support_memory[key]

    @torch.no_grad()
    def decode(self, se, qe, points, local=False):
        if local:
            return self.original._run(se.rgb, qe.rgb, points, True)[0]
        net = self.original.network
        # Only the support entry, never another query's memory.
        decoded = net._compute_decoder_out_w_mem(qe.backbone, 0, 1, {0: self.support(se, points)})
        logits = F.interpolate(decoded.masks[0][None], size=qe.backbone.orig_size[0],
                               mode='bilinear', align_corners=False)[0]
        logits = F.interpolate(logits[:, None], size=(512, 512), mode='bilinear', align_corners=False)[:, 0]
        return dict(pixel_logits=logits, pixel_probability=logits.sigmoid())

    def forward(self, support, query, prompts, local, shared, with_support=False):
        se = self.encode(support)
        qe = se if shared else self.encode(query)
        return dict(query=self.decode(se, qe, prompts, local))
