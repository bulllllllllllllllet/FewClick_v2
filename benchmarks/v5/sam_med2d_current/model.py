"""Medical public encoder frozen; same signed dense bridge and native decoder."""
from pathlib import Path
import torch
import numpy as np
import yaml
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.sam_prior_repair.model import DecoderModel as SAMBridge
from benchmarks.v5.support_match_study.model import SignedMatcher, DecoderData as SharedData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.fair_comparison.adapted_models import points

ARMS = ('med2d',)


class DecoderModel(SAMBridge):
    def __init__(self, arm, seed=20261103, device='cuda:0'):
        assert arm == 'med2d'
        nn.Module.__init__(self)
        torch.manual_seed(seed)
        from benchmarks.v4.baseline.integrations.sam_external import SAMMed2DBackend
        config = yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_med2d_zero_shot.yaml').read_text())
        config['device'] = device
        self.backend = SAMMed2DBackend(config)
        self.sam = self.backend.model
        self.sam.image_encoder.requires_grad_(False)
        self.sam.prompt_encoder.requires_grad_(True)
        self.sam.mask_decoder.requires_grad_(True)
        self.matcher = SignedMatcher().to(device)
        self.arm = arm

    @staticmethod
    def rgb(image):
        mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
        std = image.new_tensor([.229, .224, .225])[None, :, None, None]
        return ((image*std+mean)*255).round().clamp(0, 255)

    @torch.no_grad()
    def encode(self, inputs):
        features = []
        for suffix in ['', '_1', '_2']:
            rgb = self.rgb(inputs['image'+suffix])[0].byte().permute(1, 2, 0).cpu().numpy()
            mean = self.sam.pixel_mean.squeeze().detach().cpu().numpy()
            std = self.sam.pixel_std.squeeze().detach().cpu().numpy()
            normalized = torch.as_tensor((rgb.astype(np.float32)-mean)/std).permute(2, 0, 1)[None]
            normalized = normalized.to(inputs['image'+suffix].device)
            value = F.interpolate(normalized, size=(self.sam.image_encoder.img_size,)*2, mode='nearest')
            features.append(self.sam.image_encoder(value))
        size = features[0].shape[-1]
        assert size % 8 == 0
        fused = (features[0] + F.interpolate(features[1][..., size//4:3*size//4, size//4:3*size//4],
                         size=(size, size), mode='bilinear', align_corners=False) +
                 F.interpolate(features[2][..., 3*size//8:5*size//8, 3*size//8:5*size//8],
                         size=(size, size), mode='bilinear', align_corners=False))/3
        return features[0], fused

    def decode(self, se, qe, prompts, local=False):
        zero = qe[0].new_zeros(1, 512, 512)
        prior = self.matcher(se[1], qe[1], prompts,
                             dict(pixel_logits=zero, pixel_probability=zero+.5))['pixel_logits']
        source = F.normalize(se[1].float(), dim=1)
        query = F.normalize(qe[1].float(), dim=1)
        evidence = []
        for sign in ['positive', 'negative']:
            descriptor = F.normalize(sample_assignment_at_cells(source, prompts[sign+'_xy']), dim=1)
            similarity = torch.einsum('bcp,bchw->bphw', descriptor, query)
            valid = prompts[sign+'_mask'].bool()
            evidence.append((similarity*valid[:, :, None, None]).sum(1)/valid.sum(1)[:, None, None])
        prior = prior+F.interpolate((4*(evidence[0]-evidence[1]))[:, None], size=(512, 512),
                                   mode='bilinear', align_corners=False)[:, 0]
        dense_mask = F.interpolate(prior[:, None], size=self.sam.prompt_encoder.mask_input_size,
                                   mode='bilinear', align_corners=False)
        sparse, dense = self.sam.prompt_encoder(points=points(prompts, size=self.sam.image_encoder.img_size)
                         if local else None, boxes=None, masks=dense_mask)
        masks, _ = self.sam.mask_decoder(image_embeddings=qe[0], image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        logits = self.sam.postprocess_masks(masks, (self.sam.image_encoder.img_size,)*2, (512, 512))[:, 0]
        return dict(pixel_logits=logits, pixel_probability=logits.sigmoid())


class DecoderData(SharedData):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(cache, 'match', seed)


def frozen_backbone_state(model):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name not in trainable}


def frozen_decoder_state(model):
    return {}


def build(arm, checkpoint=None):
    model = DecoderModel(arm)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'])
    return model
