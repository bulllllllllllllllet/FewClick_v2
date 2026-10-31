"""Pinned, frozen LEMON encoder for the already audited 10 micrometre crops."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torchvision.transforms import Compose, Normalize, ToTensor


SOURCE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/encoder_inventory_1030_182424/LEMON')
WEIGHTS = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/lemon_weights_1031_004350/data/lemon_vits8.pth.tar')
MODEL_SHA = '946199ff11d3a358bbea531aced36ff02b49648f8b7fd752b1b6d8c772a84d26'
WEIGHTS_SHA = '1529389956ca0a128771c6481f0c9448682f231121f919c1c27eda54da0cb62e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenLemon:
    def __init__(self, device='cuda:0'):
        assert sha(SOURCE / 'model.py') == MODEL_SHA
        assert sha(WEIGHTS) == WEIGHTS_SHA
        spec = importlib.util.spec_from_file_location('pinned_continuous_lemon', SOURCE / 'model.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        model = official.build_lemon_moco_backbone('vits8')
        model.load_state_dict(official.extract_backbone_state_dict(state), strict=True)
        self.device = device
        self.model = model.to(device).eval().requires_grad_(False)
        stats = json.loads((SOURCE / 'mean_std.json').read_text())
        self.transform = Compose([ToTensor(), Normalize(stats['mean'], stats['std'])])

    @torch.inference_mode()
    def encode(self, images):
        assert images and all(im.size == (40, 40) for im in images)
        value = self.model(torch.stack([self.transform(im) for im in images]).to(self.device))
        assert value.shape == (len(images), 384) and torch.isfinite(value).all()
        return value.float().cpu().numpy()

    @staticmethod
    def identity():
        return dict(official_model_sha256=MODEL_SHA, weights_sha256=WEIGHTS_SHA,
                    normalization_sha256=sha(SOURCE / 'mean_std.json'))
