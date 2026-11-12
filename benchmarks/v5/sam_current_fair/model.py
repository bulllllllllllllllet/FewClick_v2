"""Reuse accepted native SAM bridges; only the task queue/budget is updated."""
from pathlib import Path
import torch
from benchmarks.v5.support_match_study.model import DecoderData as SharedData

ARMS = ('sam', 'sam2', 'wsi')


def DecoderModel(arm, seed=20261103, device='cuda:0'):
    assert arm in ARMS
    if arm == 'sam':
        from benchmarks.v5.sam_prior_repair.model import DecoderModel as Native
        return Native('sam_native', seed, device)
    if arm == 'sam2':
        from benchmarks.v5.sam_matched_adaptation.model import DecoderModel as Native
        return Native('sam2_ft', seed, device)
    from benchmarks.v5.wsi_sparse_bridge.model import DecoderModel as Native
    return Native('wsi_sam_ft', seed, device)


class DecoderData(SharedData):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(Path(cache), 'match', seed)


def frozen_backbone_state(model):
    trainable = {n for n, v in model.named_parameters() if v.requires_grad}
    return {n: v for n, v in model.state_dict().items() if n not in trainable}


def frozen_decoder_state(model):
    if getattr(model, 'arm', None) == 'wsi_sam_ft':
        from benchmarks.v5.wsi_author_freeze.model import frozen_decoder_state as state
        return state(model)
    return {}


def build(arm, checkpoint=None):
    model = DecoderModel(arm)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'])
    return model
