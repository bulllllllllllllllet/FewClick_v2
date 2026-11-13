"""Two one-factor ablations of fixed HG-mix, fresh matched training."""
import torch
from benchmarks.v5.hg_combination_study.model import DecoderModel as HGmix
from benchmarks.v5.support_match_study.model import DecoderData as ParentData
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state
from .no_cross_scale import IdentityHierarchy,NoCrossScaleCore
ARMS=('M','no_cross_scale')
class DecoderModel(HGmix):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS
        super().__init__('HG_mix',seed,device)
        self.study_arm=arm
        if arm=='M':
            self.extension=None
            self.arm='M'
        else:
            # Preserve parameters/state identities and all physical inputs.
            # The bypassed hierarchy has no active trainable parameters.
            self.base.core.hierarchy.__class__=IdentityHierarchy
            self.base.core.hierarchy.requires_grad_(False)
            self.base.core.__class__=NoCrossScaleCore
class DecoderData(ParentData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'match',seed)
def build(arm,checkpoint=None):
    m=DecoderModel(arm)
    if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
    return m
