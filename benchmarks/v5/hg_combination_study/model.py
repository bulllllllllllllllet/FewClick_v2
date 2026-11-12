"""Single-model H+G and a fixed whole-fine/tiled-fine fusion control."""
import torch
from benchmarks.v5.parallel_phikon_study.model import DecoderModel as HModel,TiledFineBackbone
from benchmarks.v5.support_match_study.model import DecoderData as ParentData
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state
ARMS=('HG','HG_mix')
class MixedFineBackbone(TiledFineBackbone):
    @torch.no_grad()
    def forward(self,image):
        tiled=super().forward(image)['out']
        whole=self._backbone(image)['out']
        return {'out':.5*(whole+tiled)}
class DecoderModel(HModel):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS
        super().__init__('H',seed,device)
        # Keep inherited self.arm='H' to preserve H support decoding in cached inference.
        self.study_arm=arm
        wrapper=TiledFineBackbone if arm=='HG' else MixedFineBackbone
        self.base.core.encoders[0].backbone=wrapper(self.base.core.encoders[0].backbone)
class DecoderData(ParentData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'match',seed)
def build(arm,checkpoint=None):
    m=DecoderModel(arm)
    if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
    return m
