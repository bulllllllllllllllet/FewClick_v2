"""Same signed matcher; enable the accepted original XCell region fusion."""
from benchmarks.v5.support_match_study.model import DecoderModel as Original
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
class DecoderData(SpatialData):
 def __init__(self,cache,arm,seed=20261103):
  assert arm in ['cell_on','cell_off'];super().__init__(cache,'I',seed)
class DecoderModel(Original):
 def __init__(self,arm,seed=20261103,device='cuda:0'):
  assert arm in ['cell_on','cell_off'];super().__init__('match',seed,device);self.study_arm=arm
  if arm=='cell_on':
   self.base.arm='S11';self.base.core.cell.arm='S11'
