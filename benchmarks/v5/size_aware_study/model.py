"""One support-conditioned segmenter per arm; no target-size routing."""
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as MatchedModel
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells

ARMS=('control','precision','detail')


class DecoderData(SpatialData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'I',seed)


class SupportDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb=nn.Sequential(nn.Conv2d(3,8,3,stride=2,padding=1),nn.GroupNorm(4,8),nn.GELU(),
                               nn.Conv2d(8,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.GELU())
        self.fuse=nn.Sequential(nn.Conv2d(22,24,3,padding=1),nn.GELU(),nn.Conv2d(24,16,3,padding=1),nn.GELU())
        self.output=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,support_rgb,query_rgb,prompts,base):
        se=self.rgb(support_rgb.float());qe=self.rgb(query_rgb.float());normalized=F.normalize(qe,dim=1);evidence=[]
        for sign in ['positive','negative']:
            valid=prompts[sign+'_mask'].bool();assert valid.any(1).all()
            desc=F.normalize(sample_assignment_at_cells(se,prompts[sign+'_xy'].float()),dim=1)
            similarity=torch.einsum('bcp,bchw->bphw',desc,normalized)
            evidence.extend([similarity.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True),
                             (similarity*valid[:,:,None,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None,None]])
        probability=F.interpolate(base['pixel_probability'][:,None],size=qe.shape[-2:],mode='bilinear',align_corners=False)
        features=torch.cat([qe,*evidence,probability,4*probability*(1-probability)],1)
        residual=F.interpolate(self.output(self.fuse(features)),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
        logits=base['pixel_logits']+residual
        return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())


class DecoderModel(MatchedModel):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS
        super().__init__('match',seed,device);self.arm=arm;self.refiner=None;self.detail_enabled=True
        if arm=='detail':
            devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed+903);self.refiner=SupportDetail().to(device)

    def forward(self,support,query,prompts,local,shared,with_support=True):
        result=super().forward(support,query,prompts,local,shared,with_support)
        if self.refiner is not None and self.detail_enabled:
            output=self.refiner(support['image'],query['image'],prompts,result['query'])
            source=output if shared else self.refiner(support['image'],support['image'],prompts,result['support']) if with_support else None
            result=dict(result,query=output,support=source)
        return result|dict(loss_arm=self.arm)
