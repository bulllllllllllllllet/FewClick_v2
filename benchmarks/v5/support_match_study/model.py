"""Positive/negative point-to-query spatial matching; no query labels or coordinates."""
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.pixel_decoder_study.model import DecoderModel as RegionWrapper
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
ARMS=('control','match')
class DecoderData(SpatialData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'I',seed)
class SignedMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Conv2d(256,64,1)
        self.hidden=nn.Sequential(nn.Conv2d(7,32,1),nn.GELU())
        self.output=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,support,query,prompts,base):
        se=self.project(support.float());qe=F.normalize(self.project(query.float()),dim=1)
        evidence=[]
        for sign in ['positive','negative']:
            valid=prompts[sign+'_mask'].bool()
            assert valid.any(1).all(), 'Require at least one positive and negative support point'
            descriptors=sample_assignment_at_cells(se,prompts[sign+'_xy'].float())
            descriptors=F.normalize(descriptors,dim=1)
            sim=torch.einsum('bcp,bchw->bphw',descriptors,qe)
            evidence.extend([sim.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True),
                (sim*valid[:,:,None,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None,None]])
        probability=F.interpolate(base['pixel_probability'][:,None],size=qe.shape[-2:],mode='bilinear',align_corners=False)
        features=torch.cat([*evidence,evidence[0]-evidence[2],probability,4*probability*(1-probability)],1)
        residual=F.interpolate(self.output(self.hidden(features)),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
        logits=base['pixel_logits']+residual
        return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),residual=residual)
class DecoderModel(RegionWrapper):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS
        super().__init__('region',seed,device);self.arm=arm
        if arm=='match':
            devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed+902);self.pixel=SignedMatcher().to(device)
    def forward(self,support,query,prompts,local,shared,with_support=True):
        support=self.base.transform(support);query=support if shared else self.base.transform(query)
        se=self.encode(support);qe=se if shared else self.encode(query)
        task=self.base.core.support_task(se,prompts)
        output,maps=self.base.core.decode(qe,task,prompts,local)
        if self.pixel is not None and self.pixel_enabled:
            output=self.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,output)
        if shared:source=output
        elif with_support:
            source=self.base.core.decode(se,task,prompts,True)[0]
            if self.pixel is not None and self.pixel_enabled:
                source=self.pixel(se['pixel_embedding'],se['pixel_embedding'],prompts,source)
        else:source=None
        regularization=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
        return dict(query=output,support=source,query_prompt_maps=maps,task_token=task['task_token'],assignment_regularization=regularization)
