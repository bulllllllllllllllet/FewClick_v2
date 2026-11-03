"""Matched image-only graph decoder with an optional task-conditioned pixel path."""
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData, SpatialModel

ARMS = ('region', 'pixel')


class DecoderData(SpatialData):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(cache, 'I', seed)


class PixelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep = nn.Sequential(nn.Conv2d(256,48,1),nn.GroupNorm(8,48),nn.GELU())
        self.task = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,96))
        self.context = nn.Sequential(nn.Conv2d(50,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),
                                     nn.Conv2d(48,48,3,padding=1),nn.GELU())
        self.rgb2 = nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.GELU())
        self.rgb4 = nn.Sequential(nn.Conv2d(16,24,3,stride=2,padding=1),nn.GroupNorm(8,24),nn.GELU())
        self.fuse4 = nn.Sequential(nn.Conv2d(72,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU())
        self.fuse2 = nn.Sequential(nn.Conv2d(64,24,3,padding=1),nn.GroupNorm(8,24),nn.GELU())
        self.output = nn.Conv2d(24,1,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, embedding, rgb, task, base):
        h = self.deep(embedding.float())
        scale,shift = self.task(task.float()).chunk(2,dim=1)
        h = h*(1+.1*scale[:,:,None,None].tanh())+shift[:,:,None,None]
        probability = F.interpolate(base['pixel_probability'][:,None],size=h.shape[-2:],mode='bilinear',align_corners=False)
        h = self.context(torch.cat([h,probability,4*probability*(1-probability)],1))
        rgb2 = self.rgb2(rgb.float()); rgb4 = self.rgb4(rgb2)
        h = self.fuse4(torch.cat([F.interpolate(h,size=rgb4.shape[-2:],mode='bilinear',align_corners=False),rgb4],1))
        h = self.fuse2(torch.cat([F.interpolate(h,size=rgb2.shape[-2:],mode='bilinear',align_corners=False),rgb2],1))
        residual = F.interpolate(self.output(h),size=base['pixel_logits'].shape[-2:],mode='bilinear',align_corners=False)[:,0]
        logits = base['pixel_logits']+residual
        return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),residual=residual)


class DecoderModel(nn.Module):
    def __init__(self, arm, seed=20261103, device='cuda:0'):
        super().__init__(); assert arm in ARMS
        self.arm=arm; self.base=SpatialModel('I',seed,device); self.pixel=None
        if arm=='pixel':
            devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed+901)
                self.pixel=PixelResidual().to(device)
        self.pixel_enabled=True

    def train(self, mode=True):
        super().train(mode)
        self.base.train(mode)  # Old frozen backbone/BN/dropout policy.
        return self

    def encode(self, inputs):
        captured={}
        handle=self.base.core.encoders[0].embedding.register_forward_hook(lambda m,i,o:captured.update(embedding=o))
        try: encoded=self.base.core.encode_image(inputs)
        finally:handle.remove()
        return encoded|dict(pixel_embedding=captured['embedding'],pixel_rgb=inputs['image'])

    def decode(self, encoded, task, prompts, local):
        result,maps=self.base.core.decode(encoded,task,prompts,local)
        if self.pixel is not None and self.pixel_enabled:
            result=self.pixel(encoded['pixel_embedding'],encoded['pixel_rgb'],task['task_token'],result)
        return result,maps

    def forward(self,support,query,prompts,local,shared,with_support=True):
        support=self.base.transform(support);query=support if shared else self.base.transform(query)
        se=self.encode(support);qe=se if shared else self.encode(query)
        task=self.base.core.support_task(se,prompts)
        output,maps=self.decode(qe,task,prompts,local)
        source=output if shared else self.decode(se,task,prompts,True)[0] if with_support else None
        regularization=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
        return dict(query=output,support=source,query_prompt_maps=maps,task_token=task['task_token'],
                    assignment_regularization=regularization)
