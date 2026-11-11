"""Independent E/H/G candidates on original public-fresh Phikon FewClick.
No training/evaluation labels or cross-query memory enter these modules.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Parent,DecoderData as ParentData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state,frozen_decoder_state

ARMS=('E','H','G')

def signed_similarity(sf,qf,prompts):
    values=[]
    for sign in ['positive','negative']:
        valid=prompts[sign+'_mask'].bool()
        assert valid.any(1).all()
        point=F.normalize(sample_assignment_at_cells(sf,prompts[sign+'_xy'].float()),dim=1)
        corr=torch.einsum('bcp,bchw->bphw',point,F.normalize(qf,dim=1))
        values.append(corr.masked_fill(~valid[:,:,None,None],-2).amax(1,keepdim=True))
    return values

class SelfSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(7,32,1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,sf,qf,prompts,prob):
        p=F.interpolate(prob.detach()[:,None],size=qf.shape[-2:],mode='bilinear',align_corners=False)
        pos,neg=signed_similarity(sf,qf,prompts)
        wp=((p>=.8)&(pos.detach()>=neg.detach()+.05)).to(p.dtype)*p.square()
        wn=((p<=.2)&(neg.detach()>=pos.detach()+.05)).to(p.dtype)*(1-p).square()
        active=((wp>0).sum((2,3))>=4).to(p.dtype)[:,:,None,None]
        unit=F.normalize(qf,dim=1)
        def self_similarity(weight):
            proto=F.normalize((unit*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1e-8),dim=1)
            return (unit*proto[:,:,None,None]).sum(1,keepdim=True)
        qp,qn=self_similarity(wp),self_similarity(wn)
        features=torch.cat([pos,neg,qp,qn,qp-qn,p,4*p*(1-p)],1)
        delta=2*self.head(features).tanh()*active
        return delta,{'active':active.mean(),'positive_cells':(wp>0).sum().detach()}

class DenseHierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(8,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def clicked_support_probability(source,prompts):
        p=F.interpolate(source.detach()[:,None],size=(32,32),mode='bilinear',align_corners=False)
        axis=(torch.arange(32,device=p.device,dtype=p.dtype)+.5)/32
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack([xx,yy],-1)[None,None]
        fields=[]
        for sign in ['positive','negative']:
            dist=(xy-prompts[sign+'_xy'][:,:,None,None,:]).square().sum(-1)
            field=(torch.exp(-dist/(2*(1/32)**2))*prompts[sign+'_mask'][:,:,None,None]).sum(1,keepdim=True)
            fields.append(field)
        return (p+4*fields[0])/(1+4*fields[0]+4*fields[1])

    def forward(self,sf,qf,prompts,source,prob):
        assert sf.shape[-2:]==qf.shape[-2:]==(64,64)
        s32=F.normalize(F.avg_pool2d(sf,2),dim=1);q32=F.normalize(F.avg_pool2d(qf,2),dim=1)
        s16=F.normalize(F.avg_pool2d(sf,4),dim=1);q16=F.normalize(F.avg_pool2d(qf,4),dim=1)
        coarse=torch.bmm(q16.flatten(2).transpose(1,2),s16.flatten(2))/.1
        prior=coarse.softmax(-1)
        expanded=prior.reshape(-1,16,16,16,16)
        for dim in [1,2,3,4]:expanded=expanded.repeat_interleave(2,dim=dim)
        expanded=expanded.reshape(-1,1024,1024)/4
        fine=torch.bmm(q32.flatten(2).transpose(1,2),s32.flatten(2))/.1
        affinity=(fine+.5*expanded.clamp_min(1e-8).log()).softmax(-1)
        support_p=self.clicked_support_probability(source,prompts)
        transport=torch.bmm(affinity,support_p.flatten(2).transpose(1,2)).transpose(1,2).reshape(-1,1,32,32)
        coarse_p=torch.bmm(prior,F.avg_pool2d(support_p,2).flatten(2).transpose(1,2)).transpose(1,2).reshape(-1,1,16,16)
        coarse_p=F.interpolate(coarse_p,size=(32,32),mode='bilinear',align_corners=False)
        certainty=1-(-(affinity*affinity.clamp_min(1e-8).log()).sum(-1)/math.log(1024)).reshape(-1,1,32,32)
        pos,neg=signed_similarity(sf,F.avg_pool2d(qf,2),prompts)
        p=F.interpolate(prob[:,None],size=(32,32),mode='bilinear',align_corners=False)
        features=torch.cat([transport,coarse_p,certainty,p,4*p*(1-p),pos,neg,pos-neg],1)
        delta=2*self.head(features).tanh()
        return delta,{'active':delta.new_ones(()),'transport_std':transport.std().detach()}

class TiledFineBackbone(nn.Module):
    """Four nonoverlapping256 crops resized512, native32 tokens stitched64.
    The public backbone is registered on the two untouched context encoders.
    """
    def __init__(self,backbone):
        super().__init__();object.__setattr__(self,'_backbone',backbone)

    @torch.no_grad()
    def forward(self,image):
        assert image.shape==(1,3,512,512)
        mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
        std=image.new_tensor([.229,.224,.225])[None,:,None,None]
        rgb=((image*std+mean)*255).round().clamp(0,255).byte()
        rows=[]
        for y in range(2):
            cols=[]
            for x in range(2):
                crop=rgb[...,y*256:(y+1)*256,x*256:(x+1)*256].float()
                enlarged=F.interpolate(crop,size=(512,512),mode='bilinear',align_corners=False).round().clamp(0,255).byte()
                images=[v.permute(1,2,0).cpu().numpy() for v in enlarged]
                pixel=self._backbone.processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values'].to(image.device)
                tokens=self._backbone.encoder(pixel_values=pixel).last_hidden_state[:,1:]
                assert tokens.shape==(1,1024,1024)
                cols.append(tokens.transpose(1,2).reshape(1,1024,32,32))
            rows.append(torch.cat(cols,-1))
        value=torch.cat(rows,-2)
        return {'out':torch.cat([value,value],1)/math.sqrt(2)}

class DecoderModel(Parent):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS
        super().__init__('match',seed,device);self.arm=arm;self.extension=None
        if arm=='G':
            self.base.core.encoders[0].backbone=TiledFineBackbone(self.base.core.encoders[0].backbone)
        else:
            devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed+1301)
                self.extension=(SelfSupport() if arm=='E' else DenseHierarchy()).to(device)

    def cached_decode(self,se,qe,prompts,local=False,with_support=True):
        task=self.base.core.support_task(se,prompts)
        out,maps=self.base.core.decode(qe,task,prompts,local)
        out=self.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,out)
        source=out if local else None
        if not local and (with_support or self.arm=='H'):
            source=self.base.core.decode(se,task,prompts,True)[0]
            source=self.pixel(se['pixel_embedding'],se['pixel_embedding'],prompts,source)
        if not local and self.extension is not None:
            sf=self.pixel.project(se['pixel_embedding'].float());qf=self.pixel.project(qe['pixel_embedding'].float())
            if self.arm=='E':delta,diag=self.extension(sf,qf,prompts,out['pixel_probability'])
            else:delta,diag=self.extension(sf,qf,prompts,source['pixel_probability'],out['pixel_probability'])
            residual=F.interpolate(delta,size=(512,512),mode='bilinear',align_corners=False)[:,0]
            logits=out['pixel_logits']+residual
            out=dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),extension_active=diag['active'],extension_residual_mean=residual.abs().mean())
        return out,source if with_support else None,task,maps

    def forward(self,support,query,prompts,local,shared,with_support=True):
        assert not local or shared
        support=self.base.transform(support);query=support if shared else self.base.transform(query)
        se=self.encode(support);qe=se if shared else self.encode(query)
        out,source,task,maps=self.cached_decode(se,qe,prompts,local,with_support)
        reg=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
        return dict(query=out,support=source,task_token=task['task_token'],query_prompt_maps=maps,assignment_regularization=reg)

class DecoderData(ParentData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'match',seed)

def build(arm,checkpoint=None):
    m=DecoderModel(arm)
    if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
    return m
