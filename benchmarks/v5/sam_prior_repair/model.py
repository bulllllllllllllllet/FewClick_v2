"""Native SAM decoder with a trainable support-to-query dense prompt bridge."""
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.fair_comparison.adapted_models import SAMFineTune,points
from benchmarks.v5.support_match_study.model import SignedMatcher
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData
ARMS=('sam_native',)
class DecoderData(SpatialData):
    def __init__(self,cache,arm,seed=20261103):
        assert arm in ARMS
        super().__init__(cache,'I',seed)
class DecoderModel(nn.Module):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        super().__init__();assert arm in ARMS
        torch.manual_seed(seed)
        wrapped=SAMFineTune(device);self.sam=wrapped.sam;self.resize=wrapped.transform
        self.matcher=SignedMatcher().to(device);self.arm=arm
        self.sam.image_encoder.requires_grad_(False)
    @property
    def base(self):return self.sam
    def train(self,mode=True):
        super().train(mode);self.sam.image_encoder.eval();return self
    @torch.no_grad()
    def encode(self,inputs):
        values=[]
        for suffix in ['', '_1','_2']:
            image=inputs['image'+suffix]
            mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
            std=image.new_tensor([.229,.224,.225])[None,:,None,None]
            rgb=((image*std+mean)*255).round().clamp(0,255).byte()[0].permute(1,2,0).cpu().numpy()
            transformed=self.resize.apply_image(rgb)
            value=torch.as_tensor(transformed,device=image.device).permute(2,0,1)
            values.append(self.sam.image_encoder(self.sam.preprocess(value)[None]))
        # Same physical fine extent: central 1/2 of 2x and central 1/4 of 4x.
        fusion=(values[0]+F.interpolate(values[1][...,16:48,16:48],size=(64,64),mode='bilinear',align_corners=False)+
                F.interpolate(values[2][...,24:40,24:40],size=(64,64),mode='bilinear',align_corners=False))/3
        return values[0],fusion
    def decode(self,se,qe,prompts,local):
        zero=qe[0].new_zeros(qe[0].shape[0],512,512)
        base=dict(pixel_logits=zero,pixel_probability=zero+.5)
        prior=self.matcher(se[1],qe[1],prompts,base)['pixel_logits']
        # A nonzero signed cosine prior carries target identity from the first update.
        source=F.normalize(se[1].float(),dim=1);target=F.normalize(qe[1].float(),dim=1)
        evidence=[]
        for sign in ['positive','negative']:
            descriptor=F.normalize(sample_assignment_at_cells(source,prompts[sign+'_xy']),dim=1)
            similarity=torch.einsum('bcp,bchw->bphw',descriptor,target)
            valid=prompts[sign+'_mask'].bool()
            evidence.append((similarity*valid[:,:,None,None]).sum(1)/valid.sum(1)[:,None,None])
        initial=F.interpolate((4*(evidence[0]-evidence[1]))[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0]
        prior=prior+initial
        dense_mask=F.interpolate(prior[:,None],size=(256,256),mode='bilinear',align_corners=False)
        sparse,dense=self.sam.prompt_encoder(points=points(prompts) if local else None,boxes=None,masks=dense_mask)
        masks,_=self.sam.mask_decoder(image_embeddings=qe[0],image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
        logits=self.sam.postprocess_masks(masks,(1024,1024),(512,512))[:,0]
        return dict(pixel_logits=logits,pixel_probability=logits.sigmoid())
    def forward(self,support,query,prompts,local,shared,with_support=True):
        se=self.encode(support);qe=se if shared else self.encode(query)
        output=self.decode(se,qe,prompts,local)
        source=output if shared else self.decode(se,se,prompts,True) if with_support else None
        maps=output['pixel_logits'].new_zeros(1,2,128,128)
        return dict(query=output,support=source,query_prompt_maps=maps,task_token=se[0].mean((2,3)),
            assignment_regularization=output['pixel_logits'].new_zeros(()))
