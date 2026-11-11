"""F: FewClick signed multiscale conditions + frozen public SAM2 native decoder.

Only image tensors and support clicks enter inference. No SANSA Universal weights,
query clicks, mask candidate selection, temporal memory or cross-query state.
"""
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Parent, DecoderData as ParentData
from benchmarks.v5.additional_sam.model import ROOT, rgb
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state

ARMS = ('F',)
WEIGHT = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/reference_baseline_1109_040918/SANSA/pretrain/sam2_hiera_large.pt')
CONFIG = 'configs/sam2/sam2_hiera_l.yaml'

class NativeBackbone(nn.Module):
    def __init__(self, sam, transform):
        super().__init__()
        # SAM is registered exactly once on DecoderModel, shared across all views.
        object.__setattr__(self, '_sam', sam)
        object.__setattr__(self, '_transform', transform)

    @torch.no_grad()
    def forward(self, image):
        tensor = self._transform(rgb(image)).to(image.device)[None]
        features = self._sam.forward_image(tensor)
        deep = features['backbone_fpn'][-1]
        assert deep.shape == (1, 256, 64, 64)
        native_deep = deep
        if self._sam.directly_add_no_mem_embed:
            native_deep = deep + self._sam.no_mem_embed.reshape(1,256,1,1)
        return {'out': deep.repeat(1,8,1,1)/math.sqrt(8),
                'native': {'image_embed': native_deep,
                           'high_res_feats': features['backbone_fpn'][:-1]}}

class DecoderModel(Parent):
    def __init__(self, arm='F', seed=20261103, device='cuda:0'):
        assert arm in ARMS and device == 'cuda:0'
        super().__init__('match',seed,device)
        sys.path[:0] = [str(ROOT/'isolated_deps'),str(ROOT/'sam2_source')]
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.sam = build_sam2(CONFIG,str(WEIGHT),device=device,apply_postprocessing=False).eval().requires_grad_(False)
        transform = SAM2ImagePredictor(self.sam)._transforms
        backbone = NativeBackbone(self.sam,transform)
        for encoder in self.base.core.encoders:
            encoder.backbone = backbone
        self.arm = arm

    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        return self

    def encode(self, inputs):
        captured = []
        hook = self.base.core.encoders[0].backbone.register_forward_hook(
            lambda module,args,out: captured.append(out['native']))
        try:
            result = super().encode(inputs)
        finally:
            hook.remove()
        assert len(captured) == 3
        return result | {'native': captured[0]}

    def native_refine(self, encoded, coarse, prompts, local):
        dense = F.interpolate(coarse['pixel_logits'][:,None],size=(256,256),mode='bilinear',align_corners=False)
        points = None
        if local:
            xy=[]; labels=[]
            for sign,label in [('positive',1),('negative',0)]:
                mask=prompts[sign+'_mask'][0].bool()
                xy.append(prompts[sign+'_xy'][0][mask]);labels += [label]*int(mask.sum())
            points=(torch.cat(xy)[None]*1024,torch.tensor([labels],device=dense.device,dtype=torch.int32))
        sparse, dense_embeddings = self.sam.sam_prompt_encoder(points=points,boxes=None,masks=dense)
        native, _, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=encoded['native']['image_embed'],
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,repeat_image=False,
            high_res_features=encoded['native']['high_res_feats'])
        native = F.interpolate(native,size=(512,512),mode='bilinear',align_corners=False)[:,0]
        # Fixed sum, not a validation-tuned blend. Coarse log odds retain absence evidence.
        logits = native + coarse['pixel_logits']
        return dict(pixel_logits=logits,pixel_probability=logits.sigmoid(),
                    native_logits=native,coarse_logits=coarse['pixel_logits'])

    def cached_decode(self,se,qe,prompts,local=False,with_support=True):
        task=self.base.core.support_task(se,prompts)
        coarse,maps=self.base.core.decode(qe,task,prompts,local)
        coarse=self.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,coarse)
        out=self.native_refine(qe,coarse,prompts,local)
        source=None
        if local:
            source=out
        elif with_support:
            sc=self.base.core.decode(se,task,prompts,True)[0]
            sc=self.pixel(se['pixel_embedding'],se['pixel_embedding'],prompts,sc)
            source=self.native_refine(se,sc,prompts,True)
        return out,source,task,maps

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

def frozen_decoder_state(m):
    return m.sam.sam_mask_decoder.state_dict()

def build(arm,checkpoint=None):
    m=DecoderModel(arm)
    if checkpoint:
        m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:0')['model'])
    return m
