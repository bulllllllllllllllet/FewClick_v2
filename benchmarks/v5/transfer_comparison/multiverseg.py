"""Pinned official MultiverSeg context branch, without changing global packages."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from benchmarks.v5.fair_comparison.audit import digest


class MultiverSegContext:
    def __init__(self,sources,weight,dependencies,device):
        sources=Path(sources);weight=Path(weight);dependencies=Path(dependencies)
        self.sources=json.loads(sources.read_text());self.weight=json.loads(weight.read_text());self.device=torch.device(device)
        for name in ('MultiverSeg','UniverSeg','ScribblePrompt'):
            record=self.sources[name];root=Path(record['root'])
            if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=record['commit']:
                raise ValueError('official source commit changed')
            if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip():
                raise ValueError('tracked official source modified')
            namespace={'MultiverSeg':'multiverseg','UniverSeg':'universeg','ScribblePrompt':'scribbleprompt'}[name]
            if namespace in sys.modules:
                location=getattr(sys.modules[namespace],'__file__',None)
                if not location or not str(Path(location).resolve()).startswith(str(root.resolve())+'/'):
                    raise ValueError('conflicting installed context method')
            sys.path.insert(0,str(root))
        sys.path.insert(0,str(dependencies))
        checkpoint=Path(self.weight['output'])
        if digest(checkpoint)!=self.weight['sha256']:raise ValueError('MultiverSeg weight identity changed')
        network=importlib.import_module('multiverseg.models.network')
        self.model=network.MultiverSegNet(in_channels=[5,2],encoder_blocks=[256]*4,
            block_kws=dict(conv_kws=dict(norm='layer')),cross_relu=True).to(self.device)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(payload['model'],strict=True);self.model.eval()
        self.native_class=importlib.import_module('multiverseg.models.sp_mvs').MultiverSeg
        self.prepare=importlib.import_module('scribbleprompt.models.unet').prepare_inputs
        self.prior=dict(official_sources={k:self.sources[k] for k in ('MultiverSeg','UniverSeg','ScribblePrompt')},
            weight=self.weight,weights_used='MultiverSeg v1 only; no ScribblePrompt weights or branch',
            inputs='one gray128 source image and complete GT/predicted source mask; gray128 query, no query interactions',
            preprocessing='official WBC example: PIL bilinear128 then convert L, float/255; support mask nearest128',
            training_overlap='not audited; public prior, no local adaptation',threshold=.5,
            dependencies=dict(tabulate_version=importlib.import_module('tabulate').__version__,
                              tabulate_source_sha256=digest(dependencies/'tabulate/__init__.py')))

    def gray(self,rgb):
        if isinstance(rgb,torch.Tensor):rgb=rgb[0].permute(1,2,0).cpu().numpy()
        image=Image.fromarray(np.asarray(rgb,np.uint8)).resize((128,128),Image.Resampling.BILINEAR).convert('L')
        return torch.from_numpy(np.asarray(image).copy()).float()[None,None].to(self.device)/255.

    def predict(self,source_rgb,query_rgb,source_mask,*,mask_kind,source_valid=None,verify_native=False):
        if mask_kind=='full_GT':
            if source_valid is None or np.asarray(source_valid).shape!=(512,512) or not np.asarray(source_valid).all():
                raise ValueError('full-GT context requires all source pixels annotated; ignore pixels cannot become negatives')
        elif mask_kind!='model_prediction':
            raise ValueError('declare a complete GT context or model-predicted full mask; sparse annotations are not dense context')
        mask=np.asarray(source_mask)
        if mask.shape!=(512,512) or not np.isfinite(mask).all() or not np.isin(mask,[0,1]).all():
            raise ValueError('explicit complete binary source mask required; sparse/unannotated mask is not accepted')
        source=self.gray(source_rgb);query=self.gray(query_rgb)
        resized=Image.fromarray(mask.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST)
        label=torch.from_numpy(np.asarray(resized).copy()).float()[None,None,None].to(self.device)
        context=source[:,None]
        if context.shape[1]!=1:raise ValueError('this contract requires exactly one genuine context')
        target=self.prepare(dict(img=query))
        if target[:,1:].count_nonzero():raise ValueError('unexpected query interactions or previous prediction')
        with torch.inference_mode():
            logits=self.model(target,context,label)
            if verify_native:
                # Execute the official wrapper's actual context branch. Its
                # unrelated no-context ScribblePrompt branch is not constructed.
                holder=SimpleNamespace(multiverseg=self.model,min_context=1)
                expected=self.native_class.forward(holder,target,context,label)
                if not torch.equal(logits,expected):raise ValueError('official context forward mismatch')
            logits=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False)[:,0]
            probability=logits.sigmoid()
        if probability.shape!=(1,512,512) or not torch.isfinite(probability).all():raise ValueError('invalid context prediction')
        return dict(pixel_logits=logits,pixel_probability=probability,query_interactions=0,
                    native_forward_identical=verify_native,mask_kind=mask_kind)
