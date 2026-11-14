import json, hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.whole_slide_once.data import prompts
from benchmarks.v5.encoder_research.continuous_data import sha


def load(method,checkpoint):
    if method=='HG_mix':
        from benchmarks.v5.hg_combination_study.model import build
        original=build(method,checkpoint).eval();return original,original
    if method=='sansa':
        from benchmarks.v5.same_slide_retrain.model import build
        from benchmarks.v5.sansa_cache.model import CachedSANSA
        original=build(method,checkpoint).eval();return CachedSANSA(original).eval(),original
    from benchmarks.v5.whole_slide_sam.worker import load as sam_load
    original=sam_load(method,checkpoint,'cuda:0');return original,original


def encode(model,x,method):
    return model.encode(model.base.transform(x)) if method=='HG_mix' else model.encode(x)


def decode(model,se,qe,points,method,local=False):
    if method=='HG_mix':return model.cached_decode(se,qe,points,local,with_support=False)[0]['pixel_probability']
    return model.decode(se,qe,points,local)['pixel_probability']


def clear(model):
    if hasattr(model,'support_memory'):model.support_memory.clear()


class Inputs:
    def __init__(self,rows):self.rows=rows;self.hashes={}
    def get(self,identity):
        x={};hashes=[]
        for i in range(3):
            rgb=np.asarray(Image.fromarray(native_context(self.rows[identity],i)).resize((512,512),Image.Resampling.BILINEAR)).copy()
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest());suffix='' if i==0 else '_'+str(i)
            x['image'+suffix]=he_to_tensor(rgb)[None].cuda()
            x['cells'+suffix]=torch.zeros(1,0,74,device='cuda:0')
            x['cell_valid'+suffix]=torch.zeros(1,0,device='cuda:0',dtype=torch.bool)
            x['total_cell_count' if i==0 else 'total_'+str(i)]=torch.zeros(1,device='cuda:0')
        if identity in self.hashes:assert self.hashes[identity]==hashes
        self.hashes[identity]=hashes
        return x


def tensor_bytes(value):
    seen=set()
    def visit(x):
        if isinstance(x,torch.Tensor):
            storage=x.untyped_storage();key=(str(x.device),storage.data_ptr())
            if key in seen:return 0
            seen.add(key);return storage.nbytes()
        if isinstance(x,dict):return sum(visit(v) for v in x.values())
        if isinstance(x,(list,tuple)):return sum(visit(v) for v in x)
        if hasattr(x,'__dict__') and not isinstance(x,torch.nn.Module):return visit(vars(x))
        return 0
    return visit(value)


def verify(root):
    c=json.loads((root/'contract.json').read_text())
    for p,h in c['sources'].items():assert sha(p)==h,p
    for p,h in c['files'].items():assert sha(root/p)==h,p
    return c
