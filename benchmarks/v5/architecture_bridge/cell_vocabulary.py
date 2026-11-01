"""Capacity-matched continuous/fixed/EMA cell representations for P2 engineering gates."""
import torch
from torch import nn


class CellVocabulary(nn.Module):
    def __init__(self,mode,centers):
        super().__init__();assert mode in ['continuous','fixed','ema']
        assert centers.shape==(128,64) and torch.isfinite(centers).all();self.mode=mode
        self.mapper=nn.Linear(64,64);nn.init.eye_(self.mapper.weight);nn.init.zeros_(self.mapper.bias)
        self.register_buffer('centers',None if mode=='continuous' else centers.clone().float())
        self.register_buffer('counts',torch.ones(128) if mode=='ema' else None)
        self.register_buffer('sums',centers.clone().float() if mode=='ema' else None)

    def forward(self,x):
        assert x.ndim==2 and x.shape[1]==64 and torch.isfinite(x).all()
        z=self.mapper(x)
        if self.mode=='continuous':return z,None
        distances=z.square().sum(1,keepdim=True)+self.centers.square().sum(1)[None]-2*z@self.centers.T
        indices=distances.argmin(1);q=self.centers[indices]
        return z+(q-z).detach(),indices

    @torch.no_grad()
    def update_codebook(self,train_vectors):
        if not self.training:raise RuntimeError('Codebook updates forbidden in evaluation')
        if self.mode!='ema' or len(train_vectors)==0:return
        z=self.mapper(train_vectors).detach();_,indices=self.forward(train_vectors)
        weights=torch.nn.functional.one_hot(indices,128).to(z.dtype)
        count=weights.sum(0);total=weights.T@z
        self.counts.mul_(.99).add_(count,alpha=.01);self.sums.mul_(.99).add_(total,alpha=.01)
        self.centers.copy_(self.sums/self.counts.clamp_min(1e-5)[:,None])
        assert torch.isfinite(self.centers).all()
