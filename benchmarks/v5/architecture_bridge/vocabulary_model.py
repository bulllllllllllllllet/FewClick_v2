"""P2 cell representation inserted before unchanged three-view cell-to-region attention."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .cell_vocabulary import CellVocabulary
from .deterministic_model import build_bridge_image
from benchmarks.v5.encoder_research.continuous_data import sha

VOCAB=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p2_selected_vocabulary_1102_121546/full')


class VocabularyFewClick(nn.Module):
    def __init__(self,mode,device='cuda:0',seed=20261103):
        super().__init__();proof=json.loads((VOCAB/'independent_acceptance.json').read_text())
        assert proof['status']=='accepted_train_only_K128_vocabulary' and proof['validation_samples']==0
        assert sha(VOCAB/'vocabulary.npz')==proof['vocabulary_sha256']
        self.core,self.identity=build_bridge_image('phikon',device,seed)
        with np.load(VOCAB/'vocabulary.npz',allow_pickle=False) as f:centers=torch.from_numpy(f['centers'].copy())
        self.vocabulary=CellVocabulary(mode,centers).to(device);self.mode=mode;self.pending=[];self.last_codes=[]
        self.identity.update(mode=mode,cell_source='xcell_standardized64',vocabulary_sha256=proof['vocabulary_sha256'])
        self.eval()

    def train(self,mode=True):
        super().train(mode)
        self.core.eval() # Original fixed BN/dropout policy, gradients still enabled.
        self.vocabulary.train(mode)
        if not mode:self.pending.clear()
        return self

    def transform_cells(self,inputs):
        result=dict(inputs)
        for scale in range(3):
            suffix='' if scale==0 else '_'+str(scale);key='cells'+suffix;cells=inputs[key];valid=inputs['cell_valid'+suffix]
            assert cells.ndim==3 and cells.shape[-1]==74
            x=cells[...,10:].reshape(-1,64);value,codes=self.vocabulary(x)
            result[key]=torch.cat([cells[...,:10],value.reshape(*cells.shape[:2],64)],dim=-1)
            assert torch.equal(result[key][...,:10],cells[...,:10])
            if self.training and self.mode=='ema':self.pending.append(x[valid.flatten()].detach())
            if codes is not None:self.last_codes.append(codes.detach())
        return result

    def forward(self,support,query,prompts,local,shared,with_support=True):
        self.last_codes=[];support_value=self.transform_cells(support)
        query_value=support_value if shared else self.transform_cells(query)
        return self.core(support_value,query_value,prompts,local,shared,with_support)

    @torch.no_grad()
    def after_optimizer_step(self):
        if not self.training:raise RuntimeError('EMA training updates forbidden in evaluation')
        if self.mode=='ema' and self.pending:self.vocabulary.update_codebook(torch.cat(self.pending))
        self.pending.clear()
