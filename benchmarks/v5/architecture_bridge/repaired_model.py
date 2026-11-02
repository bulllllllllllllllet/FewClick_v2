"""Explicit accepted repair cache; no fallback to historical cell inputs."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import BridgeData
from .deterministic_model import build_bridge_image
from .cell_vocabulary import CellVocabulary
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


class RepairedData(BridgeData):
    def __init__(self, cache):
        self.cache=Path(cache);proof=json.loads((self.cache/'acceptance.json').read_text())
        assert proof['status']=='accepted_repaired_normalized_cells' and proof['validation_fitting_cells']==0
        assert sha(self.cache/'index.jsonl')==proof['index_sha256']
        assert sha(self.cache/'transforms.npz')==proof['transforms_sha256']
        _,self.rows=load_cell_catalog(TASKS)
        self.entries={r['tile_id']:r for r in map(json.loads,(self.cache/'index.jsonl').read_text().splitlines())}
        assert self.entries.keys()==self.rows.keys()
        self.identity=dict(repaired_index_sha256=proof['index_sha256'],transforms_sha256=proof['transforms_sha256'])

    @lru_cache(maxsize=24)
    def inputs(self,identity,cell_arm):
        assert cell_arm=='xcell';row=self.entries[identity]
        assert sha(row['file'])==row['sha256'] and sha(row['normalized_file'])==row['normalized_sha256']
        result={}
        with np.load(row['file']) as raw,np.load(row['normalized_file']) as norm:
            for scale in range(3):
                suffix='' if scale==0 else '_'+str(scale)
                cells=norm[f'cells_{scale}'].copy()
                result['image'+suffix]=he_to_tensor(raw[f'image_{scale}'])[None]
                result['cells'+suffix]=torch.from_numpy(cells)[None]
                result['cell_valid'+suffix]=torch.ones(1,len(cells),dtype=torch.bool)
                result['total_cell_count' if scale==0 else 'total_'+str(scale)]=torch.tensor([int(raw[f'total_{scale}'])],dtype=torch.float32)
        return result


class RepairedContinuous(nn.Module):
    def __init__(self):
        super().__init__();self.core,self.identity=build_bridge_image('phikon','cuda:0',20261103)
        self.representation=CellVocabulary('continuous',torch.zeros(128,64)).cuda()

    def train(self,mode=True):
        super().train(mode);self.core.eval();return self

    def transform(self,inputs):
        result=dict(inputs)
        for s in ['', '_1', '_2']:
            x=inputs['cells'+s];z,_=self.representation(x[...,10:].reshape(-1,64))
            result['cells'+s]=torch.cat([x[...,:10],z.reshape(*x.shape[:2],64)],-1)
        return result

    def forward(self,support,query,prompts,local,shared,with_support=True):
        support=self.transform(support);query=support if shared else self.transform(query)
        return self.core(support,query,prompts,local,shared,with_support)
