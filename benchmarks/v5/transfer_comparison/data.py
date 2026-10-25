"""Image-only loading for remote inference; query labels have a separate scoring API."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.fair_comparison.audit import digest


class TransferData:
    def __init__(self,tasks):
        self.tasks=Path(tasks);self.contract=json.loads((self.tasks/'contract.json').read_text())
        self.catalog=Path(self.contract['catalog'])
        if digest(self.catalog/'tiles.parquet')!=self.contract['catalog_sha256']:raise ValueError('catalog changed')
        if digest(self.tasks/'validation_maximum.json')!=self.contract['tasks_sha256']:raise ValueError('transfer tasks changed')
        self.rows={r['tile_id']:r for r in pd.read_parquet(self.catalog/'tiles.parquet').to_dict('records')}
        self.plans=json.loads((self.tasks/'validation_maximum.json').read_text())

    @lru_cache(maxsize=32)
    def inputs(self,identity,contexts,cells):
        # Deliberately never inspect labels, counts, target class or validity GT.
        row=self.rows[identity];variant=json.loads(row['variants_json'])['original']
        if digest(variant['image'])!=variant['image_sha256']:raise ValueError('RGB identity changed')
        rgb=np.asarray(Image.open(variant['image']).convert('RGB')).copy()
        result=dict(rgb=torch.from_numpy(rgb).permute(2,0,1)[None],image=he_to_tensor(rgb)[None])
        if cells:
            if digest(variant['cells'])!=variant['cells_sha256']:raise ValueError('fine cell identity changed')
            with np.load(variant['cells'],allow_pickle=False) as cache:
                features=cache['features'].copy();total=int(cache['total'])
            result.update(cells=torch.from_numpy(features)[None],cell_valid=torch.ones(1,len(features),dtype=torch.bool),
                          total_cell_count=torch.tensor([total],dtype=torch.float32))
        if contexts:
            if digest(row['multiscale_cache'])!=row['multiscale_sha256']:raise ValueError('context identity changed')
            with np.load(row['multiscale_cache'],allow_pickle=False) as cache:
                for scale in (1,2):
                    rgb=cache[f'image_{scale}'].copy()
                    result[f'rgb_{scale}']=torch.from_numpy(rgb).permute(2,0,1)[None];result[f'image_{scale}']=he_to_tensor(rgb)[None]
                    if cells:
                        features=cache[f'cells_{scale}'].copy()
                        result.update({f'cells_{scale}':torch.from_numpy(features)[None],f'cell_valid_{scale}':torch.ones(1,len(features),dtype=torch.bool),
                                       f'total_{scale}':torch.tensor([int(cache[f'total_{scale}'])],dtype=torch.float32)})
        return result

    def materialize_inputs(self,plan,device,contexts=True,cells=True):
        s=self.rows[plan['support_id']];q=self.rows[plan['query_id']]
        if s['split']!='val' or q['split']!='val' or s['tile_id']==q['tile_id']:raise ValueError('remote identity/split mismatch')
        def move(identity):return {k:v.to(device) for k,v in self.inputs(identity,contexts,cells).items()}
        prompts={}
        for sign in ('positive','negative'):
            xy=np.asarray(plan[sign+'_xy'],float)
            if xy.ndim!=2 or xy.shape[1]!=2 or not ((xy>0)&(xy<1)).all():raise ValueError('invalid support prompts')
            prompts[sign+'_xy']=torch.tensor([plan[sign+'_xy']],device=device,dtype=torch.float32)
            prompts[sign+'_mask']=torch.ones(1,len(xy),device=device,dtype=torch.bool)
        prompts['prompt_size_id']=torch.ones(1,device=device,dtype=torch.long)
        return dict(support=move(s['tile_id']),query=move(q['tile_id']),prompts=prompts,local=False,shared=False)

    @lru_cache(maxsize=64)
    def labels(self,identity):
        row=self.rows[identity];gt=np.asarray(Image.open(row['labels'])).copy()
        if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('label identity changed')
        return gt

    def supervision(self,plan):
        # This API belongs to the benchmark scorer, not any inference method.
        query=self.labels(plan['query_id']);target=plan['target']
        return dict(truth=query==target,valid=query!=255,mpp=json.loads(self.rows[plan['query_id']]['input_mpp_xy_json']))
