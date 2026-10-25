"""Audited local inputs, separating RGB/cell/prompt tensors from target labels."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .audit import digest
from .protocol import nested
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


class LocalData:
    def __init__(self,tasks,realized=None):
        self.tasks=Path(tasks);self.contract=json.loads((self.tasks/'contract.json').read_text())
        self.catalog=Path(self.contract['catalog'])
        if digest(self.catalog/'tiles.parquet')!=self.contract['catalog_sha256']:raise ValueError('catalog changed')
        self.rows={r['tile_id']:r for r in pd.read_parquet(self.catalog/'tiles.parquet').to_dict('records')}
        if digest(self.tasks/'validation_maximum.json')!=self.contract['validation_sha256']:raise ValueError('validation changed')
        self.validation=[nested(p,'3+2') for p in json.loads((self.tasks/'validation_maximum.json').read_text()) if p['prompt_group']==0]
        self.training=None;self.realized_sha=None
        if realized is not None:
            realized=Path(realized);summary=json.loads((realized/'summary.json').read_text())
            if summary['status']!='complete' or summary['protocol_sha256']!=digest(self.tasks/'contract.json'):raise ValueError('realized protocol changed')
            self.realized_sha=digest(realized/'training_plans.jsonl')
            if self.realized_sha!=summary['training_plans_sha256']:raise ValueError('realized coordinates changed')
            self.training=[json.loads(line) for line in (realized/'training_plans.jsonl').read_text().splitlines()]

    @lru_cache(maxsize=32)
    def raw(self,identity,contexts=False,cells=False):
        row=self.rows[identity];variant=json.loads(row['variants_json'])['original']
        if digest(variant['image'])!=variant['image_sha256']:raise ValueError('RGB changed')
        image=np.asarray(Image.open(variant['image']).convert('RGB')).copy();gt=np.asarray(Image.open(row['labels'])).copy()
        if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('GT changed')
        result=dict(rgb=torch.from_numpy(image).permute(2,0,1)[None],image=he_to_tensor(image)[None])
        if cells:
            if digest(variant['cells'])!=variant['cells_sha256']:raise ValueError('cell cache changed')
            with np.load(variant['cells'],allow_pickle=False) as cache:
                features=cache['features'].copy();total=int(cache['total'])
            result.update(cells=torch.from_numpy(features)[None],cell_valid=torch.ones(1,len(features),dtype=torch.bool),
                          total_cell_count=torch.tensor([total],dtype=torch.float32))
        if contexts:
            if digest(row['multiscale_cache'])!=row['multiscale_sha256']:raise ValueError('context cache changed')
            with np.load(row['multiscale_cache'],allow_pickle=False) as cache:
                for scale in (1,2):
                    context=cache[f'image_{scale}'].copy()
                    result[f'rgb_{scale}']=torch.from_numpy(context).permute(2,0,1)[None]
                    result[f'image_{scale}']=he_to_tensor(context)[None]
                    if cells:
                        features=cache[f'cells_{scale}'].copy()
                        result.update({f'cells_{scale}':torch.from_numpy(features)[None],
                            f'cell_valid_{scale}':torch.ones(1,len(features),dtype=torch.bool),
                            f'total_{scale}':torch.tensor([int(cache[f'total_{scale}'])],dtype=torch.float32)})
        return result,gt

    def materialize(self,plan,device,contexts=False,cells=False):
        if plan['support_id']!=plan['query_id'] or plan['kind']!='local':raise ValueError('nonlocal training task')
        row=self.rows[plan['query_id']]
        if row['split']!=plan['split']:raise ValueError('split mismatch')
        raw,gt=self.raw(plan['query_id'],contexts,cells);moved={k:v.to(device) for k,v in raw.items()}
        prompts={}
        for sign in ('positive','negative'):
            xy=np.asarray(plan[sign+'_xy'],float);pixel=np.floor(xy*np.array(gt.shape[::-1])).astype(int)
            if not ((xy>0)&(xy<1)).all():raise ValueError('point bounds')
            code=gt[pixel[:,1],pixel[:,0]]
            if not (code==plan['target'] if sign=='positive' else (code!=plan['target'])&(code!=255)).all():raise ValueError('prompt sign')
            prompts[sign+'_xy']=torch.tensor([plan[sign+'_xy']],device=device,dtype=torch.float32)
            prompts[sign+'_mask']=torch.ones(1,len(xy),device=device,dtype=torch.bool)
        prompts['prompt_size_id']=torch.ones(1,device=device,dtype=torch.long)
        truth=torch.from_numpy(gt==plan['target'])[None].to(device);valid=torch.from_numpy(gt!=255)[None].to(device)
        supervision=dict(truth=truth,valid=valid,mpp=json.loads(row['input_mpp_xy_json']),labels=torch.from_numpy(gt.astype(np.int64))[None].to(device))
        # No class ID, GT or domain labels enter the interactive inputs dictionary.
        return moved,prompts,supervision


def fewclick_inputs(raw):
    result={k:v for k,v in raw.items() if not k.startswith('rgb')}
    if 'image' not in result:raise ValueError('missing established CPU-normalized image')
    return result
