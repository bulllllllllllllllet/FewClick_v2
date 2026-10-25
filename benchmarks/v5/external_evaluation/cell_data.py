"""Assemble RGB and real cell evidence without opening tissue annotations."""
import json

import pyvips
import numpy as np
import torch

from .prepare import digest
from .evaluate_public import inputs
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def assemble(row, device='cpu'):
    if row.get('cell_cache_status')!='complete':
        raise ValueError('real cell cache is required; no zero-feature substitute')
    if digest(row['cell_cache'])!=row['cell_cache_sha256']:
        raise ValueError('external cell cache changed')
    data=inputs(row,True)
    images=json.loads(row['images_json'])
    from PIL import Image
    entry=images[2]
    if digest(entry['path'])!=entry['sha256']:
        raise ValueError('coarse RGB changed')
    data['coarse']=np.asarray(Image.open(entry['path']).convert('RGB')).copy()
    result={}
    with np.load(row['cell_cache'],allow_pickle=False) as cache:
        for scale,key in enumerate(('fine','low','coarse')):
            image=data[key]
            if image.shape!=(512,512,3):
                raise ValueError('cell input image shape')
            features=cache[f'cells_{scale}'].copy()
            total=int(cache[f'total_{scale}'])
            if features.ndim!=2 or features.shape[1]!=74 or len(features)>255 or total<len(features) or not np.isfinite(features).all():
                raise ValueError('invalid cell features/count')
            suffix='' if scale==0 else f'_{scale}'
            result['rgb'+suffix]=torch.from_numpy(image).permute(2,0,1)[None]
            result['image'+suffix]=he_to_tensor(image)[None]
            result['cells'+suffix]=torch.from_numpy(features)[None]
            result['cell_valid'+suffix]=torch.ones(1,len(features),dtype=torch.bool)
            result['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([total],dtype=torch.float32)
    return {key:value.to(device) for key,value in result.items()}
