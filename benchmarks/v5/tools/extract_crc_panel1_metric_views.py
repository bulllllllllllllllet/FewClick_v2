#!/usr/bin/env python3
"""Extract raw 32px and physical ~56um (112px) target-cell CTransPath features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from benchmarks.v5.tools.run_crc_panel1_cell_identity import PREPROCESS, SOURCE, WEIGHTS, load_ctranspath

BASE = Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
SMALL = BASE/'panel1_small_cell_crop_full_1008_143551'
MAIN = BASE/'panel1_train_all_0923_170500'
SUP = BASE/'panel1_eval_supplement_0923_155834'
SIZES = (32,112)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--max-patches',type=int)
    p.add_argument('--only-patch')
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(BASE) or min(a.batch_size,a.cpu_threads)<1:
        p.error('invalid output or budget')
    if a.max_patches is not None and a.max_patches<1:
        p.error('invalid patch limit')
    a.output.mkdir(parents=True,exist_ok=False)
    source=SMALL/'cells.csv'
    cells=pd.read_csv(source)
    if cells[['patch','he_id']].duplicated().any() or cells.groupby('patch').arm_split.nunique().max()!=1:
        raise ValueError('duplicate cell or cross-split patch')
    patches=sorted(cells.patch.unique())
    if a.only_patch:
        if a.only_patch not in patches: p.error('unknown patch')
        patches=[a.only_patch]
    if a.max_patches: patches=patches[:a.max_patches]
    selected=cells[cells.patch.isin(patches)].copy().reset_index(drop=True)
    selected['feature_row']=np.arange(len(selected))
    main_meta=json.loads((MAIN/'manifest.json').read_text())
    sup_meta=json.loads((SUP/'audit.json').read_text())
    if digest(MAIN/'cells.csv')!=main_meta['manifest_sha256'] or digest(SUP/'cells.csv')!=sup_meta['manifest_sha256']:
        raise ValueError('upstream manifest changed')
    refs={rec['patch']:rec for rec in main_meta['patches']}
    supplement=pd.read_csv(SUP/'cells.csv')
    for patch,group in supplement.groupby('patch'):
        if patch in refs or group[['he_sha256','mask_sha256']].nunique().max()!=1:
            raise ValueError(f'supplement identity mismatch: {patch}')
        refs[patch]=group.iloc[0][['he_sha256','mask_sha256']].to_dict()
    features={size:np.empty((len(selected),768),np.float32) for size in SIZES}
    torch.set_num_threads(a.cpu_threads)
    model=load_ctranspath(WEIGHTS,torch.device(a.device)).requires_grad_(False)
    for ordinal,(patch,group) in enumerate(selected.groupby('patch',sort=False),1):
        image_path=SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path=SOURCE/'cell_matching/masks'/f'{patch}.npy'
        ref=refs[patch]
        if digest(image_path)!=ref['he_sha256'] or digest(mask_path)!=ref['mask_sha256']:
            raise ValueError(f'source changed: {patch}')
        image=np.asarray(Image.open(image_path).convert('RGB'))
        mask=np.load(mask_path,allow_pickle=False)
        if image.shape!=(512,512,3) or mask.shape!=(512,512):
            raise ValueError(f'invalid geometry: {patch}')
        rows=group.feature_row.to_numpy(int)
        tensors={size:[] for size in SIZES}
        for row in group.itertuples():
            cx,cy=round(float(row.he_x)),round(float(row.he_y))
            if not 56<=min(cx,cy) or max(cx,cy)>456 or int(mask[cy,cx])!=int(row.he_id):
                raise ValueError(f'112px crop or target center invalid: {patch}/{row.he_id}')
            for size in SIZES:
                half=size//2
                crop=image[cy-half:cy+half,cx-half:cx+half]
                if crop.shape!=(size,size,3):
                    raise ValueError(f'crop leaves source: {patch}/{row.he_id}')
                tensors[size].append(PREPROCESS(crop))
        with torch.inference_mode():
            for size in SIZES:
                chunks=[model(torch.stack(tensors[size][i:i+a.batch_size]).to(a.device)).float().cpu().numpy()
                        for i in range(0,len(rows),a.batch_size)]
                features[size][rows]=np.concatenate(chunks)
        if ordinal%50==0:
            print(json.dumps({'event':'extract','patches':ordinal,'total':len(patches),'cells':int(rows[-1])+1}),flush=True)
    if any(not np.isfinite(value).all() for value in features.values()):
        raise ValueError('nonfinite features')
    selected.to_csv(a.output/'cells.csv',index=False)
    np.savez_compressed(a.output/'features.npz',**{f'features_{size}_raw':v for size,v in features.items()})
    audit={'status':'complete','rows':len(selected),'patches':len(patches),'source_cells_sha256':digest(source),
           'checkpoint_sha256':digest(WEIGHTS),'mpp':0.49807829793963543,'physical_view_um':112*0.49807829793963543,
           'features_sha256':digest(a.output/'features.npz'),
           'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'complete','rows':len(selected),'patches':len(patches),'output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
