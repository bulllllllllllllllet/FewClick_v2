#!/usr/bin/env python3
"""Extract official frozen CytoFormer embeddings for the fixed Panel1 56um view."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from benchmarks.v5.tools.extract_crc_panel1_metric_views import BASE, MAIN, SMALL, SOURCE, SUP, digest


def verify_source(path):
    audit=json.loads((path/'SOURCE_AUDIT.json').read_text())
    for rec in audit['records']:
        if digest(path/rec['path'])!=rec['sha256']:
            raise ValueError(f"official source changed: {rec['path']}")
    if not (path/'cytoformer/model.py').exists():
        raise ValueError('CytoFormer code missing')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-code',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--only-patch')
    a=p.parse_args()
    if not all(x.resolve().is_relative_to(BASE) for x in (a.source_code,a.checkpoint_dir,a.output)):
        p.error('all external inputs/outputs must be under FewClick artifact root')
    if min(a.batch_size,a.cpu_threads)<1:
        p.error('invalid budget')
    source_audit=verify_source(a.source_code)
    weight_audit=json.loads((a.checkpoint_dir/'audit.json').read_text())
    checkpoint=a.checkpoint_dir/'checkpoint.pth'
    if weight_audit['status']!='complete' or digest(checkpoint)!=weight_audit['sha256']:
        raise ValueError('checkpoint integrity failure')
    a.output.mkdir(parents=True,exist_ok=False)
    cells=pd.read_csv(SMALL/'cells.csv')
    if a.only_patch:
        cells=cells.loc[cells.patch.eq(a.only_patch)].copy()
        if cells.empty:p.error('patch absent in target table')
    cells=cells.reset_index(drop=True)
    if cells[['patch','he_id']].duplicated().any() or cells.groupby('patch').arm_split.nunique().max()!=1:
        raise ValueError('invalid cell table')
    main_meta=json.loads((MAIN/'manifest.json').read_text())
    sup_meta=json.loads((SUP/'audit.json').read_text())
    if digest(MAIN/'cells.csv')!=main_meta['manifest_sha256'] or digest(SUP/'cells.csv')!=sup_meta['manifest_sha256']:
        raise ValueError('upstream manifest changed')
    refs={rec['patch']:rec for rec in main_meta['patches']}
    supplement=pd.read_csv(SUP/'cells.csv')
    for patch,group in supplement.groupby('patch'):
        if patch in refs or group[['he_sha256','mask_sha256']].nunique().max()!=1:
            raise ValueError('supplement identity failure')
        refs[patch]=group.iloc[0][['he_sha256','mask_sha256']].to_dict()
    sys.path.insert(0,str(a.source_code))
    os.environ['CYTOFORMER_ORGAN_MAP']=str(a.source_code/'cytoformer/organ_celltype_map.json')
    from cytoformer.model import CellClassifier
    torch.set_num_threads(a.cpu_threads)
    device=torch.device(a.device)
    model=CellClassifier()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    del state
    model=model.to(device).eval().requires_grad_(False)
    features=np.full((len(cells),1536),np.nan,np.float32)
    pending=[];pending_rows=[]
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)

    def flush():
        if not pending:return
        array=np.stack(pending)
        x=torch.from_numpy(array).to(device).permute(0,3,1,2).float().div_(255)
        x=(x-mean)/std
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            value=model.extract_features(x).float().cpu().numpy()
        if value.shape!=(len(pending),1536) or not np.isfinite(value).all():
            raise ValueError('invalid CytoFormer feature output')
        features[pending_rows]=value
        pending.clear();pending_rows.clear()

    for ordinal,(patch,group) in enumerate(cells.groupby('patch',sort=False),1):
        image_path=SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path=SOURCE/'cell_matching/masks'/f'{patch}.npy'
        ref=refs[patch]
        if digest(image_path)!=ref['he_sha256'] or digest(mask_path)!=ref['mask_sha256']:
            raise ValueError(f'source checksum mismatch: {patch}')
        image=np.asarray(Image.open(image_path).convert('RGB'))
        mask=np.load(mask_path,allow_pickle=False)
        if image.shape!=(512,512,3) or mask.shape!=(512,512):
            raise ValueError(f'invalid patch geometry: {patch}')
        for row in group.itertuples():
            cx,cy=round(float(row.he_x)),round(float(row.he_y))
            if cx<56 or cy<56 or cx>456 or cy>456 or int(mask[cy,cx])!=int(row.he_id):
                raise ValueError(f'bad target/crop: {patch}/{row.he_id}')
            crop=image[cy-56:cy+56,cx-56:cx+56]
            pending.append(np.asarray(Image.fromarray(crop).resize((224,224))))
            pending_rows.append(row.Index)
            if len(pending)>=a.batch_size:flush()
        if ordinal%50==0:
            print(json.dumps({'event':'extract','patches':ordinal,'total_patches':cells.patch.nunique(),
                              'completed_cells':int(np.isfinite(features[:,0]).sum())}),flush=True)
    flush()
    if not np.isfinite(features).all():
        raise ValueError('missing CytoFormer feature rows')
    np.savez_compressed(a.output/'features.npz',features=features,
                        source_cells_sha256=digest(SMALL/'cells.csv'))
    audit={'status':'complete','rows':len(cells),'patches':cells.patch.nunique(),
           'source_cells_sha256':digest(SMALL/'cells.csv'),'checkpoint_sha256':weight_audit['sha256'],
           'official_source_revision':source_audit['revision'],'features_sha256':digest(a.output/'features.npz'),
           'physical_view_um':112*0.49807829793963543,
           'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'complete','rows':len(cells),'output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
