#!/usr/bin/env python3
"""Cache all detected 32 px nuclei and one 512 px H&E token per selected patch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import soft_attention
from benchmarks.v5.tools.run_crc_panel1_cell_identity import PREPROCESS, SOURCE, WEIGHTS, load_ctranspath


BASE = Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
SMALL = BASE / 'panel1_small_cell_crop_full_1008_143551'
MAIN = BASE / 'panel1_train_all_0923_170500'
SUP = BASE / 'panel1_eval_supplement_0923_155834'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def crop(array, cx, cy, size=32):
    half = size // 2
    h, w = array.shape[:2]
    top, bottom = max(0, cy-half), min(h, cy+half)
    left, right = max(0, cx-half), min(w, cx+half)
    padding = ((max(0, half-cy), max(0, cy+half-h)),
               (max(0, half-cx), max(0, cx+half-w)))
    if array.ndim == 3:
        padding += ((0, 0),)
    return np.pad(array[top:bottom, left:right], padding, mode='edge')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--max-patches', type=int)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--cpu-threads', type=int, default=4)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(BASE) or not 0 <= a.shard_index < a.shard_count:
        p.error('invalid output or shard')
    if a.batch_size < 1 or a.cpu_threads < 1 or (a.max_patches is not None and a.max_patches < 1):
        p.error('invalid budget')
    a.output.mkdir(parents=True, exist_ok=False)
    cells = pd.read_csv(SMALL / 'cells.csv')
    main_meta = json.loads((MAIN / 'manifest.json').read_text())
    sup_meta = json.loads((SUP / 'audit.json').read_text())
    if digest(MAIN/'cells.csv') != main_meta['manifest_sha256'] or digest(SUP/'cells.csv') != sup_meta['manifest_sha256']:
        raise ValueError('source manifest hash mismatch')
    refs = {r['patch']: r for r in main_meta['patches']}
    supplement = pd.read_csv(SUP/'cells.csv')
    for patch, frame in supplement.groupby('patch'):
        if frame[['he_sha256','mask_sha256']].nunique().max() != 1 or patch in refs:
            raise ValueError(f'supplement patch identity mismatch: {patch}')
        refs[patch] = frame.iloc[0][['he_sha256','mask_sha256']].to_dict()
    patches = sorted(cells.patch.unique())
    patches = [patch for patch in patches if int(hashlib.sha256(patch.encode()).hexdigest(),16) % a.shard_count == a.shard_index]
    if a.max_patches:
        patches = patches[:a.max_patches]
    if not patches:
        raise ValueError('no patches')
    torch.set_num_threads(a.cpu_threads)
    model = load_ctranspath(WEIGHTS, torch.device(a.device)).requires_grad_(False)
    expected = cells.groupby('patch')
    records = []
    for ordinal, patch in enumerate(patches, 1):
        ref = refs[patch]
        image_path = SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path = SOURCE/'cell_matching/masks'/f'{patch}.npy'
        if digest(image_path) != ref['he_sha256'] or digest(mask_path) != ref['mask_sha256']:
            raise ValueError(f'raw source checksum mismatch: {patch}')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.load(mask_path, allow_pickle=False)
        if image.shape != (512,512,3) or mask.shape != (512,512) or mask.dtype.kind not in 'iu':
            raise ValueError(f'invalid patch geometry: {patch}')
        ids = np.unique(mask)
        ids = ids[ids != 0]
        centers = np.asarray(ndimage.center_of_mass(np.ones(mask.shape,dtype=np.uint8),mask,ids),dtype=np.float32)[:,::-1]
        if not np.isfinite(centers).all():
            raise ValueError(f'bad centroid: {patch}')
        group = expected.get_group(patch)
        loc = {int(v): i for i,v in enumerate(ids)}
        for row in group.itertuples():
            if int(row.he_id) not in loc or np.linalg.norm(centers[loc[int(row.he_id)]]-np.array([row.he_x,row.he_y])) > 0.1:
                raise ValueError(f'target mismatch: {patch}/{row.he_id}')
        tensors = []
        for label,(x,y) in zip(ids,centers):
            cx,cy = round(float(x)),round(float(y))
            region = crop(image,cx,cy)
            nucleus = crop(mask,cx,cy) == label
            tensors.append(PREPROCESS(soft_attention(region,nucleus)))
        with torch.inference_mode():
            parts = [model(torch.stack(tensors[i:i+a.batch_size]).to(a.device)).float().cpu().numpy()
                     for i in range(0,len(tensors),a.batch_size)]
            features = np.concatenate(parts)
            patch512 = model(PREPROCESS(image).unsqueeze(0).to(a.device)).float().cpu().numpy()[0]
        if features.shape != (len(ids),768) or not np.isfinite(features).all() or not np.isfinite(patch512).all():
            raise ValueError(f'bad features: {patch}')
        destination = a.output/f'{patch}.npz'
        np.savez_compressed(destination, he_ids=ids.astype(np.int32), he_xy=centers,
                            features32=features.astype(np.float16), patch512=patch512.astype(np.float16),
                            he_sha256=ref['he_sha256'], mask_sha256=ref['mask_sha256'])
        rec = {'patch':patch,'cells':len(ids),'targets':len(group),'sha256':digest(destination)}
        records.append(rec)
        with (a.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(rec)+'\n')
        print(json.dumps({'event':'patch','ordinal':ordinal,'total':len(patches),**rec}),flush=True)
    audit = {'status':'complete','shard_index':a.shard_index,'shard_count':a.shard_count,
             'patches':len(records),'cells':sum(r['cells'] for r in records),'records':records,
             'source_small_cells_sha256':digest(SMALL/'cells.csv'),'checkpoint_sha256':digest(WEIGHTS),
             'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'event':'complete','patches':len(records),'cells':audit['cells']}),flush=True)


if __name__ == '__main__':
    main()
