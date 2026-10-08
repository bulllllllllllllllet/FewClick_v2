#!/usr/bin/env python3
"""Controlled 16/32/64/128 px Panel1 target-cell crop probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import soft_attention
from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, PREPROCESS, SOURCE, WEIGHTS, load_ctranspath, score


BASE=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
MAIN=BASE/'panel1_train_all_0923_170500'
MAIN_FEATURE=BASE/'panel1_train_all_features_0923_172500'
SUP=BASE/'panel1_eval_supplement_0923_155834'
SUP_FEATURE=BASE/'panel1_eval_supplement_scores_0923_155610'
SIZES=(16,32,64)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def cut(image, cx, cy, size):
    half=size//2
    return image[cy-half:cy+half,cx-half:cx+half]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--train-per-class',type=int,default=300)
    p.add_argument('--smoke-per-class',type=int,default=0)
    p.add_argument('--seed',type=int,default=20261008)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(BASE):
        p.error('output must be under FewClick artifact root')
    if a.train_per_class<2 or a.smoke_per_class<0:
        p.error('invalid sampling')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    main=pd.read_csv(MAIN/'cells.csv')
    sup=pd.read_csv(SUP/'cells.csv')
    main_meta=json.loads((MAIN/'manifest.json').read_text())
    sup_meta=json.loads((SUP/'audit.json').read_text())
    if digest(MAIN/'cells.csv')!=main_meta['manifest_sha256'] or digest(SUP/'cells.csv')!=sup_meta['manifest_sha256']:
        raise ValueError('source manifest hash mismatch')
    if set(main.patch)&set(sup.patch):
        raise ValueError('supplement reuses main patch')
    if set(main.loc[main.split.eq('train'),'group'])&(set(main.loc[main.split.ne('train'),'group'])|set(sup.group)):
        raise ValueError('spatial group overlap')
    main['origin']='main';main['source_index']=np.arange(len(main))
    sup['origin']='supplement';sup['source_index']=np.arange(len(sup))
    rng=np.random.default_rng(a.seed)
    frames=[]
    for split in ('train','val','test'):
        part=main.loc[main.split.eq(split)]
        for klass in range(5):
            group=part.loc[part.label.eq(klass)]
            cap=(a.train_per_class if split=='train' else len(group))
            if a.smoke_per_class:
                cap=min(cap,a.smoke_per_class)
            frames.append(group.iloc[np.sort(rng.choice(len(group),size=min(cap,len(group)),replace=False))])
    for split in ('val','test'):
        part=sup.loc[sup.split.eq(split)]
        for klass in range(5):
            group=part.loc[part.label.eq(klass)]
            cap=min(len(group),a.smoke_per_class) if a.smoke_per_class else len(group)
            frames.append(group.iloc[np.sort(rng.choice(len(group),size=cap,replace=False))])
    selected=pd.concat(frames,ignore_index=True)
    selected['arm_split']=np.where(selected.origin.eq('main'),selected.split,'supplement_'+selected.split)
    if any(selected.loc[selected.arm_split.eq('train'),'label'].value_counts().reindex(range(5),fill_value=0)<2):
        raise ValueError('train class absent')
    patch_meta={x['patch']:x for x in main_meta['patches']}
    for patch,frame in sup.groupby('patch'):
        if frame[['he_sha256','mask_sha256']].nunique().max()!=1:
            raise ValueError('supplement patch hash inconsistent')
        patch_meta[patch]=frame.iloc[0][['he_sha256','mask_sha256']].to_dict()
    with np.load(MAIN_FEATURE/'features.npz',allow_pickle=False) as z:
        main128=z['features128'].copy()
    with np.load(SUP_FEATURE/'features.npz',allow_pickle=False) as z:
        sup128=z['features128'].copy()
    if main128.shape!=(len(main),768) or sup128.shape!=(len(sup),768):
        raise ValueError('128 baseline cache shape mismatch')
    features={128:np.empty((len(selected),768),np.float32)}
    main_rows=selected.origin.eq('main').to_numpy()
    features[128][main_rows]=main128[selected.loc[main_rows,'source_index'].to_numpy(int)]
    features[128][~main_rows]=sup128[selected.loc[~main_rows,'source_index'].to_numpy(int)]
    for size in SIZES:
        features[size]=np.full((len(selected),768),np.nan,np.float32)
        selected[f'nucleus_fraction_{size}']=np.nan
    model=load_ctranspath(WEIGHTS,torch.device(a.device)).requires_grad_(False)
    selected['cell_row']=np.arange(len(selected))
    for ordinal,(patch,frame) in enumerate(selected.groupby('patch',sort=False),1):
        image_path=SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path=SOURCE/'cell_matching/masks'/f'{patch}.npy'
        ref=patch_meta[patch]
        if digest(image_path)!=ref['he_sha256'] or digest(mask_path)!=ref['mask_sha256']:
            raise ValueError(f'raw patch hash mismatch: {patch}')
        image=np.asarray(Image.open(image_path).convert('RGB'))
        mask=np.load(mask_path,allow_pickle=False)
        if image.shape!=(512,512,3) or mask.shape!=(512,512):
            raise ValueError(f'invalid source dimensions: {patch}')
        tensors={size:[] for size in SIZES}
        cells=[]
        for row in frame.itertuples():
            cx,cy=round(float(row.he_x)),round(float(row.he_y))
            nucleus=mask==int(row.he_id)
            full=int(nucleus.sum())
            if full==0 or not nucleus[cy,cx]:
                raise ValueError(f'target nucleus absent at center: {patch}/{row.he_id}')
            for size in SIZES:
                crop=cut(image,cx,cy,size)
                target=cut(nucleus,cx,cy,size)
                if crop.shape!=(size,size,3) or target.shape!=(size,size):
                    raise ValueError('crop leaves source patch')
                selected.at[row.cell_row,f'nucleus_fraction_{size}']=float(target.sum()/full)
                tensors[size].append(PREPROCESS(soft_attention(crop,target)))
            cells.append(row.cell_row)
        with torch.inference_mode():
            for size in SIZES:
                chunks=[model(batch.to(a.device)).float().cpu().numpy()
                        for batch in torch.stack(tensors[size]).split(32)]
                features[size][cells]=np.concatenate(chunks)
        if ordinal%50==0:
            print(json.dumps({'event':'extract','patches':ordinal,'cells':int(np.isfinite(features[16][:,0]).sum())}),flush=True)
    if any(not np.isfinite(value).all() for value in features.values()) or selected[[f'nucleus_fraction_{x}' for x in SIZES]].isna().any().any():
        raise ValueError('incomplete or nonfinite result')
    labels=selected.label.to_numpy(int)
    split=selected.arm_split.to_numpy(str)
    train=np.flatnonzero(split=='train')
    report={'status':'complete','classes':CLASSES,'samples':selected.groupby(['arm_split','label']).size().to_dict(),
            'manifest_sha256':main_meta['manifest_sha256'],'supplement_manifest_sha256':sup_meta['manifest_sha256'],
            'training':'L2 + train-only StandardScaler + class_weight=balanced LogisticRegression(C=1)',
            'nucleus_coverage':{},'arms':{}}
    report['samples']={f'{s}:{k}':int(v) for (s,k),v in report['samples'].items()}
    for size in SIZES:
        v=selected[f'nucleus_fraction_{size}'].to_numpy(float)
        report['nucleus_coverage'][str(size)]={'quantiles':np.quantile(v,[0,.1,.25,.5,.75,.9,1]).round(4).tolist(),
            'below_90_percent':int((v<.9).sum()),'total':len(v)}
    predictions=selected[['origin','source_index','patch','he_id','mif_id','arm_split','label']].copy()
    for size,value in features.items():
        x=normalize(value)
        scaler=StandardScaler().fit(x[train])
        x=scaler.transform(x)
        clf=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=a.seed).fit(x[train],labels[train])
        y=clf.predict(x)
        predictions[f'pred_{size}']=y
        report['arms'][str(size)]={s:score(labels[split==s],y[split==s])
                                   for s in ('val','test','supplement_val','supplement_test')}
        print(json.dumps({'size':size,'val_macro_f1':report['arms'][str(size)]['val']['macro_f1'],
            'supplement_test_macro_f1':report['arms'][str(size)]['supplement_test']['macro_f1']}),flush=True)
    selected.to_csv(a.output/'cells.csv',index=False)
    predictions.to_csv(a.output/'predictions.csv',index=False)
    np.savez_compressed(a.output/'features.npz',**{f'features_{size}':value for size,value in features.items()})
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output/'run.json').write_text(json.dumps({'argv':vars(a),'checkpoint_sha256':digest(WEIGHTS)},default=str,indent=2)+'\n')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
