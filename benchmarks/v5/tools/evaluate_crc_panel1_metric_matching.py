#!/usr/bin/env python3
"""Train-gallery retrieval, class prototypes, clustering and selected-cell composition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, normalize

BASE=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
SMALL=BASE/'panel1_small_cell_crop_full_1008_143551'
MAP3=np.array([0,1,1,2,2],np.int64)
SPLITS=('val','test','supplement_val','supplement_test')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def classification(truth,pred,classes):
    return {'accuracy':float(accuracy_score(truth,pred)),
            'macro_f1':float(f1_score(truth,pred,labels=list(range(classes)),average='macro',zero_division=0)),
            'confusion':confusion_matrix(truth,pred,labels=list(range(classes))).tolist(),
            'per_class_f1':f1_score(truth,pred,labels=list(range(classes)),average=None,zero_division=0).round(6).tolist()}


def selected_composition(rows,truth,pred,split,min_cells=10):
    """Diagnostic only: selected labelled cells are not a complete patch census."""
    groups=[]
    for patch,indices in rows.groupby('patch').indices.items():
        ids=np.asarray(indices)
        if split[ids[0]] not in SPLITS or len(ids)<min_cells:
            continue
        a=np.bincount(truth[ids],minlength=3)/len(ids)
        b=np.bincount(pred[ids],minlength=3)/len(ids)
        groups.append({'patch':patch,'split':split[ids[0]],'cells':len(ids),'true':a.tolist(),
                       'pred':b.tolist(),'total_variation':float(np.abs(a-b).sum()/2)})
    result={name:{'patches':sum(g['split']==name for g in groups),
                  'mean_total_variation':float(np.mean([g['total_variation'] for g in groups if g['split']==name]))
                  if any(g['split']==name for g in groups) else None} for name in SPLITS}
    return result,groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external-features',type=Path,help='optional validated NPZ with features and source hash')
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(BASE):
        p.error('output must be under FewClick artifact root')
    a.output.mkdir(parents=True,exist_ok=False)
    rows=pd.read_csv(SMALL/'cells.csv')
    views=pd.read_csv(a.views/'cells.csv')
    audit=json.loads((a.views/'audit.json').read_text())
    if audit['status']!='complete' or audit['rows']!=len(rows) or audit['source_cells_sha256']!=digest(SMALL/'cells.csv'):
        raise ValueError('view source or shape mismatch')
    if not rows[['patch','he_id']].equals(views[['patch','he_id']]):
        raise ValueError('feature row order changed')
    if rows.groupby('patch').arm_split.nunique().max()!=1:
        raise ValueError('patch crosses splits')
    if set(rows.loc[rows.arm_split.eq('train'),'group']) & set(rows.loc[rows.arm_split.ne('train'),'group']):
        raise ValueError('spatial group overlap')
    with np.load(SMALL/'features.npz',allow_pickle=False) as cache:
        features={'ctranspath_32_soft':cache['features_32'].copy(),
                  'ctranspath_128_soft':cache['features_128'].copy()}
    with np.load(a.views/'features.npz',allow_pickle=False) as cache:
        if digest(a.views/'features.npz')!=audit['features_sha256']:
            raise ValueError('view checksum mismatch')
        features.update({'ctranspath_32_raw':cache['features_32_raw'].copy(),
                         'ctranspath_112_raw':cache['features_112_raw'].copy()})
    if a.external_features:
        with np.load(a.external_features,allow_pickle=False) as cache:
            if str(cache['source_cells_sha256'])!=digest(SMALL/'cells.csv'):
                raise ValueError('external feature manifest mismatch')
            features['external_112_raw']=cache['features'].copy()
    labels5=rows.label.to_numpy(int)
    labels3=MAP3[labels5]
    split=rows.arm_split.to_numpy(str)
    train=np.flatnonzero(split=='train')
    split_ids={name:np.flatnonzero(split==name) for name in SPLITS}
    report={'status':'running','source_cells_sha256':digest(SMALL/'cells.csv'),
            'views_sha256':audit['features_sha256'],'rows':len(rows),'train_rows':len(train),
            'splits':{s:len(v) for s,v in split_ids.items()},'taxonomy3':['Epithelial','Endothelial','Stromal'],
            'taxonomy5':['Epithelial','Blood_endothelial','Lymphatic_endothelial','CAF','SMA_stromal'],
            'protocol':'train-only gallery/scaler/prototypes/PCA/KMeans; no classifier head; fixed k=5, prototypes=3',
            'views':{}}
    all_predictions=rows[['patch','he_id','arm_split','label']].copy()
    all_predictions['label3']=labels3
    compositions=[]
    for name,original in features.items():
        if original.shape[0]!=len(rows) or not np.isfinite(original).all():
            raise ValueError(f'invalid features: {name}')
        arm={'dimensions':original.shape[1],'normalizations':{}}
        for normalization in ('unit','train_zscore_unit'):
            if normalization=='unit':
                x=normalize(original.astype(np.float32))
            else:
                scaler=StandardScaler().fit(original[train])
                x=normalize(scaler.transform(original.astype(np.float32)))
            sim=x@x[train].T
            # All evaluation rows are outside train, so no self-match is possible.
            near=np.argsort(-sim,axis=1)[:,:15]
            results={}
            for taxonomy,truth,classes in (('coarse3',labels3,3),('fine5',labels5,5)):
                ordered=truth[train][near]
                p5=ordered[:,:5]
                # Ties resolve towards the class of the closest member, then class id.
                knn=np.array([sorted(range(classes),key=lambda c:(-int((v==c).sum()),
                            int(np.flatnonzero(v==c)[0]) if (v==c).any() else 99,c))[0] for v in p5])
                prototypes=[];classes_for_prototype=[]
                for klass in range(classes):
                    part=x[train[truth[train]==klass]]
                    if len(part)<3: raise ValueError('class too small for prototypes')
                    km=KMeans(n_clusters=3,random_state=20261008,n_init=10).fit(part)
                    prototypes.append(normalize(km.cluster_centers_));classes_for_prototype.extend([klass]*3)
                proto=np.concatenate(prototypes);class_index=np.array(classes_for_prototype)
                matched=class_index[np.argmax(x@proto.T,axis=1)]
                scored={}
                for s,ids in split_ids.items():
                    scored[s]={'knn5':classification(truth[ids],knn[ids],classes),
                               'prototype3':classification(truth[ids],matched[ids],classes),
                               'retrieval_purity_at5':float(np.mean((p5[ids]==truth[ids,None]).mean(1))),
                               'balanced_retrieval_purity_at5':float(np.mean([(p5[ids[truth[ids]==c]]==c).mean()
                                  for c in range(classes)]))}
                results[taxonomy]=scored
                if taxonomy=='coarse3':
                    key=f'{name}_{normalization}'
                    all_predictions[f'{key}_knn5']=knn
                    all_predictions[f'{key}_prototype3']=matched
                    composition,patches=selected_composition(rows,truth,matched,split)
                    arm.setdefault('selected_cell_composition',{})[normalization]=composition
                    compositions.extend([{'view':name,'normalization':normalization,'method':'prototype3',**v} for v in patches])
            if normalization=='train_zscore_unit':
                pca=PCA(n_components=64,random_state=20261008).fit(x[train])
                projected=pca.transform(x)
                clusters={}
                for k in (3,5,10):
                    cluster=KMeans(n_clusters=k,random_state=20261008,n_init=10).fit(projected[train]).predict(projected)
                    clusters[str(k)]={s:{'ari3':float(adjusted_rand_score(labels3[ids],cluster[ids])),
                                         'nmi3':float(normalized_mutual_info_score(labels3[ids],cluster[ids])),
                                         'ari5':float(adjusted_rand_score(labels5[ids],cluster[ids]))} for s,ids in split_ids.items()}
                arm['clustering']=clusters
            arm['normalizations'][normalization]=results
        report['views'][name]=arm
        (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        val=arm['normalizations']['train_zscore_unit']['coarse3']['val']['prototype3']['macro_f1']
        sup=arm['normalizations']['train_zscore_unit']['coarse3']['supplement_test']['prototype3']['macro_f1']
        print(json.dumps({'event':'view_complete','view':name,'val_macro_f1':val,'supplement_test_macro_f1':sup}),flush=True)
    all_predictions.to_csv(a.output/'predictions.csv',index=False)
    pd.DataFrame(compositions).to_csv(a.output/'selected_cell_compositions.csv',index=False)
    report['status']='complete'
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
