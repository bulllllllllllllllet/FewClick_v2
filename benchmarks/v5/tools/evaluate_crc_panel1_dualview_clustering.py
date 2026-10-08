#!/usr/bin/env python3
"""Spatially held-out clustering audit of Panel1 cell and wider-view features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score


def load(manifest: Path, cache: Path, cell_weights=()):
    path = manifest / 'cells.csv'
    frame = pd.read_csv(path)
    info = manifest / 'manifest.json'
    if not info.exists():
        info = manifest / 'audit.json'
    metadata = json.loads(info.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['manifest_sha256']:
        raise ValueError(f'manifest hash mismatch: {manifest}')
    with np.load(cache / 'features.npz', allow_pickle=False) as z:
        x128 = z['features128'].copy()
        x256 = z['features256'].copy()
    if x128.shape != x256.shape or x128.shape != (len(frame), 768):
        raise ValueError('cell and context cache shape mismatch')
    if not np.isfinite(x128).all() or not np.isfinite(x256).all():
        raise ValueError('non-finite features')
    if frame[['patch', 'he_id']].duplicated().any() or frame.groupby('patch').split.nunique().max() != 1:
        raise ValueError('duplicate cell or patch split leakage')
    if not frame.label.between(0, len(CLASSES)-1).all():
        raise ValueError('invalid labels')
    features = {'cell128': normalize(x128), 'context256': normalize(x256)}
    features['fused_equal'] = normalize(np.concatenate([features['cell128'], features['context256']], axis=1))
    for weight in cell_weights:
        name = f'fused_cell{weight:g}_context1'
        features[name] = normalize(np.concatenate(
            [weight * features['cell128'], features['context256']], axis=1))
    return frame, features, metadata


def evaluate(y, cluster, train_map):
    prediction = np.array([train_map[c] for c in cluster], dtype=np.int8)
    return {'ari': float(adjusted_rand_score(y, cluster)),
            'nmi': float(normalized_mutual_info_score(y, cluster)),
            'mapped': score(y, prediction)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--supplement-manifest', type=Path, required=True)
    p.add_argument('--supplement-features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clusters', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--pca-components', type=int, default=64)
    p.add_argument('--fusion-cell-weights', nargs='*', type=float, default=[])
    p.add_argument('--only-weighted', action='store_true')
    p.add_argument('--seed', type=int, default=20261008)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('output must be in the FewClick artifact root')
    if min(*a.clusters, a.pca_components) < 2:
        p.error('invalid cluster/PCA count')
    if any(not np.isfinite(w) or w <= 0 for w in a.fusion_cell_weights):
        p.error('fusion cell weights must be positive and finite')
    if a.only_weighted and not a.fusion_cell_weights:
        p.error('--only-weighted requires --fusion-cell-weights')
    a.output.mkdir(parents=True, exist_ok=False)
    frame, features, source = load(a.manifest, a.features, a.fusion_cell_weights)
    supplement, sup_features, sup_source = load(a.supplement_manifest, a.supplement_features, a.fusion_cell_weights)
    if a.only_weighted:
        features = {k:v for k,v in features.items() if k.startswith('fused_cell')}
        sup_features = {k:v for k,v in sup_features.items() if k.startswith('fused_cell')}
    splits = {name: np.flatnonzero(frame.split.eq(name)) for name in ('train', 'val', 'test')}
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError('empty split')
    if any(set(frame.iloc[splits['train']].group) & set(frame.iloc[splits[s]].group) for s in ('val', 'test')):
        raise ValueError('train spatial group overlaps evaluation')
    if set(frame.iloc[splits['train']].group) & set(supplement.group):
        raise ValueError('supplement spatial group overlaps train')
    if set(frame.patch) & set(supplement.patch):
        raise ValueError('supplement reuses base manifest patch')
    labels = frame.label.to_numpy(int)
    sup_labels = supplement.label.to_numpy(int)
    report = {'status': 'complete', 'method': f'train-only PCA{a.pca_components} + KMeans; class map from train majority',
              'classes': CLASSES, 'base_manifest_sha256': source['manifest_sha256'],
              'supplement_manifest_sha256': sup_source['manifest_sha256'],
              'samples': {s: int(len(v)) for s,v in splits.items()},
              'supplement_samples': {s:int(supplement.split.eq(s).sum()) for s in ('val','test')},
              'source_paths': {'manifest':str(a.manifest),'features':str(a.features),
                               'supplement_manifest':str(a.supplement_manifest),
                               'supplement_features':str(a.supplement_features)},
              'arms': {}}
    for arm, values in features.items():
        pca = PCA(n_components=a.pca_components, svd_solver='randomized', random_state=a.seed)
        pca.fit(values[splits['train']])
        low = pca.transform(values)
        sup_low = pca.transform(sup_features[arm])
        if not np.isfinite(low).all() or not np.isfinite(sup_low).all():
            raise ValueError('non-finite projection')
        # Unit length prevents overall feature magnitude from defining clusters.
        low = normalize(low)
        sup_low = normalize(sup_low)
        np.savez_compressed(a.output / f'{arm}_projection.npz',
            embedding=low.astype(np.float32), supplement_embedding=sup_low.astype(np.float32),
            base_manifest_sha256=np.array(source['manifest_sha256']),
            supplement_manifest_sha256=np.array(sup_source['manifest_sha256']))
        joblib.dump(pca, a.output / f'{arm}_pca.joblib')
        arm_result = {'input_dim': values.shape[1], 'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
                      'clusterings': {}}
        for k in a.clusters:
            model = KMeans(n_clusters=k, random_state=a.seed, n_init=10, max_iter=300)
            train_cluster = model.fit_predict(low[splits['train']])
            all_cluster = model.predict(low)
            sup_cluster = model.predict(sup_low)
            mapping = {}
            for c in range(k):
                histogram = np.bincount(labels[splits['train']][train_cluster == c], minlength=len(CLASSES))
                mapping[c] = int(histogram.argmax())
            result = {'train_cluster_counts': np.bincount(train_cluster, minlength=k).tolist(),
                      'cluster_to_class': [mapping[c] for c in range(k)],
                      'base': {s: evaluate(labels[ix], all_cluster[ix], mapping)
                               for s,ix in splits.items() if s != 'train'},
                      'supplement': {s: evaluate(sup_labels[mask], sup_cluster[mask], mapping)
                                     for s in ('val','test') for mask in [supplement.split.eq(s).to_numpy()]}}
            np.savez_compressed(a.output / f'{arm}_k{k}_assignments.npz',
                base_clusters=all_cluster.astype(np.int16),
                supplement_clusters=sup_cluster.astype(np.int16))
            joblib.dump(model, a.output / f'{arm}_k{k}_model.joblib')
            arm_result['clusterings'][str(k)] = result
            print(json.dumps({'arm':arm,'k':k,'test_ari':result['base']['test']['ari'],
                'test_macro_f1':result['base']['test']['mapped']['macro_f1'],
                'supplement_test_macro_f1':result['supplement']['test']['mapped']['macro_f1']}),flush=True)
        report['arms'][arm] = arm_result
        (a.output / 'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output / 'run.json').write_text(json.dumps({'pid':os.getpid(),'argv':vars(a)},default=str,indent=2)+'\n')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__ == '__main__':
    main()
