#!/usr/bin/env python3
"""Five-class Panel1 diagnostic on the existing immutable cell feature cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('timestamped FewClick artifact directory required')
    a.output.mkdir(parents=True, exist_ok=False)
    cells = pd.read_csv(a.source/'cells.csv')
    manifest = json.loads((a.source/'manifest.json').read_text())
    if hashlib.sha256((a.source/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('source manifest mismatch')
    with np.load(a.source/'features.npz', allow_pickle=False) as cache:
        features = cache['features']
    if features.shape != (len(cells), 768) or not np.isfinite(features).all():
        raise ValueError('invalid source feature cache')
    x = torch.nn.functional.normalize(torch.from_numpy(features), dim=1).numpy()
    y = cells.label.to_numpy(dtype=int)
    ids = {s: np.flatnonzero(cells.split.eq(s).to_numpy()) for s in ('train', 'val', 'test')}
    candidates = {
        'balanced_logistic': make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight='balanced', max_iter=2000)),
        'balanced_rbf': make_pipeline(StandardScaler(), SVC(
            C=1., gamma='scale', class_weight='balanced')),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(x[ids['train']], y[ids['train']])
        validation[name] = score(y[ids['val']], model.predict(x[ids['val']]))
    selected = max(candidates, key=lambda name: (validation[name]['macro_f1'], name))
    model = candidates[selected]
    test_pred = model.predict(x[ids['test']])
    table = cells.iloc[ids['test']].copy(); table['predicted'] = test_pred
    table.to_csv(a.output/'test_predictions.csv', index=False)
    joblib.dump({'model': model, 'classes': CLASSES,
                 'source_manifest_sha256': manifest['manifest_sha256']}, a.output/'classifier.joblib')
    by_group = pd.crosstab([cells.split, cells.group], cells.label).rename(columns=dict(enumerate(CLASSES)))
    by_group.to_csv(a.output/'class_by_spatial_group.csv')
    pairwise = []
    for first in range(len(CLASSES)):
        for second in range(first + 1, len(CLASSES)):
            train_pair = ids['train'][np.isin(y[ids['train']], [first, second])]
            val_pair = ids['val'][np.isin(y[ids['val']], [first, second])]
            probe = make_pipeline(StandardScaler(), LogisticRegression(
                C=1., class_weight='balanced', max_iter=2000))
            probe.fit(x[train_pair], y[train_pair])
            predicted = probe.predict(x[val_pair])
            pairwise.append({'classes': [CLASSES[first], CLASSES[second]],
                'train_support': [int((y[train_pair] == first).sum()), int((y[train_pair] == second).sum())],
                'val_support': [int((y[val_pair] == first).sum()), int((y[val_pair] == second).sum())],
                'val_macro_f1': float(f1_score(y[val_pair], predicted,
                    labels=[first, second], average='macro', zero_division=0))})
    report = {'status': 'complete', 'scope': 'exploratory; test previously examined',
              'source_manifest_sha256': manifest['manifest_sha256'],
              'train_support': manifest['support']['train'], 'val_support': manifest['support']['val'],
              'test_support': manifest['support']['test'],
              'groups_per_class': {s: [int(cells[(cells.split == s) & (cells.label == c)].group.nunique())
                                      for c in range(len(CLASSES))] for s in ids},
              'accepted_match_distance_median': [float(cells[cells.label == c].distance.median())
                                                 for c in range(len(CLASSES))],
              'validation_candidates': validation, 'selected_on_val': selected,
              'pairwise_validation': pairwise,
              'test': score(y[ids['test']], test_pred),
              'test_majority': score(y[ids['test']], np.full(len(ids['test']),
                    np.bincount(y[ids['train']], minlength=5).argmax()))}
    write_json(a.output/'metrics.json', report)
    print(json.dumps({'selected_on_val': selected, 'val_macro_f1': validation[selected]['macro_f1'],
                      'test': report['test'], 'groups_per_class': report['groups_per_class']}), flush=True)


if __name__ == '__main__':
    main()
