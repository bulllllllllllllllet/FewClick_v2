#!/usr/bin/env python3
"""Diagnostic closed-set H&E classifier: epithelial versus blood endothelial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from benchmarks.v5.tools.run_crc_panel1_cell_identity import PREPROCESS, WEIGHTS, load_ctranspath, soft_attention, write_json

NAMES = ['Epithelial', 'Blood_endothelial']


def evaluate(truth, pred):
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=[0, 1], zero_division=0)
    return {'macro_f1': float(f1.mean()), 'accuracy': float(np.mean(truth == pred)),
            'per_class': [dict(name=NAMES[i], support=int(support[i]), precision=float(precision[i]),
                               recall=float(recall[i]), f1=float(f1[i])) for i in range(2)],
            'confusion': confusion_matrix(truth, pred, labels=[0, 1]).tolist()}


def train(a):
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.source/'manifest.json').read_text())
    if hashlib.sha256((a.source/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('source manifest mismatch')
    cells = pd.read_csv(a.source/'cells.csv')
    with np.load(a.source/'features.npz', allow_pickle=False) as cache:
        raw = cache['features']
    if raw.shape != (len(cells), 768) or not np.isfinite(raw).all():
        raise ValueError('invalid source features')
    x = torch.nn.functional.normalize(torch.from_numpy(raw), dim=1).numpy()
    choice = cells.label.isin([0, 1]).to_numpy()
    splits = {s: np.flatnonzero(choice & cells.split.eq(s).to_numpy()) for s in ('train', 'val', 'test')}
    candidates = {
        'linear': make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1000)),
        'rbf': make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', class_weight='balanced')),
    }
    y = cells.label.to_numpy()
    reports = {}
    for name, model in candidates.items():
        model.fit(x[splits['train']], y[splits['train']])
        reports[name] = evaluate(y[splits['val']], model.predict(x[splits['val']]))
    selected = max(candidates, key=lambda name: (reports[name]['macro_f1'], name))
    chosen = candidates[selected]
    truth, pred = y[splits['test']], chosen.predict(x[splits['test']])
    result = {'status': 'complete', 'scope': 'two-class only; exploratory on previously evaluated spatial test split',
              'source_manifest_sha256': manifest['manifest_sha256'], 'selected_on_val': selected,
              'validation_candidates': reports, 'test': evaluate(truth, pred),
              'support': {s: np.bincount(y[idx], minlength=2).tolist() for s, idx in splits.items()},
              'test_majority': evaluate(truth, np.zeros_like(truth))}
    joblib.dump({'model': chosen, 'classes': NAMES, 'source_manifest_sha256': manifest['manifest_sha256'],
                 'encoder': str(WEIGHTS), 'crop_size': 128}, a.output/'classifier.joblib')
    table = cells.iloc[splits['test']].copy(); table['predicted'] = pred
    table.to_csv(a.output/'test_predictions.csv', index=False)
    write_json(a.output/'metrics.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def predict(a):
    torch.set_num_threads(8)
    package = joblib.load(a.model)
    image = np.asarray(Image.open(a.image).convert('RGB'))
    mask = np.asarray(Image.open(a.mask)) > 0
    if image.shape != (128, 128, 3) or mask.shape != (128, 128) or not mask.any():
        raise ValueError('require 128x128 H&E crop and target nucleus mask')
    encoder = load_ctranspath(Path(package['encoder']), torch.device(a.device))
    with torch.inference_mode():
        feature = encoder(PREPROCESS(soft_attention(image, mask))[None].to(a.device)).cpu()
        x = torch.nn.functional.normalize(feature, dim=1).numpy()
    pred = int(package['model'].predict(x)[0])
    print(json.dumps({'predicted': NAMES[pred], 'scope': 'forced choice between epithelial and blood endothelial only'}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'predict'])
    p.add_argument('--source', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--model', type=Path); p.add_argument('--image', type=Path); p.add_argument('--mask', type=Path)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.mode == 'train':
        if a.source is None or a.output is None or not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
            p.error('source and timestamped FewClick output required')
        train(a)
    else:
        predict(a)


if __name__ == '__main__':
    main()
