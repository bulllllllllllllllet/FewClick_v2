#!/usr/bin/env python3
"""Refit Panel1 linear heads with scaled frozen features; keep earlier run immutable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scale', type=float, default=16.)
    p.add_argument('--epochs', type=int, default=30)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2') or a.scale <= 0:
        p.error('use positive scale and FewClick artifact root')
    a.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(a.source/'cells.csv')
    manifest = json.loads((a.source/'manifest.json').read_text())
    if hashlib.sha256((a.source/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('source manifest mismatch')
    with np.load(a.source/'features.npz', allow_pickle=False) as cached:
        raw = cached['features']
    if raw.shape != (len(data), 768) or not np.isfinite(raw).all():
        raise ValueError('invalid source features')
    x = torch.nn.functional.normalize(torch.tensor(raw), dim=1) * a.scale
    y = torch.tensor(data.label.to_numpy(), dtype=torch.long)
    ids = {s: torch.tensor(np.flatnonzero(data.split == s)) for s in ('train', 'val', 'test')}
    counts = torch.bincount(y[ids['train']], minlength=5)
    weights = counts.float().rsqrt(); weights /= weights.mean()
    results = []
    for seed in (17, 29, 43):
        torch.manual_seed(seed)
        head = torch.nn.Linear(768, 5)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
        best, history = -1., []
        for epoch in range(a.epochs):
            head.train()
            for batch in ids['train'][torch.randperm(len(ids['train']))].split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(x[batch]), y[batch], weight=weights)
                loss.backward(); optimizer.step()
            head.eval()
            with torch.no_grad():
                valid = score(y[ids['val']].numpy(), head(x[ids['val']]).argmax(1).numpy())
            history.append({'epoch': epoch, 'val': valid})
            if valid['macro_f1'] > best:
                best = valid['macro_f1']
                torch.save({'head': head.state_dict(), 'classes': CLASSES,
                    'feature_scale': a.scale, 'epoch': epoch,
                    'encoder': '/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth',
                    'manifest_sha256': manifest['manifest_sha256'], 'crop_size': 128},
                    a.output/f'head_{seed}.pth')
        checkpoint = torch.load(a.output/f'head_{seed}.pth', map_location='cpu', weights_only=False)
        head.load_state_dict(checkpoint['head']); head.eval()
        with torch.no_grad():
            probs = head(x[ids['test']]).softmax(1).numpy()
        pred = probs.argmax(1)
        test = score(y[ids['test']].numpy(), pred)
        table = data.iloc[ids['test'].numpy()].copy(); table['predicted'] = pred
        for c, name in enumerate(CLASSES): table[f'p_{name}'] = probs[:, c]
        table.to_csv(a.output/f'test_predictions_{seed}.csv', index=False)
        results.append({'seed': seed, 'best_epoch': checkpoint['epoch'], 'test': test, 'history': history})
        print(json.dumps({'seed': seed, 'best_epoch': checkpoint['epoch'], 'test': test}), flush=True)
    write_json(a.output/'metrics.json', {'status': 'complete', 'source': str(a.source),
        'source_manifest_sha256': manifest['manifest_sha256'], 'feature_scale': a.scale,
        'support': manifest['support'], 'runs': results,
        'mean_test_macro_f1': float(np.mean([r['test']['macro_f1'] for r in results])),
        'scope': 'exploratory follow-up on previously evaluated test split; not fresh held-out confirmation'})


if __name__ == '__main__':
    main()
