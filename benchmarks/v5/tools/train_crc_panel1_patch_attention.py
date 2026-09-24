#!/usr/bin/env python3
"""Compare target-only and within-patch cell cross-attention on frozen H&E features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score


def load_set(manifest: Path, features: Path, target_context256: bool = False):
    table_path = manifest / 'cells.csv'
    rows = pd.read_csv(table_path)
    metadata = manifest / 'manifest.json'
    if not metadata.exists():
        metadata = manifest / 'audit.json'
    meta = json.loads(metadata.read_text())
    if hashlib.sha256(table_path.read_bytes()).hexdigest() != meta['manifest_sha256']:
        raise ValueError(f'manifest hash mismatch: {manifest}')
    with np.load(features / 'features.npz', allow_pickle=False) as cache:
        image_features = cache['features128'].copy()
        context_features = cache['features256'].copy() if target_context256 else None
    if image_features.shape != (len(rows), 768) or not np.isfinite(image_features).all():
        raise ValueError(f'invalid features: {features}')
    if rows[['patch', 'he_id']].duplicated().any():
        raise ValueError('duplicate H&E nucleus')
    if rows.groupby('patch').split.nunique().max() != 1:
        raise ValueError('patch spans data splits')
    image_features /= np.maximum(np.linalg.norm(image_features, axis=1, keepdims=True), 1e-8)
    if target_context256:
        if context_features.shape != image_features.shape or not np.isfinite(context_features).all():
            raise ValueError(f'invalid 256px features: {features}')
        context_features /= np.maximum(np.linalg.norm(context_features, axis=1, keepdims=True), 1e-8)
        image_features = np.concatenate([image_features, context_features], axis=1)
    return rows, torch.from_numpy(image_features), meta


def build_neighbors(rows: pd.DataFrame, k: int):
    """Only geometry and patch ID determine peers; labels are never used."""
    n = len(rows)
    neighbors = np.repeat(np.arange(n, dtype=np.int64)[:, None], k, axis=1)
    valid = np.zeros((n, k), dtype=bool)
    offset = np.zeros((n, k, 2), dtype=np.float32)
    for _, group in rows.groupby('patch', sort=False):
        ids = group.index.to_numpy()
        xy = group[['he_x', 'he_y']].to_numpy(np.float32)
        distance = ((xy[:, None] - xy[None]) ** 2).sum(2)
        np.fill_diagonal(distance, np.inf)
        order = np.argsort(distance, axis=1)[:, :min(k, len(ids) - 1)]
        for local, peers in enumerate(order):
            if len(peers):
                target = ids[local]
                neighbors[target, :len(peers)] = ids[peers]
                valid[target, :len(peers)] = True
                offset[target, :len(peers)] = (xy[peers] - xy[local]) / 512.
    return torch.from_numpy(neighbors), torch.from_numpy(valid), torch.from_numpy(offset)


class CellPatchAttention(nn.Module):
    def __init__(self, with_context: bool, target_context256: bool = False, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.with_context = with_context
        self.target = nn.Sequential(nn.Linear(1536 if target_context256 else 768, hidden), nn.LayerNorm(hidden), nn.GELU())
        if with_context:
            self.neighbor = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.position = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, hidden))
            self.attention = nn.MultiheadAttention(hidden, heads, dropout=.1, batch_first=True)
            self.context_norm = nn.LayerNorm(hidden)
            self.gate = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                                  nn.GELU(), nn.Dropout(.25), nn.Linear(hidden, len(CLASSES)))

    def forward(self, image_features, index, neighbors, valid, offset):
        target = self.target(image_features[index])
        context = torch.zeros_like(target)
        if self.with_context:
            # MultiheadAttention rejects an all-masked row. Supply one zeroed dummy
            # key, then explicitly discard its output for singleton patches.
            has_peer = valid.any(dim=1)
            safe_valid = valid.clone()
            safe_valid[:, 0] = safe_valid[:, 0] | ~has_peer
            peer = self.neighbor(image_features[neighbors, :768])
            key = peer + self.position(offset)
            attended, _ = self.attention(target[:, None], key, peer,
                                         key_padding_mask=~safe_valid, need_weights=False)
            attended = self.context_norm(attended[:, 0])
            gated = torch.sigmoid(self.gate(torch.cat([target, attended], dim=1))) * attended
            context = torch.where(has_peer[:, None], gated, context)
        return self.head(torch.cat([target, context], dim=1))


def predict(model, data, indices, batch_size):
    feature, neigh, valid, offset = data
    result = []
    model.eval()
    with torch.inference_mode():
        for part in indices.split(batch_size):
            result.append(model(feature, part, neigh[part], valid[part], offset[part]).argmax(1).cpu())
    return torch.cat(result).numpy()


def prepare(manifest, features, k, device, target_context256=False):
    rows, feature, meta = load_set(manifest, features, target_context256)
    neighbor, valid, offset = build_neighbors(rows, k)
    return rows, (feature.to(device), neighbor.to(device), valid.to(device), offset.to(device)), meta, valid.sum(1).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--supplement-manifest', type=Path, required=True)
    p.add_argument('--supplement-features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:5')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--neighbors', type=int, default=16)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--target-context256', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.neighbors < 1:
        p.error('epochs, batch-size and neighbors must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    device = torch.device(a.device)
    rows, data, meta, counts = prepare(a.manifest, a.features, a.neighbors, device, a.target_context256)
    supplement, sup_data, sup_meta, sup_counts = prepare(
        a.supplement_manifest, a.supplement_features, a.neighbors, device, a.target_context256)
    if set(rows.loc[rows.split.eq('train'), 'group']) & set(supplement.group):
        raise ValueError('supplement spatial group overlaps training')
    if set(rows.loc[rows.split.eq('train'), 'patch']) & set(supplement.patch):
        raise ValueError('supplement patch overlaps training')
    labels = torch.tensor(rows.label.to_numpy(int), device=device)
    ids = {split: torch.tensor(np.flatnonzero(rows.split.eq(split)), device=device)
           for split in ('train', 'val', 'test')}
    if any(len(x) == 0 for x in ids.values()):
        raise ValueError('empty split')
    class_counts = torch.bincount(labels[ids['train']], minlength=len(CLASSES)).float()
    if (class_counts == 0).any():
        raise ValueError('missing train class')
    weights = class_counts.rsqrt()
    weights /= weights.mean()
    output = {'status': 'complete', 'classes': CLASSES, 'seed': a.seed, 'neighbors': a.neighbors,
              'target_context256': a.target_context256,
              'manifest_sha256': meta['manifest_sha256'],
              'supplement_sha256': sup_meta['manifest_sha256'],
              'train_class_counts': class_counts.int().tolist(),
              'peer_counts': {split: {'mean': float(counts[ids[split].cpu()].mean()),
                                      'singletons': int((counts[ids[split].cpu()] == 0).sum())}
                              for split in ids},
              'supplement_peer_counts': {split: {'mean': float(sup_counts[supplement.split.eq(split)].mean()),
                                                 'singletons': int((sup_counts[supplement.split.eq(split)] == 0).sum())}
                                         for split in ('val', 'test')}, 'arms': {}}
    for arm in ('target_only', 'cross_attention'):
        torch.manual_seed(a.seed)
        model = CellPatchAttention(arm == 'cross_attention', a.target_context256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        best = (-1., None, -1)
        stale = 0
        for epoch in range(a.epochs):
            model.train()
            order = ids['train'][torch.randperm(len(ids['train']), device=device)]
            for part in order.split(a.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = model(data[0], part, data[1][part], data[2][part], data[3][part])
                loss = F.cross_entropy(logits, labels[part], weight=weights)
                loss.backward()
                optimizer.step()
            pred = predict(model, data, ids['val'], a.batch_size)
            value = score(rows.label.to_numpy(int)[ids['val'].cpu().numpy()], pred)['macro_f1']
            if value > best[0] + 1e-6:
                best = (value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
                stale = 0
            else:
                stale += 1
            print(json.dumps({'arm': arm, 'epoch': epoch, 'val_macro_f1': value,
                              'best_val_macro_f1': best[0]}), flush=True)
            if stale >= 12:
                break
        model.load_state_dict(best[1])
        torch.save({'state_dict': best[1], 'arm': arm, 'seed': a.seed,
                    'neighbors': a.neighbors, 'target_context256': a.target_context256,
                    'manifest_sha256': meta['manifest_sha256'],
                    'classes': CLASSES}, a.output / f'{arm}.pth')
        result = {'best_epoch': best[2]}
        for split in ('val', 'test'):
            index = ids[split]
            truth = rows.label.to_numpy(int)[index.cpu().numpy()]
            pred = predict(model, data, index, a.batch_size)
            result[split] = score(truth, pred)
        sup_index = torch.arange(len(supplement), device=device)
        sup_pred = predict(model, sup_data, sup_index, a.batch_size)
        truth = supplement.label.to_numpy(int)
        result['supplement'] = {
            split: score(truth[supplement.split.eq(split)], sup_pred[supplement.split.eq(split)])
            for split in ('val', 'test')}
        output['arms'][arm] = result
        (a.output / 'metrics.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'arm': arm, 'test': result['test']['macro_f1'],
                          'supplement_test': result['supplement']['test']['macro_f1']}), flush=True)
    print(json.dumps({'event': 'complete', 'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    main()
