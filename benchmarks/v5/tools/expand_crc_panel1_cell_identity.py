#!/usr/bin/env python3
"""Expand Panel1 training cells and encode rare-class H&E augmentations."""
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from benchmarks.v5.tools.run_crc_panel1_cell_identity import (
    CLASSES, PREPROCESS, SOURCE, WEIGHTS, crop_mask, crop_region,
    load_ctranspath, score, soft_attention, write_json,
)


def identity(row):
    return (row.patch, int(row.he_id))


def check_manifest(old_root, new_root):
    old = pd.read_csv(old_root/'cells.csv')
    new = pd.read_csv(new_root/'cells.csv')
    old_meta = json.loads((old_root/'manifest.json').read_text())
    new_meta = json.loads((new_root/'manifest.json').read_text())
    for frame, meta, root in ((old, old_meta, old_root), (new, new_meta, new_root)):
        if hashlib.sha256((root/'cells.csv').read_bytes()).hexdigest() != meta['manifest_sha256']:
            raise ValueError('manifest checksum mismatch')
        if frame.duplicated(['mif_global_x', 'mif_global_y']).any():
            raise ValueError('duplicate mIF cell coordinates')
    columns = ['patch', 'he_id', 'label', 'mif_global_x', 'mif_global_y', 'group', 'split']
    for split in ('val', 'test'):
        before = old[old.split.eq(split)][columns].sort_values(columns).reset_index(drop=True)
        after = new[new.split.eq(split)][columns].sort_values(columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after, check_dtype=False)
    old_train = old[old.split.eq('train')]
    new_train = new[new.split.eq('train')]
    if not set(map(identity, old_train.itertuples())).issubset(set(map(identity, new_train.itertuples()))):
        raise ValueError('old training cell missing')
    train_groups = set(new_train.group)
    eval_groups = set(new[new.split.ne('train')].group)
    if train_groups & eval_groups:
        raise ValueError('train/eval spatial group conflict')
    before = old_meta['support']
    after = new_meta['support']
    return old, new, old_meta, new_meta, {'old_support': before,
        'expanded_support': after, 'new_unique_train_cells': len(new_train)-len(old_train),
        'old_train_groups': int(old_train.group.nunique()),
        'expanded_train_groups': int(new_train.group.nunique()),
        'val_and_test_identical': True, 'train_eval_groups_disjoint': True}


def transform_pair(image, mask, row_key, view):
    """Apply the same D4 geometry to H&E and nucleus, with mild color change."""
    seed = int(hashlib.sha256(f'{row_key}/{view}/20260923'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    turns = 1 + (view % 3)
    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    if view % 2:
        image, mask = np.fliplr(image), np.fliplr(mask)
    # Bounded staining/illumination variation; geometry remains identical.
    contrast = float(rng.uniform(.92, 1.08))
    brightness = float(rng.uniform(.92, 1.08))
    channel = rng.uniform(.97, 1.03, size=3)
    image = np.clip(((image.astype(np.float32)-128.)*contrast+128.)*brightness*channel, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def encode(model, images, device):
    if not images:
        return np.empty((0, 768), dtype=np.float32)
    values = []
    with torch.inference_mode():
        for batch in torch.stack(images).split(32):
            values.append(model(batch.to(device)).cpu().numpy())
    out = np.concatenate(values).astype(np.float32)
    if out.shape != (len(images), 768) or not np.isfinite(out).all():
        raise ValueError('invalid extracted visual features')
    return out


def extract(a):
    old, new, old_meta, new_meta, audit = check_manifest(a.old, a.expanded)
    a.output.mkdir(parents=True, exist_ok=False)
    with np.load(a.old/'features.npz', allow_pickle=False) as z:
        old_features = z['features']
    if old_features.shape != (len(old), 768) or not np.isfinite(old_features).all():
        raise ValueError('old feature cache mismatch')
    lookup = {identity(row): (idx, int(row.label)) for idx, row in enumerate(old.itertuples())}
    x = np.full((len(new), 768), np.nan, dtype=np.float32)
    reused = 0
    for idx, row in enumerate(new.itertuples()):
        old_match = lookup.get(identity(row))
        if old_match is not None:
            old_idx, old_label = old_match
            if old_label != row.label:
                raise ValueError('label changed on reused cell')
            x[idx] = old_features[old_idx]
            reused += 1
    device = torch.device(a.device)
    torch.set_num_threads(8)
    encoder = load_ctranspath(WEIGHTS, device).requires_grad_(False)
    records = {r['patch']: r for r in new_meta['patches']}
    aug_features, aug_source, aug_view = [], [], []
    completed, new_cells = 0, 0
    for patch, frame in new.groupby('patch', sort=False):
        required = frame[np.isnan(x[frame.index, 0]) | (frame.split.eq('train') & frame.label.isin([2, 3]))]
        if required.empty:
            continue
        image_path = SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path = SOURCE/'cell_matching/masks'/f'{patch}.npy'
        record = records[patch]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record['he_sha256'] or hashlib.sha256(mask_path.read_bytes()).hexdigest() != record['mask_sha256']:
            raise ValueError('image/mask changed')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.load(mask_path, allow_pickle=False)
        base_tensors, base_indices, extras = [], [], []
        for idx, row in required.iterrows():
            cy, cx = round(row.he_y), round(row.he_x)
            cropped_image = crop_region(image, cy, cx)
            target = crop_mask(mask, cy, cx) == int(row.he_id)
            if not target.any():
                raise ValueError('target H&E nucleus missing')
            if np.isnan(x[idx, 0]):
                base_tensors.append(PREPROCESS(soft_attention(cropped_image, target)))
                base_indices.append(idx)
            if row.split == 'train' and row.label in (2, 3):
                for view in (0, 1):
                    changed, changed_mask = transform_pair(cropped_image, target,
                        f'{patch}/{row.he_id}', view)
                    if not changed_mask.any() or changed_mask.sum() != target.sum():
                        raise ValueError('augmentation changed target nucleus area')
                    extras.append((idx, view, PREPROCESS(soft_attention(changed, changed_mask))))
        if base_tensors:
            x[base_indices] = encode(encoder, base_tensors, device)
            new_cells += len(base_indices)
        if extras:
            aug_features.append(encode(encoder, [item[2] for item in extras], device))
            aug_source.extend(item[0] for item in extras)
            aug_view.extend(item[1] for item in extras)
        completed += 1
        if completed % 20 == 0:
            print(json.dumps({'event': 'extracted', 'patches': completed,
                'new_independent_cells': new_cells, 'augmented_views': len(aug_source)}), flush=True)
        if a.smoke_patches and completed >= a.smoke_patches:
            break
    if a.smoke_patches:
        if not new_cells or not aug_source:
            raise ValueError('smoke did not exercise both new cells and augmented views')
        write_json(a.output/'smoke.json', {'status': 'complete', 'patches': completed,
            'new_cells': new_cells, 'augmented_views': len(aug_source), **audit})
        return
    if not np.isfinite(x).all():
        raise ValueError('incomplete full feature cache')
    aug = np.concatenate(aug_features) if aug_features else np.empty((0, 768), np.float32)
    if len(aug_source) != len(aug) or not np.isfinite(aug).all():
        raise ValueError('invalid augmentation cache')
    np.savez_compressed(a.output/'features.npz', features=x, augmented=aug,
        augmented_source_index=np.array(aug_source, np.int64), augmented_view=np.array(aug_view, np.int8))
    write_json(a.output/'feature_audit.json', {'status': 'complete', 'manifest_sha256': new_meta['manifest_sha256'],
        'old_manifest_sha256': old_meta['manifest_sha256'], 'reused_original_features': reused,
        'extracted_new_independent_features': new_cells, 'augmented_views': len(aug), **audit})


def compare(a):
    old, new, old_meta, new_meta, audit = check_manifest(a.old, a.expanded)
    meta = json.loads((a.features/'feature_audit.json').read_text())
    if meta['manifest_sha256'] != new_meta['manifest_sha256']:
        raise ValueError('feature/manifest mismatch')
    with np.load(a.features/'features.npz', allow_pickle=False) as z:
        x = z['features']; aug = z['augmented']; source_index = z['augmented_source_index']
    if x.shape != (len(new), 768) or len(aug) != len(source_index):
        raise ValueError('feature cache shape mismatch')
    if not new.iloc[source_index].split.eq('train').all() or not new.iloc[source_index].label.isin([2, 3]).all():
        raise ValueError('evaluation leakage in augmentation')
    a.output.mkdir(parents=True, exist_ok=False)
    x = torch.nn.functional.normalize(torch.from_numpy(x), dim=1).numpy()
    aug = torch.nn.functional.normalize(torch.from_numpy(aug), dim=1).numpy()
    y = new.label.to_numpy(dtype=int)
    splits = {s: np.flatnonzero(new.split.eq(s).to_numpy()) for s in ('train', 'val', 'test')}
    old_keys = set(map(identity, old[old.split.eq('train')].itertuples()))
    old_train = np.array([i for i in splits['train'] if identity(new.iloc[i]) in old_keys], dtype=int)
    arms = {'old_train': (x[old_train], y[old_train]),
            'expanded': (x[splits['train']], y[splits['train']]),
            'expanded_augmented': (np.concatenate([x[splits['train']], aug]),
                np.concatenate([y[splits['train']], y[source_index]]))}
    candidates = {
        'balanced_logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight='balanced', max_iter=2000)),
        'balanced_rbf': lambda: make_pipeline(StandardScaler(), SVC(
            C=1., gamma='scale', class_weight='balanced')),
    }
    results = {}
    for name, (train_x, train_y) in arms.items():
        validations, models = {}, {}
        for method, build in candidates.items():
            model = build(); model.fit(train_x, train_y)
            validations[method] = score(y[splits['val']], model.predict(x[splits['val']]))
            models[method] = model
        selected = max(validations, key=lambda key: (validations[key]['macro_f1'], key))
        model = models[selected]
        test_pred = model.predict(x[splits['test']])
        joblib.dump({'model': model, 'classes': CLASSES, 'manifest_sha256': new_meta['manifest_sha256'],
                     'arm': name}, a.output/f'{name}.joblib')
        predictions = new.iloc[splits['test']].copy(); predictions['predicted'] = test_pred
        predictions.to_csv(a.output/f'{name}_test_predictions.csv', index=False)
        results[name] = {'train_cells_including_augmented_views': len(train_y),
            'train_support_including_augmented_views': np.bincount(train_y, minlength=5).tolist(),
            'val_candidates': validations, 'selected_on_val': selected,
            'fixed_balanced_logistic_test': score(y[splits['test']],
                models['balanced_logistic'].predict(x[splits['test']])),
            'test': score(y[splits['test']], test_pred)}
        print(json.dumps({'arm': name, 'selected_on_val': selected,
            'val_macro_f1': validations[selected]['macro_f1'], 'test': results[name]['test']}), flush=True)
    write_json(a.output/'metrics.json', {'status': 'complete', 'scope': 'same previously examined spatial test; exploratory',
        'manifest_sha256': new_meta['manifest_sha256'], 'feature_audit': meta,
        'support': new_meta['support'], 'arms': results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['extract', 'compare'])
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--expanded', type=Path, required=True)
    p.add_argument('--features', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--smoke-patches', type=int, default=0)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('timestamped FewClick artifact path required')
    if a.mode == 'compare' and a.features is None:
        p.error('--features required')
    {'extract': extract, 'compare': compare}[a.mode](a)


if __name__ == '__main__':
    main()
