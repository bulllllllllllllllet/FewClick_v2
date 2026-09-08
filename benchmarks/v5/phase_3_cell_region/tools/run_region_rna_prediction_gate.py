#!/usr/bin/env python3
"""Spatial-block gate: can frozen H&E region/cell representations predict RNA targets?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/region_rna_prediction_gate'))
    parser.add_argument('--timestamp', required=True)
    parser.add_argument('--shuffle-seed', type=int, default=20260908)
    return parser.parse_args()


def region_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    tokens = data['region_tokens'].astype(np.float64)
    cells = data['cells'].astype(np.float64)
    valid = data['cell_valid'].astype(np.float64)
    assignment = data['assignment_at_cells'].astype(np.float64) * valid[:, None, :]
    coverage = assignment.sum(2)
    context = np.einsum('pkn,pnd->pkd', assignment / coverage[:, :, None].clip(1e-8), cells)
    return np.concatenate((tokens, context, np.log1p(coverage)[:, :, None]), axis=2)


def evaluate(features, targets, groups, *, shuffled: bool, seed: int) -> list[dict]:
    rows = []
    for fold, (train_patch, test_patch) in enumerate(GroupKFold(n_splits=4).split(np.arange(len(groups)), groups=groups)):
        x_train, x_test = features[train_patch].reshape(-1, features.shape[-1]), features[test_patch].reshape(-1, features.shape[-1])
        y_train, y_test = targets[train_patch].reshape(-1, targets.shape[-1]), targets[test_patch].reshape(-1, targets.shape[-1])
        if shuffled:
            y_train = y_train[np.random.default_rng(seed + fold).permutation(len(y_train))]
        x_scale, y_scale = StandardScaler().fit(x_train), StandardScaler().fit(y_train)
        model = RidgeCV(alphas=np.logspace(-3, 4, 12)).fit(x_scale.transform(x_train), y_scale.transform(y_train))
        prediction = y_scale.inverse_transform(model.predict(x_scale.transform(x_test)))
        baseline = y_train.mean(0, keepdims=True)
        mse = np.mean((prediction - y_test) ** 2)
        baseline_mse = np.mean((baseline - y_test) ** 2)
        rows.append({'fold': fold, 'test_patches': int(len(test_patch)), 'test_tokens': int(len(y_test)), 'mse': float(mse), 'baseline_mse': float(baseline_mse), 'normalized_mse': float(mse / max(baseline_mse, 1e-12)), 'r2_vs_train_mean': float(1 - mse / max(baseline_mse, 1e-12)), 'alpha': float(model.alpha_)})
    return rows


def main() -> None:
    args = parse_args(); output = args.output_root / f'region_rna_prediction_gate_{args.timestamp}'
    if output.exists(): raise FileExistsError(output)
    contract = np.load(args.contract_dir / 'region_rna_contract.npz')
    manifest = pd.read_csv(args.contract_dir / 'patch_manifest.csv')
    feature = region_features(contract); target = contract['rna_targets'].astype(np.float64)
    if feature.shape[:2] != target.shape[:2] or len(manifest) != len(feature) or not (np.isfinite(feature).all() and np.isfinite(target).all()):
        raise RuntimeError('invalid region/RNA contract cache')
    groups = (manifest.left_px >= manifest.left_px.median()).astype(int) + 2 * (manifest.top_px >= manifest.top_px.median()).astype(int)
    if len(np.unique(groups)) != 4: raise RuntimeError(f'expected four spatial groups, got {np.unique(groups)}')
    rows = []
    for arm, shuffled in [('paired_rna_target', False), ('shuffled_rna_target', True)]:
        for row in evaluate(feature, target, groups.to_numpy(), shuffled=shuffled, seed=args.shuffle_seed): rows.append({'arm': arm, **row})
    output.mkdir(parents=True)
    result = pd.DataFrame(rows); result.to_csv(output / 'fold_metrics.csv', index=False)
    grouped = result.groupby('arm')[['normalized_mse', 'r2_vs_train_mean']].agg(['mean', 'std']).round(6)
    summary = {arm: {metric: {stat: float(grouped.loc[arm, (metric, stat)]) for stat in ('mean', 'std')} for metric in ('normalized_mse', 'r2_vs_train_mean')} for arm in grouped.index}
    paired = summary['paired_rna_target']['r2_vs_train_mean']['mean']; shuffled = summary['shuffled_rna_target']['r2_vs_train_mean']['mean']
    metadata = {'purpose': 'frozen Phase-2 token + assignment-weighted H&E cell context predicts RNA program; RNA is target only', 'contract_dir': str(args.contract_dir), 'patches': int(len(feature)), 'tokens_per_patch': int(feature.shape[1]), 'feature_dim': int(feature.shape[2]), 'spatial_groups': groups.to_dict(), 'summary': summary, 'paired_minus_shuffled_r2': paired - shuffled, 'acceptance': 'paired target must improve R2 over shuffled target; this is a 9-patch contract gate, not segmentation evidence'}
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(output), **metadata}, ensure_ascii=False))


if __name__ == '__main__': main()
