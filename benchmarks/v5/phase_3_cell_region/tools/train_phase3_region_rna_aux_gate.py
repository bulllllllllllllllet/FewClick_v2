#!/usr/bin/env python3
"""Train a v4-contract CellToRegionAttention RNA auxiliary head on cached patches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate'))
    parser.add_argument('--timestamp', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260908, 20260909, 20260910])
    return parser.parse_args()


class RNAStudent(torch.nn.Module):
    def __init__(self, cell_dim: int, target_dim: int):
        super().__init__()
        self.fusion = CellToRegionAttention(region_dim=256, cell_dim=cell_dim, hidden_dim=128)
        self.rna_head = torch.nn.Linear(256, target_dim)

    def forward(self, tokens, assignment, cells, valid):
        fused = self.fusion(tokens, assignment, cells, valid, valid.sum(1).float())['fused_tokens']
        return self.rna_head(fused)


def train_fold(value, train, test, *, shuffled, seed, args, device):
    torch.manual_seed(seed)
    tokens, cells, valid, assignment, target = (value[name] for name in ('region_tokens', 'cells', 'cell_valid', 'assignment_at_cells', 'rna_targets'))
    y_train = target[train].clone()
    mean, std = y_train.reshape(-1, y_train.shape[-1]).mean(0), y_train.reshape(-1, y_train.shape[-1]).std(0).clamp_min(1e-6)
    if shuffled:
        flat = y_train.reshape(-1, y_train.shape[-1]); y_train = flat[torch.randperm(len(flat), generator=torch.Generator().manual_seed(seed))].reshape_as(y_train)
    student = RNAStudent(cells.shape[-1], target.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    student.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = student(tokens[train], assignment[train], cells[train], valid[train])
        loss = torch.nn.functional.mse_loss((prediction - mean) / std, (y_train - mean) / std)
        if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
        loss.backward(); optimizer.step()
    student.eval()
    with torch.inference_mode(): prediction = student(tokens[test], assignment[test], cells[test], valid[test]).cpu().numpy()
    observed = target[test].cpu().numpy(); baseline = mean.cpu().numpy()[None, None, :]
    mse, baseline_mse = np.mean((prediction - observed) ** 2), np.mean((baseline - observed) ** 2)
    return float(loss.detach()), float(mse / max(baseline_mse, 1e-12)), float(1 - mse / max(baseline_mse, 1e-12))


def main():
    args = parse_args(); output = args.output_root / f'phase3_region_rna_aux_gate_{args.timestamp}'
    if output.exists(): raise FileExistsError(output)
    raw = np.load(args.contract_dir / 'region_rna_contract.npz')
    value = {name: torch.from_numpy(raw[name]).float() for name in ('region_tokens', 'cells', 'assignment_at_cells', 'rna_targets')}
    value['cell_valid'] = torch.from_numpy(raw['cell_valid']).bool()
    if not all(torch.isfinite(value[name]).all() for name in ('region_tokens', 'cells', 'assignment_at_cells', 'rna_targets')): raise RuntimeError('non-finite contract input')
    manifest = pd.read_csv(args.contract_dir / 'patch_manifest.csv')
    groups = ((manifest.left_px >= manifest.left_px.median()).astype(int) + 2 * (manifest.top_px >= manifest.top_px.median()).astype(int)).to_numpy()
    device = torch.device(args.device); value = {name: tensor.to(device) for name, tensor in value.items()}
    rows = []
    for arm, shuffled in [('paired_rna_target', False), ('shuffled_rna_target', True)]:
        for seed in args.seeds:
            for fold, (train, test) in enumerate(GroupKFold(n_splits=4).split(np.arange(len(groups)), groups=groups)):
                loss, nmse, r2 = train_fold(value, train, test, shuffled=shuffled, seed=seed, args=args, device=device)
                rows.append({'arm': arm, 'seed': seed, 'fold': fold, 'final_train_loss': loss, 'normalized_mse': nmse, 'r2_vs_train_mean': r2, 'train_patches': len(train), 'test_patches': len(test)})
    output.mkdir(parents=True); result = pd.DataFrame(rows); result.to_csv(output / 'fold_metrics.csv', index=False)
    grouped = result.groupby('arm')[['normalized_mse', 'r2_vs_train_mean']].agg(['mean', 'std']).round(6)
    summary = {arm: {metric: {stat: float(grouped.loc[arm, (metric, stat)]) for stat in ('mean', 'std')} for metric in ('normalized_mse', 'r2_vs_train_mean')} for arm in grouped.index}
    metadata = {'purpose': 'frozen Phase-2 + trainable v4-contract CellToRegionAttention and RNA auxiliary head; RNA target only', 'contract_dir': str(args.contract_dir), 'seeds': args.seeds, 'epochs': args.epochs, 'summary': summary, 'paired_minus_shuffled_r2': summary['paired_rna_target']['r2_vs_train_mean']['mean'] - summary['shuffled_rna_target']['r2_vs_train_mean']['mean'], 'acceptance': 'paired R2 must exceed shuffled R2; 9-patch gate only, not segmentation evidence'}
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(output), **metadata}, ensure_ascii=False))


if __name__ == '__main__': main()
