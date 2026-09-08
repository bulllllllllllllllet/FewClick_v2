#!/usr/bin/env python3
"""Select spatially spread, cell-dense non-overlapping CosMx H&E tiles from a derived index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache'))
    p.add_argument('--timestamp', required=True); p.add_argument('--tile-size', type=int, default=1536); p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--min-cells-per-patch', type=int, default=20); p.add_argument('--max-tiles', type=int, default=64); return p.parse_args()


def main():
    a = parse_args(); out = a.output_root / f'cosmx_fullslide_tile_manifest_{a.timestamp}'
    if out.exists(): raise FileExistsError(out)
    if a.tile_size % a.patch_size: raise ValueError('tile size must be divisible by patch size')
    cells = pd.read_csv(a.index_dir / 'cell_rna_index.csv.gz', dtype={'cell_id': str})
    meta = json.loads((a.index_dir / 'metadata.json').read_text()); width, height = meta['he_size']; side = a.tile_size // a.patch_size
    cells = cells[(cells.he_x >= 0) & (cells.he_x < width) & (cells.he_y >= 0) & (cells.he_y < height)].copy()
    cells['tile_x'] = (cells.he_x // a.tile_size).astype(int); cells['tile_y'] = (cells.he_y // a.tile_size).astype(int)
    records=[]
    for (tx, ty), part in cells.groupby(['tile_x', 'tile_y'], sort=True):
        x, y = int(tx * a.tile_size), int(ty * a.tile_size)
        if x + a.tile_size > width or y + a.tile_size > height: continue
        lx = ((part.he_x - x) // a.patch_size).astype(int); ly = ((part.he_y - y) // a.patch_size).astype(int)
        patch_count = np.zeros((side, side), np.int64)
        np.add.at(patch_count, (ly.to_numpy(), lx.to_numpy()), 1)
        if patch_count.min() < a.min_cells_per_patch: continue
        records.append({'tile_id': f'x{x}_y{y}', 'x': x, 'y': y, 'rna_cells': int(len(part)), 'minimum_patch_cells': int(patch_count.min()), 'mean_patch_cells': float(patch_count.mean()), 'patch_counts': '|'.join(map(str, patch_count.ravel().tolist()))})
    candidates = pd.DataFrame(records)
    if len(candidates) < a.max_tiles: raise RuntimeError(f'only {len(candidates)} valid tiles; need {a.max_tiles}')
    # Deterministically cover the full slide: rank x/y grid order then take evenly spaced candidates.
    candidates = candidates.sort_values(['y', 'x']).reset_index(drop=True)
    positions = np.linspace(0, len(candidates) - 1, a.max_tiles).round().astype(int)
    selected = candidates.iloc[np.unique(positions)].copy()
    if len(selected) != a.max_tiles: raise RuntimeError('tile selection unexpectedly deduplicated')
    out.mkdir(parents=True); selected.to_csv(out / 'tile_manifest.csv', index=False)
    result = {'purpose': 'deterministic non-overlapping full-slide H&E tile selection from derived CosMx RNA index', 'source_index': str(a.index_dir), 'he_size': [width, height], 'tile_size': a.tile_size, 'patch_size': a.patch_size, 'candidates': int(len(candidates)), 'selected_tiles': int(len(selected)), 'expected_patches': int(len(selected) * side * side), 'selected_rna_cells': int(selected.rna_cells.sum()), 'minimum_patch_cells': int(selected.minimum_patch_cells.min()), 'acceptance': 'all selected tiles non-overlapping grid cells and every constituent patch has at least requested RNA-cell count'}
    (out / 'metadata.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); print(json.dumps({'event': 'complete', 'output': str(out), **result}, ensure_ascii=False))


if __name__ == '__main__': main()
