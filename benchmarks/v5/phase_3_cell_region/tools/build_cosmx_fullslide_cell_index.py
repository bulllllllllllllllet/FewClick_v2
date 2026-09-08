#!/usr/bin/env python3
"""Create a derived full-slide COAD CosMx cell/RNA index without copying source data."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyvips

from build_cosmx_region_rna_target_smoke import PROGRAMS, he_from_global


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/nfs-medical3/fyx/PKU_DATA/CosMx/COAD'))
    p.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache'))
    p.add_argument('--timestamp', required=True); return p.parse_args()


def polygon_summary(path: Path) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    for chunk in pd.read_csv(path, usecols=['fov', 'cellID', 'x_global_px', 'y_global_px'], chunksize=500_000):
        grouped = chunk.groupby(['cellID', 'fov'], sort=False).agg(x_sum=('x_global_px', 'sum'), y_sum=('y_global_px', 'sum'), x_min=('x_global_px', 'min'), y_min=('y_global_px', 'min'), x_max=('x_global_px', 'max'), y_max=('y_global_px', 'max'), vertices=('x_global_px', 'size'))
        for (cell, fov), row in grouped.iterrows():
            key = f'{int(cell)}_{int(fov)}'; item = row.to_numpy(dtype=np.float64)
            if key in values:
                old = values[key]; old[0:2] += item[0:2]; old[2:4] = np.minimum(old[2:4], item[2:4]); old[4:6] = np.maximum(old[4:6], item[4:6]); old[6] += item[6]
            else: values[key] = item
    rows = [{'cell_id': key, 'x_global': item[0] / item[6], 'y_global': item[1] / item[6], 'x_min_global': item[2], 'y_min_global': item[3], 'x_max_global': item[4], 'y_max_global': item[5], 'vertices': int(item[6])} for key, item in values.items()]
    return pd.DataFrame(rows)


def add_counts(path: Path, ids: set[str]) -> tuple[dict[str, int], dict[str, np.ndarray], set[str]]:
    totals: dict[str, int] = defaultdict(int); programs: dict[str, np.ndarray] = {}
    marker_to_programs: dict[str, list[int]] = defaultdict(list)
    for index, markers in enumerate(PROGRAMS.values()):
        for marker in markers: marker_to_programs[marker].append(index)
    seen: set[str] = set(); source = ds.dataset(path, format='parquet')
    for batch in source.scanner(columns=['fov', 'cell_ID', 'target'], batch_size=262_144).to_batches():
        frame = batch.to_pandas(); frame['cell_id'] = frame.cell_ID.astype(str) + '_' + frame.fov.astype(str); frame = frame[frame.cell_id.isin(ids)]
        if frame.empty: continue
        for key, count in frame.groupby('cell_id', sort=False).size().items(): totals[key] += int(count)
        observed = set(frame.target.unique()); seen.update(observed)
        for program_index, markers in enumerate(PROGRAMS.values()):
            part = frame[frame.target.isin(markers)]
            for key, count in part.groupby('cell_id', sort=False).size().items():
                if key not in programs: programs[key] = np.zeros(len(PROGRAMS), np.int64)
                programs[key][program_index] += int(count)
    return totals, programs, seen


def main():
    a = parse_args(); out = a.output_root / f'cosmx_fullslide_cell_index_{a.timestamp}'
    if out.exists(): raise FileExistsError(out)
    polygons = polygon_summary(a.data_root / 'segmentation_mask/polygons.csv')
    image = pyvips.Image.new_from_file(str(a.data_root / 'HE.tif'), access='sequential')
    hx, hy = he_from_global(polygons.x_global.to_numpy(), polygons.y_global.to_numpy(), image.width, image.height)
    polygons['he_x'], polygons['he_y'] = hx, hy
    polygons['he_area_bbox'] = (polygons.x_max_global - polygons.x_min_global) / (60215 / image.width) * (polygons.y_max_global - polygons.y_min_global) / (85107 / image.height)
    totals, counts, seen = add_counts(a.data_root / 'transcripts/tx_file.parquet', set(polygons.cell_id))
    polygons['transcripts'] = polygons.cell_id.map(totals).fillna(0).astype(np.int64)
    matrix = np.stack([counts.get(key, np.zeros(len(PROGRAMS), np.int64)) for key in polygons.cell_id])
    denominator = polygons.transcripts.to_numpy().clip(1)[:, None]
    score = np.log1p(matrix / denominator * 1e4).astype(np.float32)
    for index, name in enumerate(PROGRAMS): polygons[f'program_{name}'] = score[:, index]
    valid = polygons.transcripts.gt(0)
    in_bounds = (polygons.he_x >= 0) & (polygons.he_x < image.width) & (polygons.he_y >= 0) & (polygons.he_y < image.height)
    result = polygons.loc[valid].copy()
    if result.cell_id.duplicated().any() or not np.isfinite(result.select_dtypes('number').to_numpy()).all(): raise RuntimeError('non-unique or non-finite derived index')
    out.mkdir(parents=True); result.to_csv(out / 'cell_rna_index.csv.gz', index=False, compression='gzip')
    meta = {'purpose': 'derived full-slide CosMx per-cell geometry and semantic RNA program index; RNA remains a training target only', 'he_size': [image.width, image.height], 'polygon_cells': int(len(polygons)), 'cells_with_transcripts': int(valid.sum()), 'cell_id_unique': bool(not result.cell_id.duplicated().any()), 'center_in_he_fraction_all_polygons': float(in_bounds.mean()), 'program_names': list(PROGRAMS), 'marker_coverage': {name: sorted(set(markers) & seen) for name, markers in PROGRAMS.items()}, 'acceptance': {'all_finite': True, 'all_retained_cells_inside_he': bool(in_bounds[valid].all())}}
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(out), **meta}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
