#!/usr/bin/env python3
"""Read-only source/feature and marker-rule audit for Panel1 matched cells."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, label


MARKERS = ('CKPAN', 'CD31', 'D2-40', 'FAP', 'SMA', 'S100')
SOURCE_COLUMNS = (['cell_id', 'x', 'y', 'base_cell_type', 'cell_type']
                  + [f'{m}_positive' for m in MARKERS]
                  + [f'{m}_intensity' for m in MARKERS])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_label(row) -> str:
    if row['CKPAN_positive']:
        kind = 'Epithelial'
    elif row['CD31_positive'] and row['D2-40_positive']:
        kind = 'Lymphatic_endothelial'
    elif row['CD31_positive']:
        kind = 'Blood_endothelial'
    elif row['D2-40_positive']:
        kind = 'D2-40_positive'
    elif row['FAP_positive'] and row['SMA_positive']:
        kind = 'FAP_SMA_CAF'
    elif row['FAP_positive']:
        kind = 'FAP_CAF'
    elif row['SMA_positive']:
        kind = 'SMA_stromal'
    else:
        kind = 'Undefined'
    return kind


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-patches', type=int, default=0)
    p.add_argument('--skip-source', action='store_true', help='Audit cached rows only when source NFS is unavailable')
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('output outside FewClick artifact root')
    a.output.mkdir(parents=True, exist_ok=False)
    method = ('cached feature/label/coordinate checks only' if a.skip_source
              else 'source hash, marker intensity/cutoff, label-rule, feature/coordinate checks')
    report = {'status': 'running', 'cases': {}, 'method': method + '; shifted control is risk diagnostic only',
              'source_tables_checked': not a.skip_source}
    errors = []
    for root in a.roots:
        meta = json.loads((root / 'audit.json').read_text())
        case = meta['case_id']
        if case in report['cases'] or meta['classes'] != CLASSES:
            raise ValueError('duplicate case or class vocabulary mismatch')
        patches = pd.read_csv(root / 'patch_audit.csv')
        complete = patches.loc[patches.status.eq('complete')]
        cutoffs = None
        if not a.skip_source:
            source_dir = Path(meta['source_annotation'])
            summaries = list(source_dir.glob('*_summary.json'))
            if len(summaries) != 1:
                raise ValueError(f'expected one marker summary for {case}')
            cutoffs = json.loads(summaries[0].read_text())['positive_cutoffs_minDisplay']
        if a.max_patches:
            complete = complete.head(a.max_patches)
        counts = Counter()
        by_class = Counter()
        distances = []
        rows = []
        case_errors = []
        for ordinal, (_, row) in enumerate(complete.iterrows(), 1):
            patch = row['patch']
            npz = root / 'patches' / f'{patch}.npz'
            table = Path(row['source_table'])
            try:
                with np.load(npz, allow_pickle=False) as raw:
                    z = {name: raw[name].copy() for name in raw.files}
                n = len(z['labels'])
                if z['features'].shape != (n, 768) or not np.isfinite(z['features']).all():
                    raise ValueError('feature shape/nonfinite')
                if any(len(z[k]) != n for k in ('he_ids','he_xy','mif_ids','mif_global_xy','base_cell_types','match_distance_px')):
                    raise ValueError('row count mismatch')
                if len(np.unique(z['he_ids'])) != n or len(np.unique(z['mif_ids'])) != n:
                    raise ValueError('duplicate ID within patch')
                mapped = np.array([label(x) for x in z['base_cell_types']], dtype=np.int8)
                if not np.array_equal(mapped, z['labels']) or not np.all((0 <= mapped) & (mapped < len(CLASSES))):
                    raise ValueError('five-class mapping mismatch')
                if not a.skip_source:
                    if sha256(table) != str(z['source_table_sha256']):
                        raise ValueError('source table checksum mismatch')
                    source = pd.read_csv(table, sep='\t', usecols=SOURCE_COLUMNS).set_index('cell_id', verify_integrity=True)
                    selected = source.loc[z['mif_ids']]
                    if not np.allclose(selected[['x','y']].to_numpy(float), z['mif_global_xy'], atol=.01):
                        raise ValueError('source mIF coordinates differ')
                    if not np.array_equal(selected.base_cell_type.astype(str).to_numpy(), z['base_cell_types']):
                        raise ValueError('source labels differ')
                    for marker in MARKERS:
                        intensity = selected[f'{marker}_intensity'].to_numpy(float)
                        positive = selected[f'{marker}_positive'].to_numpy()
                        if not np.isfinite(intensity).all() or not np.array_equal(
                                intensity >= cutoffs[marker], positive.astype(bool)):
                            raise ValueError(f'{marker} positivity differs from numeric cutoff')
                    predicted = selected.apply(expected_label, axis=1).to_numpy(str)
                    if not np.array_equal(predicted, z['base_cell_types']):
                        raise ValueError('source marker hierarchy differs from base class')
                    overlay = np.where(selected['S100_positive'].to_numpy(bool),
                        np.where(predicted == 'Undefined', 'S100_positive', np.char.add(predicted, '_S100')),
                        predicted)
                    if not np.array_equal(overlay, selected['cell_type'].astype(str).to_numpy()):
                        raise ValueError('source S100 overlay differs from cell_type')
                patch_xy = np.array([row['x'],row['y']], float)
                delta = np.linalg.norm(z['he_xy'] - (z['mif_global_xy'] - patch_xy), axis=1)
                # Global mIF coordinates were cached as float32; at WSI coordinates
                # this introduces a few thousandths of a pixel of rounding.
                if not np.allclose(delta, z['match_distance_px'], atol=.01) or np.any(delta > 4.01):
                    raise ValueError('HE-mIF distance mismatch')
                if np.any((z['he_xy'] < 64) | (z['he_xy'] > 447)):
                    raise ValueError('unsafe HE crop')
                if n != int(row['eligible_safe_crop']) or np.bincount(mapped, minlength=5).tolist() != ast.literal_eval(row['class_counts']):
                    raise ValueError('patch audit count mismatch')
                actual = int(row['all_label_mutual_le4'])
                null = np.asarray(ast.literal_eval(row['shift_null_counts']), dtype=float)
                if null.shape != (4,) or actual < n:
                    raise ValueError('invalid shifted-control audit')
                null_mean = float(null.mean())
                advantage = actual / null_mean if null_mean else None
                counts['cells'] += n
                counts['patches'] += 1
                counts['candidate_patches_real_le_2x_null'] += int(null_mean > 0 and actual <= 2 * null_mean)
                counts['candidate_cells_real_le_2x_null'] += n * int(null_mean > 0 and actual <= 2 * null_mean)
                by_class.update(map(int, mapped))
                distances.extend(delta.tolist())
                rows.append({'case_id':case,'patch':patch,'candidate_cells':n,'all_type_matches':actual,
                             'shift_null_mean':null_mean,'real_to_shift_ratio':advantage,
                             'class_counts':row['class_counts']})
            except Exception as exc:
                case_errors.append({'case_id':case,'patch':patch,'error':str(exc)})
            if ordinal % 500 == 0:
                print(json.dumps({'case':case,'processed':ordinal,'total':len(complete),'errors':len(case_errors)}),flush=True)
        errors.extend(case_errors)
        d=np.asarray(distances)
        report['cases'][case]={'audited_patches':len(complete),'valid_patches':counts['patches'],
            'valid_cells':counts['cells'],'class_counts':{CLASSES[i]:by_class[i] for i in range(5)},
            'distance_px_quantiles':np.quantile(d,[0,.25,.5,.75,.9,.99,1]).round(4).tolist() if len(d) else [],
            'real_le_2x_shift_null_patches':counts['candidate_patches_real_le_2x_null'],
            'real_le_2x_shift_null_cells':counts['candidate_cells_real_le_2x_null'],
            'source_declared_status':meta['label_status'],
            'numeric_marker_cutoffs_checked':not a.skip_source,'errors':len(case_errors)}
        pd.DataFrame(rows).to_csv(a.output / f'{case}_patch_risk.csv', index=False)
        (a.output / 'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'case':case,**report['cases'][case]},ensure_ascii=False),flush=True)
    report['status']='complete' if not errors else 'failed_integrity'
    report['errors']=errors[:100]
    report['error_count']=len(errors)
    (a.output / 'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if errors:
        raise ValueError(f'{len(errors)} integrity failures; see metrics.json')


if __name__=='__main__':
    main()
