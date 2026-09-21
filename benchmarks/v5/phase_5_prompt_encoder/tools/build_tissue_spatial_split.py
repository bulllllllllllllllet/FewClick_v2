#!/usr/bin/env python3
"""Version a spatial split of frozen caches; never claim clean pretraining exposure."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_spatial_split import assign_context

ROOT = Path('/nfs-medical3/zyh/v4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp', required=True)
    args = p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', args.timestamp):
        raise ValueError('timestamp must be MMDD_HHMMSS')
    out = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt') / ('tissue_spatial_split_' + args.timestamp)
    out.mkdir(parents=True, exist_ok=False)
    paths = dict(patches=ROOT/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet',
                 cache=ROOT/'phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet',
                 labels=ROOT/'phase4/data/fine_region_labels_20260719_230128/label_index.parquet',
                 eligibility=ROOT/'phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet')
    tables = {k: pd.read_parquet(v) for k, v in paths.items()}
    patches = tables['patches']
    if patches.patch_id.duplicated().any():
        raise ValueError('duplicate patch IDs')
    source_keys = {v: str(Path(v).resolve()) for v in patches.wsi_path.unique()}
    records = []
    for row in patches.to_dict('records'):
        boxes = [tuple(row[f'{field}{suffix}_level0'] for field in ('x', 'y', 'width', 'height'))
                 for suffix in ('', '_5x', '_2p5x')]
        source = source_keys[row['wsi_path']]
        record = assign_context(source, boxes)
        if record:
            records.append(dict(patch_id=row['patch_id'], wsi_id=row['wsi_id'], patient_id=row['patient_id'],
                                source_key=source, historical_split=row['split'], **record))
    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError('no isolated patches')
    # Restrict all derivative indices to the identical available patch universe.
    available = set(manifest.patch_id)
    for table in tables.values():
        available &= set(table.patch_id)
    manifest = manifest[manifest.patch_id.isin(available)].sort_values('patch_id').reset_index(drop=True)
    if manifest.empty or set(manifest.split) != {'train', 'val', 'test'}:
        raise ValueError('incomplete split coverage')
    mapping = manifest.set_index('patch_id')['split']
    for name, table in tables.items():
        selected = table[table.patch_id.isin(available)].copy()
        if 'split' in selected:
            selected['historical_split'] = selected['split']
        selected['split'] = selected.patch_id.map(mapping)
        selected.to_parquet(out / f'{name}_index.parquet', index=False)
        tables[name] = selected
    manifest.to_parquet(out/'spatial_manifest.parquet', index=False)
    # Exhaustive block invariant implies no cross-split context intersection:
    # each source/block has one split and each union lies in its buffered interior.
    if manifest.groupby('spatial_group_id').split.nunique().max() != 1:
        raise AssertionError('cross-split group')
    eligibility = tables['eligibility']
    coverage = eligibility.groupby(['split', 'target_class', 'prompt_size']).size()
    missing = [(s, c, z) for s in ('train', 'val', 'test') for c in range(12)
               for z in ('point', 'small', 'large') if (s, c, z) not in coverage.index]
    report = dict(status='complete', coordinate_unit='level0_pixel', seed=20260922,
                  block_size=16384, margin_each_side=256, minimum_cross_block_gap=512,
                  case_grouping=False, input_patch_count=len(patches), retained_patches=len(manifest),
                  counts=manifest.groupby('split').size().to_dict(),
                  groups=manifest.groupby('split').spatial_group_id.nunique().to_dict(),
                  same_case_across_splits=int((manifest.groupby('patient_id').split.nunique() > 1).sum()),
                  missing_task_size_strata=missing,
                  coverage=[dict(split=s, target_class=int(c), prompt_size=z, episodes=int(n)) for (s,c,z),n in coverage.items()],
                  context_audit='union of fine, middle and coarse crop; buffered block containment',
                  pretrained_exposure='historical Phase2/3 encoders previously exposed to cohort; frozen-cache head comparison only',
                  independent_test=False, cross_rescan_registration='unresolved; canonical path aliases grouped, distinct rescans not verified',
                  scope='within-cohort spatial head-development comparison; no cross-patient or novel-class claims',
                  excluded_scope='no protein/RNA auxiliary targets or graph neighborhoods used; re-audit before adding them',
                  sources={k: str(v) for k,v in paths.items()},
                  source_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
                  manifest_sha256=hashlib.sha256((out/'spatial_manifest.parquet').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(out), **{k:report[k] for k in ('counts','groups','same_case_across_splits','missing_task_size_strata')})), flush=True)


if __name__ == '__main__':
    main()
