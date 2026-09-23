#!/usr/bin/env python3
"""Add matched Panel1 cells from previously unused training spatial blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from benchmarks.v5.tools.run_crc_panel1_cell_identity import (
    CLASSES, SOURCE, digest, label, spatial_split, write_json,
)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bundle(a):
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.base/'manifest.json').read_text())
    cells = pd.read_csv(a.base/'cells.csv')
    with np.load(a.feature_source, allow_pickle=False) as z:
        features = z['features']
    if features.shape != (len(cells), 768) or not np.isfinite(features).all():
        raise ValueError('base feature cache does not match cell manifest')
    if hashlib.sha256((a.base/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('base manifest checksum mismatch')
    shutil.copyfile(a.base/'cells.csv', a.output/'cells.csv')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    np.savez_compressed(a.output/'features.npz', features=features)
    print(json.dumps({'status': 'complete', 'cells': len(cells), 'features': features.shape,
        'manifest_sha256': manifest['manifest_sha256']}), flush=True)


def prepare(a):
    base = pd.read_csv(a.base/'cells.csv')
    manifest = json.loads((a.base/'manifest.json').read_text())
    if hashlib.sha256((a.base/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('base manifest checksum mismatch')
    seen = {(round(r.mif_global_x, 3), round(r.mif_global_y, 3)) for r in base.itertuples()}
    old_train_groups = set(base.loc[base.split.eq('train'), 'group'])
    # Older manifests store patch records; keep the exact train/eval patch
    # identities and only revisit unused matched patches in train territory.
    old_patch_names = {p['patch'] if isinstance(p, dict) else p
                       for p in manifest.get('patches', [])}
    candidates = {}
    folder = SOURCE/'cell_matching/patch_matches'
    for path in folder.glob('*.csv.gz'):
        stem = path.name.removesuffix('.csv.gz')
        x, y = map(int, stem.replace('x', '').replace('y', '').split('_'))
        assigned = spatial_split(x, y)
        if assigned is None or assigned[0] != 'train' or stem in old_patch_names:
            continue
        candidates.setdefault(assigned[1], []).append(path)
    for group in candidates:
        candidates[group].sort(key=lambda p: digest('panel1-spatial-add-20260923/'+p.name))

    new_rows, patch_records, audit = [], [], []
    per_group, target_patches = {}, a.patches_per_group
    for group in sorted(candidates, key=lambda s: digest('panel1-group-20260923/'+s)):
        accepted_patches = 0
        for path in candidates[group]:
            if accepted_patches >= target_patches:
                break
            stem = path.name.removesuffix('.csv.gz')
            transform_path = SOURCE/'transforms'/f'{stem}.json'
            if not transform_path.is_file() or json.loads(transform_path.read_text()).get('status') != 'ok':
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            needed = {'accepted_match','base_cell_type','he_cell_id','he_x','he_y','he_aligned_x','he_aligned_y','mif_x','mif_y','x','y','cell_id'}
            if not needed.issubset(frame.columns):
                raise ValueError(f'incomplete matching schema: {path}')
            klass = frame.base_cell_type.map(label)
            distance = np.hypot(frame.he_aligned_x-frame.mif_x, frame.he_aligned_y-frame.mif_y)
            accepted = frame.accepted_match.astype(str).str.lower().eq('true')
            keep = accepted & (distance <= 4.) & (klass >= 0)
            keep &= frame.he_x.between(64, 447) & frame.he_y.between(64, 447)
            valid = frame.loc[keep].copy()
            valid['label'] = klass[keep].astype(int)
            valid['distance'] = distance[keep].astype(float)
            if valid.empty:
                continue
            chosen = []
            for cls in range(len(CLASSES)):
                subset = valid[valid.label.eq(cls)].sample(frac=1., random_state=20260923+cls)
                chosen.append(subset.head(a.cells_per_class_per_patch))
            selected = pd.concat(chosen, ignore_index=True).drop_duplicates('cell_id')
            if len(selected) < a.min_cells_per_patch:
                continue
            image_path = SOURCE/'registered_he_patches'/f'{stem}.png'
            mask_path = SOURCE/'cell_matching/masks'/f'{stem}.npy'
            if not image_path.is_file() or not mask_path.is_file():
                continue
            image = np.asarray(Image.open(image_path).convert('RGB'))
            mask = np.load(mask_path, allow_pickle=False)
            if image.shape != (512, 512, 3) or mask.shape != (512, 512):
                raise ValueError(f'unexpected geometry in {stem}')
            patch_rows = []
            for cell in selected.itertuples():
                key = (round(cell.x, 3), round(cell.y, 3))
                if key in seen:
                    continue
                nucleus = mask == int(cell.he_cell_id)
                if not nucleus.any():
                    continue
                cy, cx = ndimage.center_of_mass(nucleus)
                if np.hypot(cx-cell.he_x, cy-cell.he_y) > .1:
                    raise ValueError(f'nucleus centroid mismatch: {stem}/{cell.he_cell_id}')
                patch_rows.append(dict(patch=stem, split='train', group=group,
                    he_id=int(cell.he_cell_id), mif_id=int(cell.cell_id), he_x=float(cell.he_x), he_y=float(cell.he_y),
                    mif_global_x=float(cell.x), mif_global_y=float(cell.y), label=int(cell.label),
                    source_label=cell.base_cell_type, distance=float(cell.distance)))
            if len(patch_rows) < a.min_cells_per_patch:
                continue
            new_rows.extend(patch_rows)
            seen.update((round(r['mif_global_x'],3),round(r['mif_global_y'],3)) for r in patch_rows)
            patch_records.append({'patch': stem, 'split': 'train', 'group': group,
                'matches_sha256': file_hash(path), 'he_sha256': file_hash(image_path), 'mask_sha256': file_hash(mask_path)})
            audit.append({'patch': stem, 'group': group, 'added_cells': len(patch_rows),
                          'class_counts': np.bincount([r['label'] for r in patch_rows],minlength=5).tolist()})
            accepted_patches += 1
        per_group[group] = accepted_patches

    if not new_rows:
        raise ValueError('no new cells selected')
    combined = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
    if combined.duplicated(['mif_global_x','mif_global_y']).any():
        raise ValueError('duplicate global cell coordinates')
    old_eval = base[base.split.ne('train')][['patch','he_id','label','mif_global_x','mif_global_y','group','split']].sort_values(['patch','he_id']).reset_index(drop=True)
    new_eval = combined[combined.split.ne('train')][['patch','he_id','label','mif_global_x','mif_global_y','group','split']].sort_values(['patch','he_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(old_eval,new_eval,check_dtype=False)
    out = dict(manifest)
    out['manifest_sha256'] = None
    out['patches'] = list(manifest['patches']) + patch_records
    out['patch_counts'] = dict(manifest['patch_counts'])
    out['patch_counts']['train'] += len(patch_records)
    out['support'] = {s: np.bincount(combined.loc[combined.split.eq(s),'label'],minlength=5).tolist()
                      for s in ('train','val','test')}
    out['spatial_supplement'] = {'added_cells':len(new_rows),'added_patches':len(patch_records),
        'requested_patches_per_unused_group':target_patches,'old_train_groups':len(old_train_groups),
        'new_train_groups':int(combined.loc[combined.split.eq('train'),'group'].nunique()),
        'added_groups':len([k for k,v in per_group.items() if v]),'per_group_patch_counts':per_group,
        'new_support':np.bincount([r['label'] for r in new_rows],minlength=5).tolist(),
        'new_groups_per_class':[len({r['group'] for r in new_rows if r['label']==c}) for c in range(5)],
        'eval_identical':True}
    a.output.mkdir(parents=True,exist_ok=False)
    combined.to_csv(a.output/'cells.csv',index=False)
    out['manifest_sha256']=hashlib.sha256((a.output/'cells.csv').read_bytes()).hexdigest()
    write_json(a.output/'manifest.json',out)
    pd.DataFrame(audit).to_csv(a.output/'added_patch_audit.csv',index=False)
    print(json.dumps({'status':'prepared',**out['spatial_supplement'],'support':out['support']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['bundle','prepare'])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feature-source',type=Path)
    p.add_argument('--patches-per-group',type=int,default=4)
    p.add_argument('--cells-per-class-per-patch',type=int,default=24)
    p.add_argument('--min-cells-per-patch',type=int,default=5)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('timestamped FewClick artifact path required')
    if a.mode=='bundle':
        if a.feature_source is None:p.error('--feature-source required for bundle')
        bundle(a)
    else:
        prepare(a)


if __name__=='__main__':main()
