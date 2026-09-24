#!/usr/bin/env python3
"""Extract geometrically matched candidate H&E cell features from Panel1 registered WSI and mIF patches."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import time
from pathlib import Path

from cellpose import models
import numpy as np
import openslide
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import crop_mask, crop_region, soft_attention
from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, PREPROCESS, WEIGHTS, label, load_ctranspath


ROOT = Path('/nfs-medical6/data/SRRSH/crc/panel_1/registration_outputs')
CASES = {
    '2017-23800D': ('17-23800D_FAP', Path('/nfs-medical5/data/SRRSH/crc/panel_1/17-23800D_cell_annotation')),
    '2019-00538F': ('17-00538F_FAP', Path('/nfs-medical6/data/SRRSH/crc/panel_1/CRC_1_cell_annotation')),
    '2022-39941E': ('22-39941E_FAP', Path('/nfs-medical6/data/SRRSH/crc/panel_1/22-39941E_cell_annotation')),
}


def coords(stem):
    return tuple(map(int, stem.replace('x', '').replace('y', '').split('_')))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def candidates(folder, shard_index, shard_count):
    paths = []
    for path in (folder / 'cell_tables').glob('mif_x*_y*_cells.tsv'):
        stem = path.name.removeprefix('mif_').removesuffix('_cells.tsv')
        if int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16) % shard_count == shard_index:
            paths.append((stem, path))
    return sorted(paths, key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())


class PatchAudit(list):
    """Persist every outcome, including skips, before proceeding."""
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.started = time.monotonic()

    def append(self, row):
        super().append(row)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if len(self) % 20 == 0 or len(self) == 1:
            print(json.dumps({'event': 'progress', 'processed': len(self),
                'elapsed_seconds': round(time.monotonic() - self.started, 1),
                'last_patch': row['patch'], 'last_status': row['status']}), flush=True)


def extract(a):
    case_dir, annotation = CASES[a.case]
    source = ROOT / case_dir / 'large_context_sift_registration'
    slide_path = next(source.glob('*.svs'))
    paths = candidates(annotation, a.shard_index, a.shard_count)
    if not paths:
        raise ValueError('no source cell tables in shard')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'patches').mkdir()
    torch.set_num_threads(8)
    device = torch.device(a.device)
    detector = models.CellposeModel(gpu=device.type == 'cuda',
        pretrained_model='/home/zhaoyh/.cellpose/models/cpsam', device=device)
    encoder = load_ctranspath(WEIGHTS, device).requires_grad_(False) if a.encode else None
    slide = openslide.OpenSlide(str(slide_path))
    audit = PatchAudit(a.output / 'patch_audit.jsonl')
    print(json.dumps({'event': 'start', 'pid': os.getpid(), 'case': a.case,
        'source_patch_tables': len(paths), 'code_sha256': digest(Path(__file__))}), flush=True)
    total = Counter()
    for idx, (stem, table_path) in enumerate(paths, 1):
        if a.max_patches and idx > a.max_patches:
            break
        x, y = coords(stem)
        row = {'case_id': a.case, 'patch': stem, 'x': x, 'y': y,
               'status': 'pending', 'source_table': str(table_path)}
        try:
            frame = pd.read_csv(table_path, sep='\t', usecols=['cell_id', 'x', 'y', 'base_cell_type'])
        except pd.errors.EmptyDataError:
            row['status'] = 'empty_annotation'; audit.append(row); continue
        row['mif_all_cells'] = len(frame)
        frame['label'] = frame.base_cell_type.fillna('').map(label)
        row['mif_five_class_cells'] = int(frame.label.ge(0).sum())
        total['mif_all_cells'] += row['mif_all_cells']
        total['mif_five_class_cells'] += row['mif_five_class_cells']
        if not row['mif_five_class_cells']:
            row['status'] = 'no_five_class_cells'; audit.append(row); continue
        image = np.asarray(slide.read_region((x, y), 0, (512, 512)).convert('RGB'))
        row['he_std'] = float(image.std())
        if row['he_std'] < a.min_he_std:
            row['status'] = 'empty_he'; audit.append(row); continue
        mask = np.asarray(detector.eval(image, diameter=18, batch_size=1)[0], dtype=np.int32)
        ids = np.unique(mask); ids = ids[ids > 0]
        row['he_detected_cells'] = len(ids)
        total['he_detected_cells'] += len(ids)
        if not len(ids):
            row['status'] = 'no_he_nuclei'; audit.append(row); continue
        he = np.asarray(ndimage.center_of_mass(np.ones(mask.shape, np.uint8), mask, ids))[:, ::-1]
        mif = frame[['x', 'y']].to_numpy(float) - np.array([x, y])
        distance, nearest = cKDTree(mif).query(he)
        _, reverse = cKDTree(he).query(mif)
        mutual = reverse[nearest] == np.arange(len(he))
        accepted = mutual & (distance <= 4.)
        row['all_label_mutual_le4'] = int(accepted.sum())
        # Shifted-coordinate null counts are diagnostics, not an acceptance threshold.
        null_counts = []
        for shift in ((32, 0), (-32, 0), (0, 32), (0, -32)):
            shifted = (mif + np.asarray(shift)) % 512
            nd, ni = cKDTree(shifted).query(he)
            _, nr = cKDTree(he).query(shifted)
            null_counts.append(int(((nr[ni] == np.arange(len(he))) & (nd <= 4.)).sum()))
        row['shift_null_counts'] = null_counts
        row['registration_qc'] = 'candidate_only_pending_independent_registration_review'
        accepted &= frame.label.to_numpy()[nearest] >= 0
        row['mutual_le4'] = int(accepted.sum())
        total['mutual_le4'] += row['mutual_le4']
        accepted &= (he[:, 0] >= 64) & (he[:, 0] <= 447) & (he[:, 1] >= 64) & (he[:, 1] <= 447)
        row['eligible_safe_crop'] = int(accepted.sum())
        total['eligible_safe_crop'] += row['eligible_safe_crop']
        if not accepted.any():
            row['status'] = 'no_accepted_matches'; audit.append(row); continue
        selected = np.flatnonzero(accepted)
        chosen = frame.iloc[nearest[selected]].reset_index(drop=True)
        if a.encode:
            tensors = [PREPROCESS(soft_attention(
                crop_region(image, round(he[i, 1]), round(he[i, 0])),
                crop_mask(mask, round(he[i, 1]), round(he[i, 0])) == int(ids[i]))) for i in selected]
            with torch.inference_mode():
                feature = torch.cat([encoder(batch.to(device)).cpu()
                    for batch in torch.stack(tensors).split(32)]).numpy().astype(np.float32)
            if feature.shape != (len(selected), 768) or not np.isfinite(feature).all():
                raise ValueError(f'invalid feature tensor for {stem}')
        else:
            feature = np.empty((len(selected), 0), np.float32)
        np.savez_compressed(a.output / 'patches' / f'{stem}.npz',
            features=feature, he_ids=ids[selected], he_xy=he[selected].astype(np.float32),
            mif_ids=chosen.cell_id.to_numpy(np.int64),
            mif_global_xy=chosen[['x', 'y']].to_numpy(np.float32),
            labels=chosen.label.to_numpy(np.int8),
            base_cell_types=chosen.base_cell_type.to_numpy(dtype=str),
            match_distance_px=distance[selected].astype(np.float32),
            source_table_sha256=np.array(digest(table_path)))
        row['class_counts'] = np.bincount(chosen.label.to_numpy(int), minlength=5).tolist()
        row['status'] = 'complete'; audit.append(row)
        if idx % 20 == 0 or a.max_patches:
            print(json.dumps({'event': 'patch', 'case': a.case, 'processed': idx,
                'total': len(paths), 'accepted_cells': total['eligible_safe_crop']}), flush=True)
    slide.close()
    report = {'status': 'complete', 'case_id': a.case, 'classes': CLASSES,
        'code_sha256': digest(Path(__file__)), 'weights_sha256': digest(Path(WEIGHTS)) if a.encode else None,
        'label_status': 'candidate correspondence, not independently verified labels',
        'matching_rule': 'mutual nearest across ALL source types, distance <=4px; then select five classes',
        'registered_he_wsi': str(slide_path), 'source_annotation': str(annotation),
        'shard_index': a.shard_index, 'shard_count': a.shard_count,
        'source_patch_tables': len(paths), 'processed_patches': len(audit),
        'counts': dict(total), 'complete_feature_patches': sum(r['status'] == 'complete' for r in audit),
        'registered_source_patch_qc': 'global registered WSI; independent local transform QC unavailable for other cases',
        'feature_definition': '128px target HE nucleus soft attention, frozen CTransPath-768' if a.encode else 'matching audit only'}
    pd.DataFrame(audit).to_csv(a.output / 'patch_audit.csv', index=False)
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=tuple(CASES), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--max-patches', type=int, default=0)
    p.add_argument('--min-he-std', type=float, default=5.)
    p.add_argument('--encode', action='store_true')
    a = p.parse_args()
    if not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2'):
        p.error('output must be under FewClick artifact root')
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count or a.max_patches < 0:
        p.error('invalid shard or patch count')
    extract(a)


if __name__ == '__main__':
    main()
