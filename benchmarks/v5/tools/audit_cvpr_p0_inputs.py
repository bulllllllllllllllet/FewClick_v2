"""Bounded real-input inventory, without granting cell registration acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import pyvips
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    source = Path('/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_161850')
    frame = pd.read_csv(source / 'sampled_cells.csv')
    report = dict(worker_pid=os.getpid(), purpose=__doc__, source_manifest=str(source),
                  source_sha256=digest(source / 'sampled_cells.csv'), cohorts=[], cells=[],
                  source_code_sha256=digest(__file__), GPU_used=False)
    # Hash ordering depends only on provenance/identity, never model predictions.
    frame['selection_key'] = [hashlib.sha256(f'{c}|{i}'.encode()).hexdigest()
                              for c, i in zip(frame.cohort_id, frame.cell_id)]
    for cohort, group in frame.groupby('cohort_id', sort=True):
        sample = group.sort_values('selection_key').head(3)
        image = pyvips.Image.new_from_file(str(group.he_path.iloc[0]), access='random')
        with h5py.File(group.h5ad_path.iloc[0], 'r') as handle:
            obs = handle['obs']
            ids = obs[obs.attrs['_index']].asstr()[:]
            lookup = {s: i for i, s in enumerate(ids)}
            annotation = obs['annotation']
            categories = annotation['categories'].asstr()[:]
            codes = annotation['codes'][:]
            resolutions = {}
            for key in handle.get('uns', {}):
                node = handle['uns'][key]
                if 'resolution' in key.lower() and isinstance(node, h5py.Dataset):
                    resolutions[key] = str(node[()])
            for row in sample.itertuples(index=False):
                idx = lookup[str(row.cell_id)]
                assert codes[idx] >= 0 and categories[codes[idx]] == row.raw_annotation
                left, top = round(row.x - 64), round(row.y - 64)
                assert left >= 0 and top >= 0 and left + 128 <= image.width and top + 128 <= image.height
                crop = image.crop(left, top, 128, 128)
                assert crop.format == 'uchar' and crop.bands >= 3
                pixels = np.frombuffer(crop.write_to_memory(), np.uint8).reshape(128, 128, crop.bands)[:, :, :3].copy()
                target = args.output / f'{cohort}_{row.cell_id}.png'
                Image.fromarray(pixels).save(target)
                report['cells'].append(dict(cohort_id=cohort, cell_id=row.cell_id,
                    raw_annotation=row.raw_annotation, broad_annotation=row.broad_annotation,
                    annotation_id_match=True, original_coordinate_status=row.coordinate_status,
                    center_xy=[row.x, row.y], crop_box_pixels=[left, top, 128, 128],
                    he_path=row.he_path, label_source='transcriptome obs annotation, not tissue GT',
                    crop=str(target), crop_sha256=digest(target), pixel_std=float(pixels.std()),
                    split='unassigned_audit_only', registration_accepted=False,
                    physical_crop_um=None, physical_scale_status='needs_units_and_transform_evidence'))
        report['cohorts'].append(dict(cohort_id=cohort, historical_sample_count=len(group),
            source_label_count=len(ids), image_shape=[image.height, image.width, image.bands],
            tiff_xres=image.xres, tiff_yres=image.yres,
            tiff_description=str(image.get('image-description')) if image.get_typeof('image-description') else None,
            h5ad_resolution_metadata=resolutions, sampled_real_crops=len(sample),
            coordinate_warning='inside-raster and finite-feature checks do not prove cell-level registration'))

    from benchmarks.v5.fair_comparison.local_data import LocalData
    tasks = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
    data = LocalData(tasks)
    report['tissue_catalog'] = dict(path=str(data.catalog), sha256=digest(data.catalog/'tiles.parquet'),
        counts=pd.DataFrame(data.rows.values()).groupby(['dataset','split']).size().to_dict().__str__(),
        exposure='historical train/dev; not unexposed confirmation')
    report['tissue_inputs'] = []
    rows = pd.DataFrame(data.rows.values()).sort_values('tile_id')
    for (dataset, split), group in rows[rows.dataset.isin(['crc','tiger'])].groupby(['dataset','split']):
        row = group.iloc[0].to_dict()
        raw, gt = data.raw(row['tile_id'], contexts=False, cells=False)
        assert tuple(raw['rgb'].shape) == (1, 3, 512, 512)
        report['tissue_inputs'].append(dict(tile_id=row['tile_id'], dataset=dataset, split=split,
            source_id=row['source_id'], patient_id=str(row['patient_id']), space_key=row['space_key'],
            mpp_xy=json.loads(row['input_mpp_xy_json']), labels=np.unique(gt).tolist(),
            actual_loader_hash_checks_passed=True, full_input_footprint_accepted=False))
    weights = Path('/nfs-medical3/zyh/fewclick_v2/model_weights')
    report['weight_inventory'] = {name: [dict(file=str(p), bytes=p.stat().st_size) for p in sorted((weights/name).glob('*')) if p.is_file()]
                                  for name in ['phikon-v2','kaiko_vit_large_patch14_reg4','LEMON']}
    report['status'] = 'complete_bounded_inventory_P0_unaccepted'
    report['remaining'] = ['cell-level registration and coordinate units','physical crop contract',
        'spatial-block split and maximum-context footprint','candidate weight identity and pretraining exposure',
        'unexposed confirmation domain','matched local/remote task and budget contract']
    report['elapsed_seconds'] = time.time()-start
    (args.output/'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], worker_pid=os.getpid(), cells=len(report['cells']),
                         tissue_inputs=len(report['tissue_inputs']), seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
