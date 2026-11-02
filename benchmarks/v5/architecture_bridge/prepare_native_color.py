"""Fixed-mask native XCell color inputs; no encoder, detector or tissue GT run."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import pyvips
import numpy as np
from PIL import Image

from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha
from benchmarks.v5.multidomain_data.fast_nuclei import CachedCenterCropper
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import cells_from_instances, attended_crop
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features

CONDITIONS = dict(identity=None, OD_RG_plus=[1.1, .9, 1.], OD_RG_minus=[.9, 1.1, 1.],
                  OD_light=[.9, .9, .9], OD_dark=[1.1, 1.1, 1.1])


def priority(tile):
    return hashlib.sha256(('P2nativeColor20261103|' + tile).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', action='store_true')
    p.add_argument('--accept', action='store_true')
    a = p.parse_args()
    catalog, rows = load_cell_catalog(TASKS)
    selected = []
    for dataset in ['crc', 'tiger']:
        for split in ['train', 'val']:
            subset = sorted([k for k, r in rows.items() if r['dataset'] == dataset and r['split'] == split], key=priority)
            selected += subset[:1 if a.gate else 4]
    identity = dict(catalog_sha256=sha(catalog / 'tiles.parquet'), source_sha256=sha(__file__),
                    native_geometry_source_sha256=sha(inspect.getfile(cells_from_instances)),
                    cropper_source_sha256=sha(inspect.getfile(CachedCenterCropper)), gate=a.gate)
    if a.accept:
        report = json.loads((a.output / 'report.json').read_text())
        assert report['identity'] == identity
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for tile in selected:
        row = rows[tile]; variant = json.loads(row['variants_json'])['original']
        assert sha(variant['image']) == variant['image_sha256']
        assert sha(variant['cells']) == variant['cells_sha256']
        with Image.open(variant['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        mx, my = json.loads(row['input_mpp_xy_json'])
        size = (round(rgb.shape[1] * mx / .25), round(rgb.shape[0] * my / .25))
        high = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)).copy()
        with np.load(variant['cells'], allow_pickle=False) as f:
            inst, meta_old, reg, features, total_old = [f[k].copy() for k in ['instances', 'meta', 'reg', 'features', 'total']]
        assert inst.shape == high.shape[:2]
        meta, ids, total = cells_from_instances(inst)
        if len(ids):
            area = np.bincount(inst.ravel())[ids].astype(np.float64)
            actual_mpp = [rgb.shape[1] * mx / size[0], rgb.shape[0] * my / size[1]]
            meta[:, 2] = np.log1p(area * actual_mpp[0] * actual_mpp[1] / (.25 * .25))
        assert np.array_equal(meta, meta_old) and total == int(total_old)
        assert np.array_equal(encode_xcell_features(meta, reg), features)
        geometry = a.output / (tile + '_geometry.npz')
        if a.accept:
            with np.load(geometry, allow_pickle=False) as f:
                for key, value in dict(instances=inst, instance_ids=ids, features=features).items():
                    assert np.array_equal(f[key], value)
        else:
            cropper = CachedCenterCropper()
            for label in ids[[0, len(ids) // 2, -1]] if len(ids) else []:
                assert np.array_equal(cropper(high, inst, int(label)), attended_crop(high, inst, int(label)))
            np.savez_compressed(geometry, instances=inst, instance_ids=ids, features=features)
        variants = []
        for condition, factors in CONDITIONS.items():
            path = a.output / (tile + '_' + condition + '.png')
            if a.accept:
                with Image.open(path) as im:
                    value = np.asarray(im.convert('RGB')).copy()
                independent = high.copy()
                if factors is not None:
                    for c, factor in enumerate(factors):
                        independent[:, :, c] = np.rint(256 * ((high[:, :, c].astype(float) + 1) / 256) ** factor - 1).clip(0, 255).astype(np.uint8)
                assert np.array_equal(value, independent)
            else:
                value = high if factors is None else np.rint(256 * np.exp(np.log((high.astype(np.float64) + 1) / 256) * np.asarray(factors)) - 1).clip(0, 255).astype(np.uint8)
                Image.fromarray(value).save(path)
            variants.append(dict(condition=condition, path=str(path), file_sha256=sha(path),
                                 pixel_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                                 changed_channels=int(np.count_nonzero(value != high))))
        records.append(dict(tile_id=tile, dataset=row['dataset'], split=row['split'],
                            patient_id=row['patient_id'], space_key=row['space_key'],
                            priority=priority(tile), geometry=str(geometry), geometry_sha256=sha(geometry),
                            source_rgb_sha256=variant['image_sha256'], source_cells_sha256=variant['cells_sha256'],
                            retained_cells=len(ids), detected_before_cap=total, variants=variants))
        print(json.dumps(dict(frames=len(records), total=len(selected), accept=a.accept)), flush=True)
    if a.accept:
        assert report['records'] == records
        result = dict(status='accepted_fixed_mask_native_color_input_gate' if a.gate else 'accepted_fixed_mask_native_color_inputs',
                      report_sha256=sha(a.output / 'report.json'), worker_pid=os.getpid(),
                      frames=len(records), retained_cells=sum(r['retained_cells'] for r in records),
                      feature_identity_accepted=False, color_code_stability_accepted=False)
        (a.output / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        result = dict(status='native_color_inputs_unaccepted', identity=identity, records=records,
                      conditions=CONDITIONS, worker_pid=os.getpid(), labels_read=False, detector_rerun=False,
                      scope='Fine views only; fixed detection/geometry/context; color applied after physical resampling before attended crop',
                      limitations=['synthetic RGB OD factors, not real stain transfer', 'no detector robustness claim',
                                   'original feature identity still requires native encoder gate'])
        (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
