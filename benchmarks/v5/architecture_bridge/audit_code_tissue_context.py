"""Offline tissue-context coverage of frozen codes, never an inference input."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--census', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', action='store_true')
    p.add_argument('--accept', action='store_true')
    a = p.parse_args()
    catalog, inventory = load_cell_catalog(TASKS)
    proof = json.loads((a.census / 'independent_acceptance.json').read_text())
    assert proof['status'] == 'accepted_retained_cell_census'
    assert proof['report_sha256'] == sha(a.census / 'report.json')
    census = json.loads((a.census / 'report.json').read_text())
    assert census['identity']['catalog_sha256'] == sha(catalog / 'tiles.parquet')
    indexed = list(enumerate(census['frames']))
    assert len(indexed) == 1737
    if a.gate:
        ids = {min(k for k, r in inventory.items() if r['dataset'] == d and r['split'] == s)
               for d in ['crc', 'tiger'] for s in ['train', 'val']}
        indexed = [(i, r) for i, r in indexed if r['tile_id'] in ids]
        assert len(indexed) == 4
    identity = dict(census_report_sha256=proof['report_sha256'], source_sha256=sha(__file__),
                    catalog_sha256=sha(catalog / 'tiles.parquet'), gate=a.gate)
    if a.accept:
        prior = json.loads((a.output / 'report.json').read_text())
        assert prior['identity'] == identity
    else:
        a.output.mkdir(parents=True, exist_ok=False); (a.output / 'frames').mkdir()
    histograms = defaultdict(lambda: np.zeros(128, dtype=np.int64))
    cells = defaultdict(int); pixels = defaultdict(int); images = defaultdict(set)
    cell_images = defaultdict(set); spaces = defaultdict(set); error_sums = defaultdict(float)
    small_images = defaultdict(int)
    frames = []; total = ignored = changed_labels = 0
    for index, frame in indexed:
        tile = frame['tile_id']; row = inventory[tile]; variant = json.loads(row['variants_json'])['original']
        codefile = a.census / 'frames' / f'{index:04d}.npz'
        assert sha(codefile) == frame['file_sha256']
        assert sha(variant['cells']) == variant['cells_sha256']
        with np.load(codefile, allow_pickle=False) as f:
            codes, errors = f['codes_0'].copy(), f['errors_0'].copy()
        with np.load(variant['cells'], allow_pickle=False) as f:
            instances, meta = f['instances'].copy(), f['meta'].copy()
        assert codes.shape == errors.shape == (len(meta),) and len(meta) == frame['retained_per_scale'][0]
        ids = np.unique(instances); ids = ids[ids > 0][:255]
        yx = np.asarray(ndimage.center_of_mass(np.ones_like(instances, dtype=np.uint8), instances, ids)).reshape(-1, 2) if len(ids) else np.empty((0, 2))
        order = np.lexsort((yx[:, 1], yx[:, 0])); ids, yx = ids[order], yx[order]
        expected_xy = (yx[:, ::-1] / np.array(instances.shape[::-1])).astype(np.float32)
        assert np.array_equal(expected_xy, meta[:, :2]) and len(ids) == len(meta)
        with Image.open(row['labels']) as im:
            gt = np.asarray(im).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest() == row['labels_array_sha256']
        # Pixel-center physical bins, correcting the legacy metadata's omitted half pixel.
        projected = np.floor((yx + .5) / np.array(instances.shape) * np.array(gt.shape)).astype(int)
        label = gt[projected[:, 0], projected[:, 1]]
        old_xy = np.floor(meta[:, :2].astype(np.float64) * np.array(gt.shape[::-1])).astype(int)
        legacy_label = gt[old_xy[:, 1], old_xy[:, 0]]
        changed_labels += int((label != legacy_label).sum())
        if a.accept and len(ids):
            boxes = ndimage.find_objects(instances)
            for j in sorted({0, len(ids)//2, len(ids)-1}):
                sy, sx = boxes[int(ids[j])-1]
                yy, xx = np.nonzero(instances[sy, sx] == ids[j])
                center = np.array([yy.mean() + sy.start, xx.mean() + sx.start])
                assert np.allclose(center, yx[j], atol=1e-10, rtol=0)
        out = a.output / 'frames' / f'{index:04d}.npz'
        arrays = dict(instance_ids=ids, tissue_context=label, legacy_tissue_context=legacy_label)
        if a.accept:
            with np.load(out, allow_pickle=False) as f:
                assert all(np.array_equal(f[k], v) for k, v in arrays.items())
        else:
            np.savez_compressed(out, **arrays)
        valid = label != 255; ignored += int((~valid).sum()); total += len(label)
        classes, counts = np.unique(gt, return_counts=True)
        for target, n_pixels in zip(classes, counts):
            if target == 255:
                continue
            key = f'{row["dataset"]}|{row["split"]}|{int(target)}'
            select = label == target; count = int(select.sum())
            pixels[key] += int(n_pixels); images[key].add(tile)
            cells[key] += count; histograms[key] += np.bincount(codes[select], minlength=128)
            error_sums[key] += float(errors[select].sum())
            small_images[key] += int(n_pixels / max(1, int((gt != 255).sum())) < .05)
            if count:
                cell_images[key].add(tile); spaces[key].add(row['space_key'])
        frames.append(dict(tile_id=tile, cells=len(ids), ignored_context_cells=int((~valid).sum()),
                           labels_sha256=row['labels_array_sha256'], file_sha256=sha(out)))
        if len(frames) % 100 == 0:
            print(json.dumps(dict(frames=len(frames), total=len(indexed), accept=a.accept)), flush=True)
    groups = {}
    for key, hist in sorted(histograms.items()):
        n = cells[key]; assert int(hist.sum()) == n
        p_used = hist[hist > 0] / n if n else np.array([])
        groups[key] = dict(raw_label_id=int(key.split('|')[-1]), pixels=pixels[key], frames_with_pixels=len(images[key]),
                           frames_with_center_assigned_cells=len(cell_images[key]), cell_space_groups=len(spaces[key]),
                           small_area_frames=small_images[key], retained_center_assigned_cells=n,
                           code_counts=hist.tolist(), occupied_codes=int((hist > 0).sum()),
                           effective_codes=float(np.exp(-(p_used*np.log(p_used)).sum())) if n else None,
                           quantization_per_dimension_MSE=error_sums[key] / n / 64 if n else None)
    assert sum(v['retained_center_assigned_cells'] for v in groups.values()) + ignored == total
    if not a.gate:
        assert len(frames) == 1737 and total == 304747
    if a.accept:
        assert prior['groups'] == groups and prior['frames'] == frames
        assert prior['total_fine_cells'] == total and prior['ignored_context_cells'] == ignored
        assert prior['half_pixel_context_label_changes'] == changed_labels
        (a.output / 'independent_acceptance.json').write_text(json.dumps(dict(
            status='accepted_offline_tissue_context_gate' if a.gate else 'accepted_offline_tissue_context_diagnostics',
            worker_pid=os.getpid(), frames=len(frames), fine_cells=total,
            report_sha256=sha(a.output / 'report.json'), diagnostic_only=True, P2_accepted=False), indent=2) + '\n')
    else:
        report = dict(status='offline_tissue_context_unaccepted', identity=identity, groups=groups, frames=frames,
                      total_fine_cells=total, ignored_context_cells=ignored, half_pixel_context_label_changes=changed_labels,
                      offline_GT_read=True, inference_or_fit_changed=False, P2_accepted=False,
                      limitations=['tissue pixel at a nucleus center is contextual, not a cell-type label',
                                   'fixed retained/capped cells; sparse-nucleus tissue coverage may be low',
                                   'raw background IDs retained; only 255 ignored',
                                   'overlapping views and same-source cells are dependent; not WSI area or independent samples'])
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
