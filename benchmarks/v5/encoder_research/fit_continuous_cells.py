"""Fit train-only normalization/PCA64, then freeze two same-dimension cell grids."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.decomposition import PCA

from .cell_catalog import load_cell_catalog
from .cell_grids import cell_vectors, cell_grid
from .cache_continuous_cells import TASKS, array_sha
from .lemon_features import sha


def population_stats(values):
    mean = values.mean(0, dtype=np.float64)
    std = values.std(0, dtype=np.float64)
    std[std == 0] = 1
    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    receipt = json.loads((args.raw / 'independent_acceptance.json').read_text())
    assert receipt['status'] == ('accepted_raw16_continuous_cell_gate' if args.gate else 'accepted_raw1737_continuous_cells')
    if not args.gate:
        assert args.gate_receipt is not None
        gate = json.loads(args.gate_receipt.read_text())
        assert gate['status'] == 'accepted_train_only16_cell_grid_gate'
        assert gate['fitting_source_sha256'] == sha(__file__)
        assert gate['grid_source_sha256'] == sha(Path(__file__).with_name('cell_grids.py'))
        assert gate['acceptance_source_sha256'] == sha(Path(__file__).with_name('accept_cell_grids.py'))
    assert receipt['index_sha256'] == sha(args.raw / 'index.jsonl')
    catalog, inventory = load_cell_catalog(TASKS)
    entries = [json.loads(s) for s in (args.raw / 'index.jsonl').read_text().splitlines()]
    assert len(entries) == (16 if args.gate else 1737)
    if not args.gate:
        assert [r['tile_id'] for r in entries] == sorted(inventory)
    training = [r for r in entries if inventory[r['tile_id']]['split'] == 'train']
    assert len(training) == (8 if args.gate else 1356) and all(r['split'] == 'train' for r in training)
    status = dict(status='loading_train', worker_pid=os.getpid(), completed=0, total=len(entries))
    def save():
        status['elapsed_seconds'] = time.time() - start
        (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    save()
    xcell, lemon = [], []
    with (args.output / 'fitting_cells.jsonl').open('w') as stream:
        for row in training:
            assert sha(row['file']) == row['file_sha256']
            with np.load(row['file'], allow_pickle=False) as raw:
                xcell.append(raw['old_reg64'].copy())
                lemon.append(raw['lemon384'].copy())
                assert array_sha(raw['instance_ids']) == row['arrays']['instance_ids']
            stream.write(json.dumps(dict(tile_id=row['tile_id'], split='train', cells=row['retained'],
                                         raw_sha256=row['file_sha256'], instance_ids_sha256=row['arrays']['instance_ids'])) + '\n')
    xcell = np.concatenate(xcell)
    lemon = np.concatenate(lemon)
    assert xcell.shape == (len(lemon), 64) and lemon.shape[1] == 384 and len(lemon) > 64
    transform = {}
    transform['xcell_mean'], transform['xcell_std'] = population_stats(xcell)
    transform['lemon_mean'], transform['lemon_std'] = population_stats(lemon)
    standardized = ((lemon.astype(np.float64) - transform['lemon_mean']) / transform['lemon_std']).astype(np.float32)
    del xcell, lemon
    status.update(status='fitting_PCA', fitting_cells=len(standardized))
    save()
    pca = PCA(n_components=64, svd_solver='randomized', random_state=20261031, iterated_power=4, whiten=False)
    pca.fit(standardized)
    transform['pca_mean'] = np.ascontiguousarray(pca.mean_)
    transform['components'] = np.ascontiguousarray(pca.components_)
    projected = (standardized - transform['pca_mean']) @ transform['components'].T
    transform['projected_mean'], transform['projected_std'] = population_stats(projected)
    fitting_cells = len(standardized)
    del standardized, projected
    np.savez_compressed(args.output / 'transforms.npz', **transform)
    with np.load(args.output / 'transforms.npz', allow_pickle=False) as frozen:
        transform = {key: frozen[key].copy(order='C') for key in frozen.files}
    status.update(status='transforming')
    save()
    with (args.output / 'index.jsonl').open('w') as stream:
        for index, row in enumerate(entries):
            assert sha(row['file']) == row['file_sha256']
            with np.load(row['file'], allow_pickle=False) as raw:
                xgrid, counts = cell_grid(raw['positions'], cell_vectors(raw, transform, 'xcell'))
                lgrid, lcounts = cell_grid(raw['positions'], cell_vectors(raw, transform, 'lemon'))
                assert np.array_equal(counts, lcounts)
            path = args.output / (row['tile_id'] + '.npz')
            np.savez_compressed(path, xcell_grid=xgrid, lemon_grid=lgrid, counts=counts)
            output = dict(tile_id=row['tile_id'], dataset=row['dataset'], split=row['split'], retained=row['retained'],
                          file=str(path), file_sha256=sha(path), raw_sha256=row['file_sha256'],
                          xcell_array_sha256=array_sha(xgrid), lemon_array_sha256=array_sha(lgrid), counts_sha256=array_sha(counts))
            stream.write(json.dumps(output) + '\n')
            stream.flush()
            status['completed'] = index + 1
            save()
    report = dict(status='complete_train_only_cell_grids_unaccepted', gate=args.gate, worker_pid=os.getpid(), frames=len(entries),
                  fitting_frames=len(training), fitting_cells=fitting_cells, validation_fitting_cells=0,
                  fitting_manifest_sha256=sha(args.output / 'fitting_cells.jsonl'), transforms_sha256=sha(args.output / 'transforms.npz'),
                  raw_index_sha256=receipt['index_sha256'], raw_acceptance_sha256=sha(args.raw / 'independent_acceptance.json'),
                  catalog_sha256=sha(catalog / 'tiles.parquet'), index_sha256=sha(args.output / 'index.jsonl'),
                  source_sha256=sha(__file__), grid_source_sha256=sha(Path(__file__).with_name('cell_grids.py')),
                  gate_receipt_sha256=sha(args.gate_receipt) if args.gate_receipt else None,
                  PCA=dict(components=64, solver='randomized', random_state=20261031, iterated_power=4, whiten=False,
                           explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum())),
                  numpy_version=np.__version__, sklearn_version=sklearn.__version__,
                  elapsed_seconds=time.time() - start, GT_read=False,
                  channels='10 unchanged metadata +64 normalized representation +log1p retained-cell count',
                  scope='input grids only, no tissue quality or discrete vocabulary result')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    status.update(status='complete_unaccepted', completed=len(entries))
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
