"""Audit train membership/statistics and replay all cell grid reductions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .cache_continuous_cells import TASKS, array_sha
from .cell_catalog import load_cell_catalog
from .cell_grids import cell_vectors
from .lemon_features import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--grids', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.grids / 'report.json').read_text())
    assert report['status'] == 'complete_train_only_cell_grids_unaccepted'
    receipt = json.loads((args.raw / 'independent_acceptance.json').read_text())
    expected_raw = 'accepted_raw16_continuous_cell_gate' if report['gate'] else 'accepted_raw1737_continuous_cells'
    assert receipt['status'] == expected_raw
    assert sha(args.raw / 'independent_acceptance.json') == report['raw_acceptance_sha256']
    assert sha(args.raw / 'index.jsonl') == receipt['index_sha256'] == report['raw_index_sha256']
    assert sha(args.grids / 'index.jsonl') == report['index_sha256']
    assert sha(args.grids / 'transforms.npz') == report['transforms_sha256']
    assert sha(args.grids / 'fitting_cells.jsonl') == report['fitting_manifest_sha256']
    assert sha(Path(__file__).with_name('fit_continuous_cells.py')) == report['source_sha256']
    assert sha(Path(__file__).with_name('cell_grids.py')) == report['grid_source_sha256']
    catalog, inventory = load_cell_catalog(TASKS)
    assert sha(catalog / 'tiles.parquet') == report['catalog_sha256']
    source = [json.loads(s) for s in (args.raw / 'index.jsonl').read_text().splitlines()]
    grids = [json.loads(s) for s in (args.grids / 'index.jsonl').read_text().splitlines()]
    fitting = [json.loads(s) for s in (args.grids / 'fitting_cells.jsonl').read_text().splitlines()]
    expected_fit = [r for r in source if inventory[r['tile_id']]['split'] == 'train']
    assert [r['tile_id'] for r in fitting] == [r['tile_id'] for r in expected_fit]
    assert len(fitting) == report['fitting_frames'] == (8 if report['gate'] else 1356)
    for fit, raw in zip(fitting, expected_fit):
        assert fit == dict(tile_id=raw['tile_id'], split='train', cells=raw['retained'],
                           raw_sha256=raw['file_sha256'], instance_ids_sha256=raw['arrays']['instance_ids'])
    assert len(grids) == len(source) == (16 if report['gate'] else 1737)
    with np.load(args.grids / 'transforms.npz', allow_pickle=False) as data:
        transform = {k: data[k].copy() for k in data.files}
    assert transform['components'].shape == (64, 384)
    assert np.allclose(transform['components'] @ transform['components'].T, np.eye(64), atol=1e-5)
    with np.load(source[0]['file'], allow_pickle=False) as raw:
        singleton = {key: raw[key][:1].copy() for key in ['old_reg64', 'lemon384', 'metadata10', 'positions']}
    alternate_layout = dict(transform, components=np.asfortranarray(transform['components']))
    assert np.array_equal(cell_vectors(singleton, transform, 'lemon'), cell_vectors(singleton, alternate_layout, 'lemon'))
    from .cell_grids import cell_grid
    empty = {key: value[:0] for key, value in singleton.items()}
    empty_grid, empty_counts = cell_grid(empty['positions'], cell_vectors(empty, transform, 'lemon'))
    assert not empty_grid.any() and not empty_counts.any()
    sums = {'xcell': np.zeros(64), 'lemon': np.zeros(384)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    projected_sum, projected_square = np.zeros(64), np.zeros(64)
    count = 0
    for index, (raw_row, row) in enumerate(zip(source, grids)):
        for key in ['tile_id', 'dataset', 'split', 'retained']:
            assert row[key] == raw_row[key]
        assert inventory[row['tile_id']]['split'] == row['split']
        assert row['raw_sha256'] == raw_row['file_sha256'] == sha(raw_row['file'])
        assert sha(row['file']) == row['file_sha256']
        with np.load(raw_row['file'], allow_pickle=False) as raw, np.load(row['file'], allow_pickle=False) as cached:
            position = raw['positions']
            expected_counts = np.zeros((32, 32), np.int64)
            xy = np.floor(position * 32).astype(int)
            for x, y in xy:
                expected_counts[y, x] += 1
            assert expected_counts.sum() == row['retained'] and np.array_equal(cached['counts'], expected_counts)
            assert array_sha(cached['counts']) == row['counts_sha256']
            for arm in ['xcell', 'lemon']:
                vectors = cell_vectors(raw, transform, arm)
                average = np.zeros((74, 32, 32), np.float64)
                for (x, y), value in zip(xy, vectors):
                    average[:, y, x] += value
                average /= np.maximum(expected_counts[None], 1)
                expected = np.concatenate([average, np.log1p(expected_counts[None])]).astype(np.float32)
                assert cached[arm + '_grid'].shape == (75, 32, 32)
                assert np.array_equal(cached[arm + '_grid'], expected)
                assert array_sha(expected) == row[arm + '_array_sha256']
                if row['split'] == 'train' and arm == 'lemon':
                    projected = vectors[:, 10:].astype(np.float64)
                    projected_sum += projected.sum(0)
                    projected_square += (projected ** 2).sum(0)
            if row['split'] == 'train':
                count += row['retained']
                for name, key in [('xcell', 'old_reg64'), ('lemon', 'lemon384')]:
                    value = raw[key].astype(np.float64)
                    sums[name] += value.sum(0)
                    squares[name] += ((value - transform[name + '_mean']) ** 2).sum(0)
        if (index + 1) % 25 == 0:
            (args.grids / 'accept_status.json').write_text(json.dumps(dict(status='replaying', worker_pid=os.getpid(), frames=index + 1)) + '\n')
    assert count == report['fitting_cells'] == sum(r['cells'] for r in fitting)
    for name in sums:
        mean = sums[name] / count
        std = np.sqrt(squares[name] / count)
        std[std == 0] = 1
        assert np.allclose(mean, transform[name + '_mean'], atol=1e-8, rtol=1e-6)
        assert np.allclose(std, transform[name + '_std'], atol=1e-8, rtol=1e-6)
    projected_mean = projected_sum / count
    projected_std = np.sqrt(projected_square / count - projected_mean ** 2)
    assert np.max(np.abs(projected_mean)) < 5e-4 and np.max(np.abs(projected_std - 1)) < 5e-4
    assert report['validation_fitting_cells'] == 0
    result = dict(status='accepted_train_only16_cell_grid_gate' if report['gate'] else 'accepted_train_only1737_cell_grids',
                  worker_pid=os.getpid(), frames_replayed=len(grids), fitting_frames=len(fitting), fitting_cells=count,
                  validation_fitting_cells=0, all_grid_arrays_exact=True, train_statistics_recomputed=True,
                  singleton_C_F_layout_exact=True, empty_cell_input_valid=True,
                  projected_train_mean_max_abs=float(np.abs(projected_mean).max()),
                  projected_train_std_max_error=float(np.abs(projected_std - 1).max()),
                  index_sha256=report['index_sha256'], transforms_sha256=report['transforms_sha256'],
                  fitting_source_sha256=report['source_sha256'], grid_source_sha256=report['grid_source_sha256'],
                  acceptance_source_sha256=sha(__file__), raw_index_sha256=report['raw_index_sha256'],
                  GT_read=False, scope='train-only transforms and matched cell inputs accepted; no tissue quality result')
    (args.grids / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.grids / 'accept_status.json').write_text(json.dumps(dict(status='complete', worker_pid=os.getpid(), frames=len(grids))) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
