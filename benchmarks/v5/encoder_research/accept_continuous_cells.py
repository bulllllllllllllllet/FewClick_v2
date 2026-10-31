"""Replay raw cached cell identities, original vectors and canonical crop pixels."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .cache_continuous_cells import TASKS, INPUT_GATE, array_sha
from .cell_inputs import load_continuous_cell_inputs
from .cell_catalog import load_cell_catalog
from .lemon_features import FrozenLemon, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.cache / 'report.json').read_text())
    assert report['status'] == 'complete_raw_continuous_cell_cache_unaccepted'
    assert sha(args.cache / 'index.jsonl') == report['index_sha256']
    catalog, catalog_rows = load_cell_catalog(TASKS)
    assert sha(catalog / 'tiles.parquet') == report['catalog_sha256']
    assert report['encoder_identity'] == FrozenLemon.identity()
    assert report['input_gate_sha256'] == sha(INPUT_GATE)
    for key, name in [('extraction_source_sha256', 'cache_continuous_cells.py'),
                      ('cell_input_source_sha256', 'cell_inputs.py'), ('lemon_source_sha256', 'lemon_features.py'),
                      ('catalog_source_sha256', 'cell_catalog.py')]:
        assert report[key] == sha(Path(__file__).with_name(name))
    rows = [json.loads(s) for s in (args.cache / 'index.jsonl').read_text().splitlines()]
    expected = [r['tile_id'] for r in json.loads(INPUT_GATE.read_text())['rows']] if report['gate'] else sorted(catalog_rows)
    assert [r['tile_id'] for r in rows] == expected
    assert len(rows) == (16 if report['gate'] else 1737)
    retained, padding, coverage = 0, 0, {}
    for index, row in enumerate(rows):
        source = catalog_rows[row['tile_id']]
        assert (row['dataset'], row['split']) == (source['dataset'], source['split'])
        variant = json.loads(source['variants_json'])['original']
        assert row['source_cells_sha256'] == variant['cells_sha256']
        assert row['source_image_sha256'] == variant['image_sha256']
        assert sha(row['file']) == row['file_sha256']
        original = load_continuous_cell_inputs(source)
        with np.load(row['file'], allow_pickle=False) as cached:
            for key in ['instance_ids', 'positions', 'metadata10', 'old_reg64']:
                assert np.array_equal(cached[key], original[key])
            features = cached['lemon384']
            assert features.shape == (row['retained'], 384) and np.isfinite(features).all()
            assert all(array_sha(cached[key]) == value for key, value in row['arrays'].items())
        assert row['retained'] == len(original['crops'])
        assert row['total_detected'] == original['total'] and row['edge_padded'] == original['edge_padded']
        retained += row['retained']
        padding += row['edge_padded']
        key = row['dataset'] + '|' + row['split']
        coverage[key] = coverage.get(key, 0) + 1
        if (index + 1) % 25 == 0:
            (args.cache / 'accept_status.json').write_text(json.dumps(dict(status='replaying', worker_pid=os.getpid(), frames=index + 1)) + '\n')
    assert retained == report['retained_cells'] and padding == report['edge_padded']
    if report['gate']:
        assert report['fresh_reload_max_abs_error'] <= 1e-5
    else:
        assert coverage == {'crc|train': 1200, 'crc|val': 300, 'tiger|train': 156, 'tiger|val': 81}
    result = dict(status='accepted_raw16_continuous_cell_gate' if report['gate'] else 'accepted_raw1737_continuous_cells',
                  worker_pid=os.getpid(), frames_replayed=len(rows), retained_cells=retained, edge_padded=padding,
                  coverage=coverage, index_sha256=report['index_sha256'], encoder_identity=report['encoder_identity'],
                  extraction_source_sha256=report['extraction_source_sha256'], cell_input_source_sha256=report['cell_input_source_sha256'],
                  lemon_source_sha256=report['lemon_source_sha256'], acceptance_source_sha256=sha(__file__),
                  catalog_source_sha256=report['catalog_source_sha256'],
                  GT_read=False, transform_fitted=False, scope='raw vectors/instances accepted; downstream contribution untested')
    (args.cache / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.cache / 'accept_status.json').write_text(json.dumps(dict(status='complete', worker_pid=os.getpid(), frames=len(rows))) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
