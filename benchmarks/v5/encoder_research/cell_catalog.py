"""Read the frozen input inventory without importing slide IO or label loaders."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def load_cell_catalog(tasks):
    contract = json.loads((Path(tasks) / 'contract.json').read_text())
    catalog = Path(contract['catalog'])
    assert hashlib.sha256((catalog / 'tiles.parquet').read_bytes()).hexdigest() == contract['catalog_sha256']
    records = pd.read_parquet(catalog / 'tiles.parquet').to_dict('records')
    rows = {r['tile_id']: r for r in records}
    assert len(rows) == len(records) == 1737
    counts = {}
    for row in records:
        key = row['dataset'] + '|' + row['split']
        counts[key] = counts.get(key, 0) + 1
    assert counts == {'crc|train': 1200, 'crc|val': 300, 'tiger|train': 156, 'tiger|val': 81}
    return catalog, rows
