"""Version training-only resolution exclusions; retain validation byte identities."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import pandas as pd

from benchmarks.v5.multidomain_data.prepare import dump, sha
from .catalog import INPUTS, split_overlap_audit


def resolution_decision(row, maximum):
    """File sampling, input grid and degradation all limit available detail."""
    values = [*json.loads(row['available_mpp_xy_json']),
              *json.loads(row['input_mpp_xy_json']),
              *json.loads(row['preparation_json'])['effective_quality_mpp_xy']]
    if len(values) != 6 or not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('missing/invalid resolution evidence')
    return max(values) <= maximum, max(values)


def build(catalog, inputs, output, maximum):
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('invalid maximum detail MPP')
    root = Path('/nfs-medical3/zyh/fewclick_v2').resolve()
    if root not in output.resolve().parents:
        raise ValueError('output outside artifact root')
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'status.json', dict(status='filtering', worker_pid=os.getpid()))
    try:
        tiles = pd.read_parquet(catalog/'tiles.parquet')
        variants = pd.read_parquet(inputs)
        original = variants[variants.quality == 'original'].set_index('tile_id', verify_integrity=True)
        if tiles.tile_id.duplicated().any() or not set(tiles.split) <= {'train', 'val'}:
            raise ValueError('invalid parent catalog')
        if 'puma' in set(tiles.dataset):
            raise ValueError('heldout data in parent catalog')
        rows, decisions = [], []
        for row in tiles.to_dict('records'):
            if row['split'] == 'val':
                rows.append(row)
                continue
            source = original.loc[row['tile_id']].to_dict()
            if source['source_id'] != row['source_id'] or not row['training_allowed']:
                raise ValueError('source/training identity mismatch')
            allowed, detail = resolution_decision(source, maximum)
            decision = dict(tile_id=row['tile_id'], source_id=row['source_id'],
                            dataset=row['dataset'], organ=row['organ'],
                            detail_mpp=detail, kept=allowed,
                            reason='original_detail_pass' if allowed else 'native_detail_too_coarse')
            decisions.append(decision)
            if allowed:
                v = json.loads(row['variants_json'])['original']
                if any(v[k] != source[k] for k in ('image', 'image_sha256', 'cells', 'cells_sha256')):
                    raise ValueError('original image/cell identity mismatch')
                rows.append(row | dict(variants_json=json.dumps({'original': v})))
        filtered = pd.DataFrame(rows, columns=tiles.columns)
        pd.testing.assert_frame_equal(tiles[tiles.split == 'val'].reset_index(drop=True),
                                      filtered[filtered.split == 'val'].reset_index(drop=True))
        split_audit = split_overlap_audit(rows)
        train = filtered[filtered.split == 'train']
        default = {'crc': .30, 'tiger': .25, 'ignite': .25, 'ocelot': .20}
        datasets = [d for d in default if d in set(train.dataset)]
        if not datasets:
            raise ValueError('empty training pool')
        weights = np.array([default[d] for d in datasets]); weights /= weights.sum()
        filtered.to_parquet(output/'tiles.parquet', index=False)
        pd.DataFrame(decisions).to_csv(output/'training_resolution_decisions.csv', index=False)
        policy = dict(version='original_detail_1023_025438', datasets=datasets,
                      probabilities=weights.tolist(), qualities=['original'],
                      max_detail_mpp=maximum,
                      catalog_sha256=sha((output/'tiles.parquet').read_bytes()),
                      validation_unchanged=True)
        dump(output/'training_policy.json', policy)
        dump(output/'run.json', dict(worker_pid=os.getpid(), catalog=str(catalog), inputs=str(inputs),
                                    parent_catalog_sha256=sha((catalog/'tiles.parquet').read_bytes()),
                                    input_manifest_sha256=sha(inputs.read_bytes()),
                                    source_sha256=sha(Path(__file__).read_bytes()), maximum=maximum))
        summary = dict(status='complete', counts=dict(Counter(filtered.dataset+'|'+filtered.split)),
                       train_tiles_before=int((tiles.split == 'train').sum()),
                       train_tiles_after=len(train), train_variants_after=len(train),
                       removed_train_by_dataset=dict(Counter(d['dataset'] for d in decisions if not d['kept'])),
                       validation_unchanged=True, validation_tiles=int((tiles.split == 'val').sum()),
                       split_overlap=split_audit, policy=policy, puma_used=False, test_used=False)
        dump(output/'summary.json', summary)
        dump(output/'status.json', dict(status='complete', worker_pid=os.getpid()))
        return summary
    except Exception as exc:
        dump(output/'status.json', dict(status='failed', worker_pid=os.getpid(), error=repr(exc)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=INPUTS)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-detail-mpp', type=float, default=.6)
    a = p.parse_args()
    print(json.dumps(build(a.catalog, a.inputs, a.output, a.max_detail_mpp), ensure_ascii=False))


if __name__ == '__main__':
    main()
