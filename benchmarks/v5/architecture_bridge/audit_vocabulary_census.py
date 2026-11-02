"""All retained three-view cell observations, with frozen train/val diagnostics."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from .audit_vocabulary_samples import sha
from .cell_vocabulary import CellVocabulary
from .data import BASE
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS


def raw_cells(row, mean, std):
    """Match BridgeData's cell path exactly, without loading RGB or query labels."""
    variant = json.loads(row['variants_json'])['original']
    assert sha(variant['cells']) == variant['cells_sha256']
    assert sha(row['multiscale_cache']) == row['multiscale_sha256']
    with np.load(variant['cells'], allow_pickle=False) as f:
        cells = [f['features'].copy()]
    with np.load(row['multiscale_cache'], allow_pickle=False) as f:
        cells += [f['cells_1'].copy(), f['cells_2'].copy()]
    for old in cells:
        assert old.ndim == 2 and old.shape[1] == 74
        value = np.concatenate([old[:, :10], (old[:, 10:].astype(np.float64) - mean) / std], 1).astype(np.float32)
        assert np.isfinite(value).all() and np.array_equal(value[:, :10], old[:, :10])
        yield value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['fixed', 'ema'], default='fixed')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gate', action='store_true')
    p.add_argument('--accept', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    catalog, rows = load_cell_catalog(TASKS)
    vocab = BASE / 'p2_selected_vocabulary_1102_121546/full'
    vp = json.loads((vocab / 'independent_acceptance.json').read_text())
    assert vp['status'] == 'accepted_train_only_K128_vocabulary'
    assert vp['validation_samples'] == 0
    assert sha(vocab / 'vocabulary.npz') == vp['vocabulary_sha256']
    transform = BASE / 'continuous_cell_grids_layout_repair_1031_145335/full/transforms.npz'
    assert sha(transform) == '5ee94c93cb3bab8911aba3e39b651a7411659a3ca3ec0ad018cc5649b7f3429b'
    with np.load(transform, allow_pickle=False) as f:
        mean, std = f['xcell_mean'].copy(), f['xcell_std'].copy()
    with np.load(vocab / 'vocabulary.npz', allow_pickle=False) as f:
        model = CellVocabulary(a.mode, torch.from_numpy(f['centers'].copy()))
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        assert saved.get('arm', a.mode) == a.mode
        model.load_state_dict({k.removeprefix('vocabulary.'): v for k, v in saved['model'].items()
                               if k.startswith('vocabulary.')}, strict=True)
    else:
        assert a.mode == 'fixed'
    model.eval()
    selected = sorted(rows)
    if a.gate:
        selected = [min(k for k, r in rows.items() if r['dataset'] == d and r['split'] == s)
                    for d in ['crc', 'tiger'] for s in ['train', 'val']]
    identity = dict(source_sha256=sha(__file__),
                    cell_vocabulary_source_sha256=sha(Path(__file__).with_name('cell_vocabulary.py')),
                    catalog_sha256=sha(catalog / 'tiles.parquet'),
                    vocabulary_sha256=vp['vocabulary_sha256'],
                    transforms_sha256=sha(transform), mode=a.mode, gate=a.gate,
                    checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None)
    if a.accept:
        report = json.loads((a.output / 'report.json').read_text())
        assert report['identity'] == identity
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / 'frames').mkdir()
    counts = defaultdict(lambda: np.zeros(128, dtype=np.int64))
    error_sums = defaultdict(float)
    frame_records = []
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for index, tile in enumerate(selected):
            row = rows[tile]
            codes_out, errors_out, sizes = [], [], []
            path = a.output / 'frames' / f'{index:04d}.npz'
            prior = np.load(path, allow_pickle=False) if a.accept else None
            for scale, cells in enumerate(raw_cells(row, mean, std)):
                x = torch.from_numpy(cells[:, 10:].copy())
                _, codes = model(x)
                z = model.mapper(x)
                errors = (z.double() - model.centers[codes].double()).square().sum(1).numpy()
                if a.accept:
                    c = prior['codes_' + str(scale)]
                    e = prior['errors_' + str(scale)]
                    assert c.shape == e.shape == (len(cells),)
                    # Independently recompute affine and nearest distances in float64.
                    zv = x.numpy().astype(np.float64) @ model.mapper.weight.numpy().astype(np.float64).T
                    zv += model.mapper.bias.numpy().astype(np.float64)
                    distances = ((zv[:, None] - model.centers.numpy().astype(np.float64)[None]) ** 2).sum(2)
                    exact = distances[np.arange(len(c)), c.astype(np.int64)]
                    assert np.all(exact <= distances.min(1) + 1e-4 * np.maximum(1, distances.min(1)))
                    assert np.allclose(e, exact, rtol=1e-4, atol=1e-4)
                    assert np.array_equal(c, codes.numpy()) and np.array_equal(e, errors)
                codes_out.append(codes.numpy().astype(np.uint8)); errors_out.append(errors); sizes.append(len(x))
                for key in ['all', row['split'], f'{row["dataset"]}|{row["split"]}|{scale}']:
                    counts[key] += np.bincount(codes.numpy(), minlength=128)
                    error_sums[key] += float(errors.sum())
            if a.accept:
                prior.close()
            else:
                np.savez_compressed(path, **{**{'codes_' + str(s): c for s, c in enumerate(codes_out)},
                                             **{'errors_' + str(s): e for s, e in enumerate(errors_out)}})
            frame_records.append(dict(tile_id=tile, dataset=row['dataset'], split=row['split'],
                                      retained_per_scale=sizes, file_sha256=sha(path)))
            if index % 100 == 0:
                print(json.dumps(dict(frames=index + 1, total=len(selected), accept=a.accept)), flush=True)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    groups = {}
    for name, c in counts.items():
        n = int(c.sum()); p_used = c[c > 0] / n
        entropy = float(-(p_used * np.log(p_used)).sum())
        groups[name] = dict(observations=n, occupied_codes=int((c > 0).sum()), counts=c.tolist(),
                            effective_codes=float(np.exp(entropy)), largest_code_fraction=float(c.max() / n),
                            per_dimension_mse=error_sums[name] / n / 64)
    if not a.gate:
        assert len(selected) == 1737 and groups['all']['observations'] == 1160545
    if a.accept:
        assert report['groups'] == groups and report['frames'] == frame_records
        result = dict(status='accepted_retained_cell_census_gate' if a.gate else 'accepted_retained_cell_census',
                      report_sha256=sha(a.output / 'report.json'), worker_pid=os.getpid(), frames=len(selected),
                      observations=groups['all']['observations'], frozen_validation=True,
                      quality_accepted=False, efficiency_accepted=False)
        (a.output / 'independent_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        result = dict(identity=identity, groups=groups, frames=frame_records, worker_pid=os.getpid(),
                      status='retained_cell_census_unaccepted', labels_read=False, eval_state_unchanged=True,
                      scope='Retained observations; detection cap255 and repeated cross-scale cells remain',
                      not_full_WSI=True, fitting_performed=False)
        (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
