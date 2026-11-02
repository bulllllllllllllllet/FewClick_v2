"""Frozen final-state color diagnostics, separate from tissue quality selection."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from .cell_vocabulary import CellVocabulary
from .data import BASE
from .prepare_native_color import CONDITIONS
from benchmarks.v5.encoder_research.continuous_data import sha

TRANSFORMS = BASE / 'continuous_cell_grids_layout_repair_1031_145335/full/transforms.npz'
TRANSFORM_SHA = '5ee94c93cb3bab8911aba3e39b651a7411659a3ca3ec0ad018cc5649b7f3429b'


def load_vocabulary(checkpoint, mode):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved.get('arm', mode) == mode
    state = {k.removeprefix('vocabulary.'): v for k, v in saved['model'].items()
             if k.startswith('vocabulary.')}
    model = CellVocabulary(mode, torch.zeros(128, 64))
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def measure(model, original, changed):
    assert original.shape == changed.shape and original.ndim == 2 and original.shape[1] == 64
    left, right = torch.from_numpy(original), torch.from_numpy(changed)
    z0, z1 = model.mapper(left), model.mapper(right)
    y0, c0 = model(left); y1, c1 = model(right)
    # Independent affine / direct broadcast distances in float64.
    for x, z, codes in [(original, z0, c0), (changed, z1, c1)]:
        exact_z = x.astype(np.float64) @ model.mapper.weight.numpy().astype(np.float64).T
        exact_z += model.mapper.bias.numpy().astype(np.float64)
        assert np.allclose(z.numpy(), exact_z, atol=1e-4, rtol=1e-4)
        if codes is not None:
            d = ((exact_z[:, None] - model.centers.numpy().astype(np.float64)[None]) ** 2).sum(2)
            chosen = d[np.arange(len(x)), codes.numpy()]
            assert np.all(chosen <= d.min(1) + 1e-4 * np.maximum(1, d.min(1)))
    return dict(cells=len(original), changed_codes=int((c0 != c1).sum()) if c0 is not None else None,
                raw_squared_sum=float(np.square(original.astype(np.float64) - changed).sum()),
                mapped_squared_sum=float((z0.double() - z1.double()).square().sum()),
                output_squared_sum=float((y0.double() - y1.double()).square().sum()),
                mapped_reference_norm_squared=float(z0.double().square().sum()),
                output_reference_norm_squared=float(y0.double().square().sum()))


def metrics(stats):
    n = stats['cells']
    return dict(cells=n, code_flip_fraction=stats['changed_codes'] / n if n and stats['changed_codes'] is not None else None,
                standardized_raw_RMSE=float(np.sqrt(stats['raw_squared_sum'] / n / 64)) if n else None,
                mapped_RMSE=float(np.sqrt(stats['mapped_squared_sum'] / n / 64)) if n else None,
                representation_RMSE=float(np.sqrt(stats['output_squared_sum'] / n / 64)) if n else None,
                mapped_relative_L2=float(np.sqrt(stats['mapped_squared_sum'] / max(stats['mapped_reference_norm_squared'], 1e-12))) if n else None,
                representation_relative_L2=float(np.sqrt(stats['output_squared_sum'] / max(stats['output_reference_norm_squared'], 1e-12))) if n else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--accept', action='store_true')
    a = p.parse_args(); torch.set_num_threads(2)
    assert json.loads((a.run / 'stage.json').read_text())['status'] == 'complete'
    fp = json.loads((a.features / 'independent_acceptance.json').read_text())
    ip = json.loads((a.inputs / 'independent_acceptance.json').read_text())
    assert fp['status'] == 'accepted_native_color_features' and ip['status'] == 'accepted_fixed_mask_native_color_inputs'
    assert sha(a.features / 'report.json') == fp['report_sha256']
    assert sha(a.inputs / 'report.json') == ip['report_sha256']
    feature_report = json.loads((a.features / 'report.json').read_text())
    input_report = json.loads((a.inputs / 'report.json').read_text())
    assert feature_report['identity']['input_report_sha256'] == ip['report_sha256']
    assert input_report['conditions'] == CONDITIONS
    assert sha(TRANSFORMS) == TRANSFORM_SHA
    with np.load(TRANSFORMS, allow_pickle=False) as f:
        mean, std = f['xcell_mean'], f['xcell_std']
    models, weights = {}, {}
    for mode in ['continuous', 'fixed', 'ema']:
        folder = a.run / mode; checkpoint = folder / 'checkpoint.pt'
        accepted = json.loads((folder / 'independent_acceptance.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        assert accepted['status'] == 'accepted_P2_vocabulary_evaluation' and not accepted['smoke']
        assert accepted['rows'] == 3711 and report['updates'] == 2000 and report['arm'] == mode
        assert accepted['report_sha256'] == sha(folder / 'report.json')
        assert accepted['checkpoint_sha256'] == report['checkpoint_sha256'] == sha(checkpoint)
        models[mode] = load_vocabulary(checkpoint, mode); weights[mode] = sha(checkpoint)
    before = {m: {k: v.clone() for k, v in model.state_dict().items()} for m, model in models.items()}
    assert len(input_report['records']) == len(feature_report['frames']) == 16
    records = []
    for row, saved in zip(input_report['records'], feature_report['frames']):
        assert row['tile_id'] == saved['tile_id']
        path = a.features / saved['file']; assert sha(path) == saved['sha256']
        with np.load(path, allow_pickle=False) as f, np.load(row['geometry'], allow_pickle=False) as g:
            assert sha(row['geometry']) == row['geometry_sha256']
            assert np.array_equal(f['instance_ids'], g['instance_ids']) and list(f['conditions']) == list(CONDITIONS)
            values = ((f['features'].astype(np.float64) - mean) / std).astype(np.float32)
        for mode, model in models.items():
            for index, condition in enumerate(CONDITIONS):
                stats = measure(model, values[0], values[index])
                records.append(dict(tile_id=row['tile_id'], dataset=row['dataset'], split=row['split'],
                                    mode=mode, condition=condition, stats=stats, metrics=metrics(stats)))
    assert all(torch.equal(v, before[m][k]) for m, model in models.items() for k, v in model.state_dict().items())
    grouped = defaultdict(list)
    for row in records:
        grouped['|'.join(row[k] for k in ['mode', 'condition', 'dataset', 'split'])].append(row)
    groups = {}
    for key, rows in grouped.items():
        sums = {k: sum(r['stats'][k] for r in rows) if rows[0]['stats'][k] is not None else None for k in rows[0]['stats']}
        names = list(rows[0]['metrics'])[1:]
        frame_mean = {k: float(np.mean([r['metrics'][k] for r in rows if r['metrics'][k] is not None]))
                      if any(r['metrics'][k] is not None for r in rows) else None for k in names}
        groups[key] = dict(frames=len(rows), empty_frames=sum(r['stats']['cells'] == 0 for r in rows),
                           pooled_cells=metrics(sums), mean_over_nonempty_frames=frame_mean)
    identity = dict(feature_report_sha256=fp['report_sha256'], input_report_sha256=ip['report_sha256'],
                    checkpoints=weights, transforms_sha256=TRANSFORM_SHA, source_sha256=sha(__file__))
    if a.accept:
        report = json.loads((a.output / 'report.json').read_text())
        assert report['identity'] == identity and report['records'] == records and report['groups'] == groups
        (a.output / 'independent_acceptance.json').write_text(json.dumps(dict(
            status='accepted_final_native_color_diagnostics', report_sha256=sha(a.output / 'report.json'),
            worker_pid=os.getpid(), records=len(records), diagnostic_only=True, P2_accepted=False), indent=2) + '\n')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        result = dict(status='final_native_color_diagnostics_unaccepted', worker_pid=os.getpid(),
                      identity=identity, records=records, groups=groups, labels_read=False,
                      eval_state_unchanged=True, fitted_statistics=False, P2_accepted=False,
                      limitations=['synthetic OD stress, fixed fine-view instances and cell context',
                                   'cells within a field are dependent; no biological cell-type claim',
                                   'diagnostics do not replace tissue quality, efficiency or seed replication'])
        (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
