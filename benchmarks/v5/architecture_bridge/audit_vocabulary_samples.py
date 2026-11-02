"""Frozen P2 train inventory diagnostics; never fit or select a vocabulary here."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .cell_vocabulary import CellVocabulary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(codes, errors, records):
    groups = {"all": np.arange(len(records))}
    for dataset in sorted({r["dataset"] for r in records}):
        for scale in range(3):
            groups[f"{dataset}|{scale}"] = np.array([
                i for i, r in enumerate(records)
                if r["dataset"] == dataset and r["scale"] == scale
            ])
    result = {}
    for name, indices in groups.items():
        assert len(indices)
        counts = np.bincount(codes[indices], minlength=128)
        p = counts[counts > 0] / len(indices)
        entropy = float(-(p * np.log(p)).sum())
        result[name] = dict(
            observations=len(indices), occupied_codes=int((counts > 0).sum()),
            counts=counts.tolist(), entropy_nats=entropy,
            effective_codes=float(np.exp(entropy)),
            largest_code_fraction=float(counts.max() / len(indices)),
            mean_squared_distance=float(errors[indices].mean()),
            per_dimension_mse=float(errors[indices].mean() / 64),
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--mode', choices=['fixed', 'ema'], default='fixed')
    p.add_argument('--accept', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    proof = json.loads((a.samples / 'independent_acceptance.json').read_text())
    assert proof['status'] == 'accepted_train_only_matched_vocabulary_samples'
    assert proof['validation_samples'] == 0 and not proof['gate']
    assert sha(a.samples / 'samples.npz') == proof['array_sha256']
    assert sha(a.samples / 'samples.jsonl') == proof['index_sha256']
    vp = json.loads((a.vocabulary / 'independent_acceptance.json').read_text())
    assert vp['status'] == 'accepted_train_only_K128_vocabulary'
    assert sha(a.vocabulary / 'vocabulary.npz') == vp['vocabulary_sha256']
    records = list(map(json.loads, (a.samples / 'samples.jsonl').read_text().splitlines()))
    with np.load(a.samples / 'samples.npz', allow_pickle=False) as f:
        x = f['xcell'].copy()
    with np.load(a.vocabulary / 'vocabulary.npz', allow_pickle=False) as f:
        centers = f['centers'].copy()
    assert len(x) == len(records) == proof['samples']
    model = CellVocabulary(a.mode, torch.from_numpy(centers))
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        assert saved.get('arm', a.mode) == a.mode
        state = {k.removeprefix('vocabulary.'): v for k, v in saved['model'].items()
                 if k.startswith('vocabulary.')}
        model.load_state_dict(state, strict=True)
    else:
        assert a.mode == 'fixed', 'EMA diagnostics require an actual checkpoint'
    model.eval()
    identity = dict(samples_sha256=proof['array_sha256'],
                    index_sha256=proof['index_sha256'],
                    vocabulary_sha256=vp['vocabulary_sha256'],
                    checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None,
                    source_sha256=sha(__file__), mode=a.mode)
    if a.accept:
        report = json.loads((a.output / 'report.json').read_text())
        assert report['identity'] == identity
        with np.load(a.output / 'assignments.npz', allow_pickle=False) as f:
            codes, errors = f['codes'], f['squared_distance']
        # Independent float64 broadcast distances, rather than torch's GEMM formula.
        for start in range(0, len(x), 128):
            raw = x[start:start + 128].astype(np.float64)
            z = raw @ model.mapper.weight.detach().numpy().astype(np.float64).T
            z += model.mapper.bias.detach().numpy().astype(np.float64)
            dist = ((z[:, None] - model.centers.numpy().astype(np.float64)[None]) ** 2).sum(2)
            c = codes[start:start + len(raw)].astype(np.int64)
            exact = dist[np.arange(len(raw)), c]
            # CPU fp32 affine/GEMM can select numerically tied nearest centers.
            assert np.all(exact <= dist.min(1) + 1e-4 * np.maximum(1, dist.min(1)))
            assert np.allclose(errors[start:start + len(raw)], exact, rtol=1e-4, atol=1e-4)
        assert summarize(codes, errors, records) == report['groups']
        (a.output / 'independent_acceptance.json').write_text(json.dumps(dict(
            status='accepted_frozen_train_inventory_diagnostics', worker_pid=os.getpid(),
            report_sha256=sha(a.output / 'report.json'), observations=len(x),
            validation_observations=0, downstream_quality_accepted=False,
            efficiency_accepted=False), indent=2) + '\n')
        return
    a.output.mkdir(exist_ok=False)
    codes, errors = [], []
    with torch.no_grad():
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for chunk in torch.from_numpy(x).split(512):
            _, c = model(chunk)
            z = model.mapper(chunk)
            error = (z.double() - model.centers[c].double()).square().sum(1)
            codes.append(c.numpy().astype(np.uint8)); errors.append(error.numpy())
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    codes, errors = np.concatenate(codes), np.concatenate(errors)
    np.savez_compressed(a.output / 'assignments.npz', codes=codes, squared_distance=errors)
    report = dict(status='frozen_train_inventory_diagnostics_unaccepted', identity=identity,
                  worker_pid=os.getpid(), groups=summarize(codes, errors, records),
                  scope='Matched train observations only; repeated physical cells across views possible',
                  labels_read=False, fitting_performed=False, eval_state_unchanged=True,
                  storage_or_WSI_speed_claim=False, validation_coverage=False)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
