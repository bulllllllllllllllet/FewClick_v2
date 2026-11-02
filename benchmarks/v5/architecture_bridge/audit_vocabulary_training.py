"""Read-only final training budget, optimizer and representation-state audit."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import BASE


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--arm', choices=['continuous', 'fixed', 'ema'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    contract = json.loads((a.pilot / 'contract.json').read_text())
    folder = a.pilot / a.arm
    # Scoring starts only after save and the trainer's own disk reload check.
    assert any('"status": "scoring"' in s for s in (a.pilot / (a.arm + '_train.log')).read_text().splitlines()), 'Wait for completed checkpoint save/reload'
    for path, digest in contract['sources'].items():
        assert sha(path) == digest, path
    source = BASE / 'phikon_joint_tasks_1031_222610/data/training4000.jsonl'
    source_bytes = source.read_bytes(); consumed = (folder / 'training_consumed.jsonl').read_bytes()
    assert len(source_bytes.splitlines()) == 4000 and consumed == source_bytes*2
    reference = a.pilot / 'continuous/report.json'
    if reference.exists():
        assert sha(source) == json.loads(reference.read_text())['training_sha256']
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['updates'] == 2000 and checkpoint['arm'] == a.arm
    assert checkpoint['identity']['seed'] == contract['seed']
    states = checkpoint['optimizer']['state']; steps = Counter(int(s['step']) for s in states.values())
    assert len(steps) == 1 and 2000 in steps
    for group in checkpoint['optimizer']['param_groups']:
        assert group['lr'] == group['weight_decay'] == 1e-4
    model = checkpoint['model']; seen = set(); checked = 0
    for tensor in list(model.values()) + [v for s in states.values() for v in s.values()]:
        if not isinstance(tensor, torch.Tensor):
            continue
        key = (tensor.data_ptr(), tensor.numel(), str(tensor.dtype))
        if key not in seen:
            seen.add(key); assert torch.isfinite(tensor).all(); checked += 1
    weight, bias = model['vocabulary.mapper.weight'], model['vocabulary.mapper.bias']
    change = max(float((weight-torch.eye(64)).abs().max()), float(bias.abs().max()))
    assert change > 0  # A state-change check, not a functional-learning claim.
    vocab = BASE / 'p2_selected_vocabulary_1102_121546/full'
    vp = json.loads((vocab / 'independent_acceptance.json').read_text())
    assert vp['vocabulary_sha256'] == sha(vocab / 'vocabulary.npz') == checkpoint['identity']['vocabulary_sha256']
    with np.load(vocab / 'vocabulary.npz', allow_pickle=False) as f:
        initial = torch.from_numpy(f['centers'].copy())
    book = {}
    if a.arm == 'continuous':
        assert 'vocabulary.centers' not in model
        book['codebook_present'] = False
    elif a.arm == 'fixed':
        assert torch.equal(model['vocabulary.centers'], initial)
        assert 'vocabulary.counts' not in model and 'vocabulary.sums' not in model
        book['fixed_centers_bit_exact'] = True
    else:
        centers, counts, sums = [model['vocabulary.' + k] for k in ['centers', 'counts', 'sums']]
        assert (counts > 0).all() and not torch.equal(centers, initial)
        expected = sums / counts.clamp_min(1e-5)[:, None]
        assert torch.allclose(centers, expected, rtol=1e-5, atol=1e-5)
        book = dict(EMA_centers_changed=True, center_state_ratio_max_abs_error=float((centers-expected).abs().max()),
                    count_min=float(counts.min()), count_max=float(counts.max()))
    result = dict(status='accepted_P2_training_budget_and_representation_state', worker_pid=os.getpid(),
                  arm=a.arm, updates=2000, consumed_tasks=8000, source_task_rows=4000,
                  optimizer_step_histogram=dict(steps), finite_unique_tensor_blocks=checked,
                  mapper_max_abs_change=change, representation_state=book,
                  checkpoint_sha256=sha(folder / 'checkpoint.pt'), training_sha256=sha(source),
                  consumed_sha256=sha(folder / 'training_consumed.jsonl'), source_sha256=sha(__file__),
                  mask_replay_accepted=False, quality_accepted=False, P2_accepted=False)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
