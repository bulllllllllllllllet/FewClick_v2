"""Shared encoding, four fixed inference interventions, exact historical control."""
import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import build, decode_variants, MODES
from .engineering import configure
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.paper_multiscale.data import native_context, intersects
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--gate', action='store_true')
    a = parser.parse_args()
    r = a.root
    c = json.loads((r/'contract.json').read_text())
    for name, digest in c['sources'].items():
        assert sha(name) == digest, name
    assert sha(r/'tasks.json') == c['tasks_sha256']
    assert sha(r/'query_rows.json') == c['query_rows_sha256']
    ck = c['checkpoints']['H']
    assert sha(ck['path']) == ck['sha256']
    all_tasks = json.loads((r/'tasks.json').read_text())
    tasks = ([all_tasks[i] for i in c['gate_indices']] if a.gate else
             [t for t in all_tasks if t['index'] % 2 == a.shard])
    out = r/('gate' if a.gate else f'shard_{a.shard}')
    out.mkdir(exist_ok=False)
    dump(out/'worker.json', dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], gate=a.gate))
    configure()
    data = DecoderData(Path(c['cache']), 'match')
    data.rows.update(json.loads((r/'query_rows.json').read_text()))
    reference_rows = {}
    reference_masks = {}
    reference = Path(c['reference_evaluation'])
    for shard in [0, 1]:
        directory = reference/f'H_{shard}'
        acceptance = json.loads((directory/'acceptance.json').read_text())
        assert acceptance['status'] == 'complete_accepted'
        assert acceptance['checkpoint_sha256'] == ck['sha256']
        assert acceptance['tasks_sha256'] == c['tasks_sha256']
        assert sha(directory/'masks.npy') == acceptance['masks_sha256']
        assert sha(directory/'rows.json') == acceptance['rows_sha256']
        reference_masks[shard] = np.load(directory/'masks.npy', mmap_mode='r')
        for i, row in enumerate(json.loads((directory/'rows.json').read_text())):
            assert row['index'] == 2*i+shard
            reference_rows[row['index']] = row['metrics']
    model = build(ck['path'])
    hashes = {}

    @lru_cache(maxsize=32)
    def inputs(identity):
        item = data.rows[identity]
        x = {k: v.to('cuda:0') for k, v in data.inputs(item.get('template_id', identity), 'xcell').items()}
        fingerprints = []
        for scale in range(3):
            rgb = np.asarray(Image.fromarray(native_context(item, scale)).resize(
                (512, 512), Image.Resampling.BILINEAR)).copy()
            fingerprints.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            x['image'+('' if scale == 0 else '_'+str(scale))] = he_to_tensor(rgb)[None].to('cuda:0')
        hashes[identity] = fingerprints
        return x

    @lru_cache(maxsize=64)
    def encoded(identity):
        return model.encode(model.base.transform(inputs(identity)))

    rows = {mode: [] for mode in MODES}
    arrays = {}
    for mode in MODES:
        (out/mode).mkdir()
        arrays[mode] = np.lib.format.open_memmap(out/mode/'masks.npy', mode='w+', dtype=np.uint8,
                                                shape=(len(tasks), 512*512//8))
    started = time.monotonic()
    for ordinal, task in enumerate(tasks):
        assert task['support_space'] == task['query_space']
        assert not task['query_has_human_clicks'] and not task['local']
        assert not intersects(json.loads(data.rows[task['support_id']]['multiscale_json'])['context_um'],
                              json.loads(data.rows[task['query_id']]['multiscale_json'])['context_um'])
        prompts = {'prompt_size_id': torch.ones(1, device='cuda:0', dtype=torch.long)}
        for sign in ['positive', 'negative']:
            prompts[sign+'_xy'] = torch.tensor([task[sign+'_xy']], device='cuda:0', dtype=torch.float32)
            prompts[sign+'_mask'] = torch.ones(1, len(task[sign+'_xy']), device='cuda:0', dtype=torch.bool)
        predictions = decode_variants(model, encoded(task['support_id']), encoded(task['query_id']), prompts)
        if a.gate or ordinal < 2:
            for mode in MODES:
                model.extension.mode = mode
                full = model(support=inputs(task['support_id']), query=inputs(task['query_id']),
                             prompts=prompts, local=False, shared=False, with_support=False)
                assert torch.equal(full['query']['pixel_probability'], predictions[mode]), mode
            model.extension.mode = 'control'
        # Ground truth enters scoring only after all four predictions are fixed.
        gt, mpp = data.labels(task['query_id'])
        for mode, prob in predictions.items():
            assert torch.isfinite(prob).all()
            mask = prob[0].cpu().numpy() >= .5
            packed = np.packbits(mask)
            metrics = score(mask, gt == task['target'], gt != 255, mpp)
            assert metrics['positive'] == task['query_positive_pixels']
            if mode == 'control':
                assert np.array_equal(packed, reference_masks[task['index'] % 2][task['index'] // 2])
                assert metrics == reference_rows[task['index']]
            rows[mode].append(dict(index=task['index'], metrics=metrics,
                                   probability_change=float((prob-predictions['control']).abs().mean())))
            arrays[mode][ordinal] = packed
        if ordinal % 25 == 0:
            for array in arrays.values():
                array.flush()
            dump(out/'progress.json', dict(status='inference_running', pid=os.getpid(), rows=ordinal+1,
                                           total=len(tasks), variants=4, seconds=time.monotonic()-started))
            print(json.dumps(dict(rows=ordinal+1, total=len(tasks), variants=4)), flush=True)
    dump(out/'input_hashes.json', hashes)
    receipts = {}
    for mode in MODES:
        arrays[mode].flush()
        dump(out/mode/'rows.json', rows[mode])
    dump(out/'progress.json', dict(status='disk_mask_replay', pid=os.getpid(), rows=len(tasks), total=len(tasks)))
    # Reload disk artifacts and independently score each saved prediction.
    stored = {mode: np.load(out/mode/'masks.npy', mmap_mode='r') for mode in MODES}
    saved = {mode: json.loads((out/mode/'rows.json').read_text()) for mode in MODES}
    for i, task in enumerate(tasks):
        gt, mpp = data.labels(task['query_id'])
        for mode in MODES:
            row = saved[mode][i]
            assert row['index'] == task['index']
            mask = np.unpackbits(stored[mode][i], count=512*512).reshape(512, 512).astype(bool)
            assert score(mask, gt == task['target'], gt != 255, mpp) == row['metrics']
    for mode in MODES:
        receipts[mode] = dict(rows_sha256=sha(out/mode/'rows.json'), masks_sha256=sha(out/mode/'masks.npy'))
    dump(out/'acceptance.json', dict(status='complete_accepted', rows=len(tasks), modes=receipts,
                                     control_matches_historical_masks_and_metrics=True,
                                     checkpoint_sha256=ck['sha256'], tasks_sha256=c['tasks_sha256'],
                                     input_hashes_sha256=sha(out/'input_hashes.json'),
                                     seconds=time.monotonic()-started,
                                     cached_full_exact_count=len(tasks) if a.gate else 2))


if __name__ == '__main__':
    main()
