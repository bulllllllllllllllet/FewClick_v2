"""Warm-started frozen-encoder controls for progressively richer native features."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from . import module_source
from .continue_dense_boundary import DenseContextModel, dense_loss
from .multiscale_dense_decoder import MultiscaleDenseModel
from .train_dense_boundary import evaluate


def decoder_milliseconds(model, sample):
    with torch.no_grad():
        for _ in range(3):
            model(sample)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            model(sample)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end)/10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:
        raise ValueError('epochs/artifact root')
    if json.loads((args.cache/'status.json').read_text())['status'] != 'complete':
        raise RuntimeError('feature cache incomplete')
    cache = json.loads((args.cache/'run.json').read_text())
    if bool(cache['args']['smoke']) != args.smoke:
        raise RuntimeError('smoke/cache scope mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    seed = 20261014
    checkpoint = Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/next_stage_1014_200033/matched96k_1014_221710/dense128_epoch_031.pth')
    coarse_state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    refs = cache['refs']
    source_ids = {s: [r['identity']['patch_id'] for r in rows] for s, rows in refs.items()}
    if set(source_ids['train']) & set(source_ids['val']) or any(len(set(v)) != len(v) for v in source_ids.values()):
        raise RuntimeError('unique patch/split identity')
    modules = ['train_multiscale_dense', 'multiscale_dense_decoder', 'prepare_multiscale_cache',
               'continue_dense_boundary', 'dense_prompt_decoder', 'train_dense_boundary']
    run = {'args': {**cache['args'], 'epochs': args.epochs, 'seed': seed}, 'refs': refs,
           'cache': str(args.cache), 'cache_version': cache['cache_version'], 'cache_bytes': cache['cache_bytes'],
           'coarse_checkpoint': str(checkpoint), 'coarse_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'source_sha256': {str(module_source(m)): hashlib.sha256(module_source(m).read_bytes()).hexdigest() for m in modules},
           'worker_pid': os.getpid(), 'test_used': False, 'baseline_label': 'Frozen dense128 coarse',
           'comparison_pairs': [['native', 'control'], ['middle', 'native'], ['full256', 'middle']],
           'optimizer': {'coarse_lr': 1e-5, 'new_layers_lr': 1e-4, 'weight_decay': 1e-4},
           'budget_note': 'Same warm-start, manifests, updates and loss; architecture FLOPs/parameters/runtime differ.',
           'additional_reference_rows': {'point_global_final': 'point_global_reference_rows.json'},
           'scope': 'developmental spatial validation; same val used for selection; historical encoder exposure'}
    point_path = Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/point_objective_1015_125213/pilot_1015_125213/global_epoch_007_rows.json')
    point_rows = {row['patch_id']: row for row in json.loads(point_path.read_text())}
    reference_rows = [point_rows[r['identity']['patch_id']] for r in refs['val']]
    (args.output/'point_global_reference_rows.json').write_text(json.dumps(reference_rows))
    (args.output/'run.json').write_text(json.dumps(run, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'baseline', 'worker_pid': os.getpid()}))
    baseline_model = DenseContextModel().to(device)
    baseline_model.load_state_dict(coarse_state, strict=True)
    baseline_model.eval().requires_grad_(False)
    baseline, _ = evaluate(baseline_model, refs['val'], device, args.output/'baseline_rows.json')
    result = {'baseline': baseline, 'arms': {}}
    real = torch.load(refs['val'][0]['file'], map_location=device, weights_only=False)
    with torch.no_grad():
        expected = baseline_model(real)['pixel_probability']
    print('BASELINE', json.dumps(baseline), flush=True)
    for mode in ['control', 'native', 'middle', 'full256']:
        torch.manual_seed(seed)
        model = MultiscaleDenseModel(coarse_state, mode).to(device)
        model.eval()
        with torch.no_grad():
            initial = model(real)['pixel_probability']
            error = float((initial-expected).abs().max())
            if error > 2e-6:
                raise RuntimeError(f'warm-start mismatch {mode}: {error}')
            altered = {**real, 'truth': ~real['truth'], 'valid': ~real['valid'], 'target_class': torch.tensor([11], device=device)}
            if not torch.equal(initial, model(altered)['pixel_probability']):
                raise RuntimeError('GT-dependent inference')
        groups = [{'params': model.coarse.parameters(), 'lr': 1e-5}]
        if mode != 'control':
            groups.append({'params': model.fusion.parameters(), 'lr': 1e-4})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        history = []
        max_gradient = 0.
        max_new_gradient = 0.
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(args.epochs):
            start = time.time()
            model.train()
            losses = []
            for index in np.random.default_rng(seed+epoch).permutation(len(refs['train'])):
                sample = torch.load(refs['train'][index]['file'], map_location=device, weights_only=False)
                loss = dense_loss(model(sample), sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError('loss')
                optimizer.zero_grad()
                loss.backward()
                if mode != 'control':
                    new_norm = sum(float(p.grad.detach().square().sum()) for p in model.fusion.parameters() if p.grad is not None)**.5
                    if not np.isfinite(new_norm):
                        raise FloatingPointError('new feature branch gradient')
                    max_new_gradient = max(max_new_gradient, new_norm)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(norm):
                    raise FloatingPointError('gradient')
                max_gradient = max(max_gradient, float(norm))
                optimizer.step()
                losses.append(float(loss.detach()))
            train_seconds = time.time()-start
            model.eval()
            summary, _ = evaluate(model, refs['val'], device, args.output/f'{mode}_epoch_{epoch:03}_rows.json')
            with torch.no_grad():
                altered = {**real, 'truth': ~real['truth'], 'valid': ~real['valid']}
                if not torch.equal(model(real)['pixel_probability'], model(altered)['pixel_probability']):
                    raise RuntimeError('trained GT-dependent inference')
            summary.update(epoch=epoch, train_loss=float(np.mean(losses)), train_seconds=train_seconds,
                           seconds=time.time()-start, optimizer_steps=(epoch+1)*len(refs['train']),
                           trainable_parameters=sum(p.numel() for p in model.parameters()),
                           peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            summary['eligible'] = (summary['bf1_2'] >= baseline['bf1_2']+.01
                                   and summary['dice'] >= baseline['dice']-.002
                                   and summary['micro_dice'] >= baseline['micro_dice']-.002
                                   and summary['credible_violations'] <= baseline['credible_violations'])
            history.append(summary)
            result['arms'][mode] = {'history': history, 'max_gradient_before_clip': max_gradient,
                                   'max_new_branch_gradient': max_new_gradient, 'initial_error': error}
            torch.save({'model': model.state_dict(), 'mode': mode, 'epoch': epoch, 'summary': summary, 'run': run},
                       args.output/f'{mode}_epoch_{epoch:03}.pth')
            (args.output/'summary.json').write_text(json.dumps(result, indent=2))
            (args.output/'status.json').write_text(json.dumps({'status': 'training', 'arm': mode, 'epoch': epoch, 'worker_pid': os.getpid()}))
            print('EPOCH', mode, json.dumps(summary), flush=True)
        if max_gradient == 0 or (mode != 'control' and max_new_gradient == 0):
            raise RuntimeError('missing trainable gradient')
        loaded = MultiscaleDenseModel(coarse_state, mode).to(device)
        loaded.load_state_dict(torch.load(args.output/f'{mode}_epoch_{args.epochs-1:03}.pth', map_location=device, weights_only=False)['model'], strict=True)
        loaded.eval()
        with torch.no_grad():
            reload_error = float((loaded(real)['pixel_probability']-model(real)['pixel_probability']).abs().max())
        if reload_error != 0:
            raise RuntimeError('checkpoint round trip')
        result['arms'][mode]['reload_max_error'] = reload_error
        result['arms'][mode]['decoder_ms'] = decoder_milliseconds(model, real)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid()}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
