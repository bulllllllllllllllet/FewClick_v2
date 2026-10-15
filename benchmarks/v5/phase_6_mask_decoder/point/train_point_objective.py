"""Matched point-head controls: restore boundary BCE, then whole-image Dice."""
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
from .continue_dense_boundary import DenseContextModel
from .point_boundary_detail import PointBoundaryDetail, training_indices
from .point_objective import inference_indices, inference_composite, objective
from .train_dense_boundary import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:
        raise ValueError('epochs/artifact root')
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/point_detail_1014_233341/pilot_1014_233341/run.json')
    source = json.loads(source_path.read_text())
    for p, digest in source['source_sha256'].items():
        if hashlib.sha256(module_source(Path(p).stem).read_bytes()).hexdigest() != digest:
            raise RuntimeError('historical point source changed')
    refs = source['refs']
    if args.smoke:
        refs = {split: rows[:4 if split == 'train' else 2] for split, rows in refs.items()}
    if {r['identity']['patch_id'] for r in refs['train']} & {r['identity']['patch_id'] for r in refs['val']}:
        raise RuntimeError('train/val patch overlap')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    seed = 20261014
    coarse_state = torch.load(source['coarse_checkpoint'], map_location='cpu', weights_only=False)['model']
    coarse = DenseContextModel().to(device)
    coarse.load_state_dict(coarse_state, strict=True)
    coarse.requires_grad_(False).eval()
    paths = [module_source(name.removesuffix('.py')) for name in ['train_point_objective.py', 'point_objective.py', 'point_boundary_detail.py',
                                   'continue_dense_boundary.py', 'dense_prompt_decoder.py', 'train_dense_boundary.py']]
    run = {'args': {**source['args'], 'epochs': args.epochs, 'smoke': args.smoke, 'seed': seed},
           'refs': refs, 'source': str(source_path), 'coarse_checkpoint': source['coarse_checkpoint'],
           'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           'worker_pid': os.getpid(), 'test_used': False, 'training_sampled_points': 4096,
           'training_inference_points': 65536, 'inference_points': 65536,
           'comparison_pairs': [['boundary', 'control'], ['global', 'boundary']],
           'baseline_label': 'Frozen dense128 coarse',
           'scope': 'development spatial validation; same val selection; historical encoder exposure',
           'budget_note': 'All arms forward 69632 points; backward work differs by loss. More work than historical 4096-point pilot.'}
    (args.output/'run.json').write_text(json.dumps(run, indent=2))
    baseline, _ = evaluate(coarse, refs['val'], device, args.output/'baseline_rows.json')
    result = {'baseline': baseline, 'arms': {}}
    real = torch.load(refs['val'][0]['file'], map_location=device, weights_only=False)
    for mode in ['control', 'boundary', 'global']:
        torch.manual_seed(seed)
        model = PointBoundaryDetail(coarse_state).to(device)
        model.eval()
        with torch.no_grad():
            first = model(real)
            if (first['pixel_probability']-first['coarse_probability']).abs().max() > 2e-6:
                raise RuntimeError('zero initialization')
            if not torch.equal(first['indices'], model({**real, 'truth': ~real['truth'], 'valid': ~real['valid']})['indices']):
                raise RuntimeError('GT-dependent inference')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        history = []
        max_gradient = 0.
        for epoch in range(args.epochs):
            model.train()
            totals = []
            terms = []
            start = time.time()
            for index in np.random.default_rng(seed+epoch).permutation(len(refs['train'])):
                sample = torch.load(refs['train'][index]['file'], map_location=device, weights_only=False)
                if args.smoke and (sample['image'].shape != (1, 3, 512, 512)
                                   or not all(torch.isfinite(v).all() for v in sample.values())):
                    raise ValueError('real cache contract')
                with torch.no_grad():
                    probability = coarse(sample)['pixel_probability']
                    sampled = training_indices(probability, sample['truth'], sample['valid'], 'uncertain', seed+epoch*100000+int(index))
                    render = inference_indices(probability)
                output = model(sample, torch.cat((sampled, render), 1))
                loss, parts = objective(output, sample, sampled.shape[1], mode)
                if not torch.isfinite(loss):
                    raise FloatingPointError('objective')
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 5)
                if not torch.isfinite(norm):
                    raise FloatingPointError('gradient')
                max_gradient = max(max_gradient, float(norm))
                if args.smoke:
                    with torch.no_grad():
                        composite, _ = inference_composite(output, sampled.shape[1])
                        actual = model(sample)['pixel_probability']
                        if (composite-actual).abs().max() > 2e-6:
                            raise RuntimeError('training/inference composite mismatch')
                optimizer.step()
                totals.append(float(loss.detach()))
                terms.append(parts)
            train_seconds = time.time()-start
            model.eval()
            summary, _ = evaluate(model, refs['val'], device, args.output/f'{mode}_epoch_{epoch:03}_rows.json')
            with torch.no_grad():
                output = model(real)
                keep = torch.ones_like(output['pixel_probability'], dtype=torch.bool).flatten(1)
                keep.scatter_(1, output['indices'], False)
                if not torch.equal(output['pixel_probability'].flatten(1)[keep], output['coarse_probability'].flatten(1)[keep]):
                    raise RuntimeError('outside selection modified')
            summary.update(epoch=epoch, train_loss=float(np.mean(totals)),
                           loss_terms={k: float(np.mean([t[k] for t in terms])) for k in terms[0]},
                           train_seconds=train_seconds, seconds=time.time()-start,
                           optimizer_steps=(epoch+1)*len(refs['train']),
                           forward_points=(epoch+1)*len(refs['train'])*(4096+65536))
            summary['eligible'] = (summary['bf1_2'] >= baseline['bf1_2']+.01
                                   and summary['dice'] >= baseline['dice']-.002
                                   and summary['micro_dice'] >= baseline['micro_dice']-.002
                                   and summary['credible_violations'] <= baseline['credible_violations'])
            history.append(summary)
            result['arms'][mode] = {'history': history, 'max_gradient_before_clip': max_gradient}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'summary': summary, 'run': run},
                       args.output/f'{mode}_epoch_{epoch:03}.pth')
            (args.output/'summary.json').write_text(json.dumps(result, indent=2))
            (args.output/'status.json').write_text(json.dumps({'status': 'training', 'arm': mode, 'epoch': epoch, 'worker_pid': os.getpid()}))
            print('EPOCH', mode, json.dumps(summary), flush=True)
        if max_gradient == 0:
            raise RuntimeError('no trainable gradient')
        loaded = PointBoundaryDetail(coarse_state).to(device)
        loaded.load_state_dict(torch.load(args.output/f'{mode}_epoch_{args.epochs-1:03}.pth', map_location=device, weights_only=False)['model'], strict=True)
        loaded.eval()
        with torch.no_grad():
            error = float((loaded(real)['pixel_probability']-model(real)['pixel_probability']).abs().max())
        if error != 0:
            raise RuntimeError('checkpoint round trip')
        result['arms'][mode]['reload_max_error'] = error
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid()}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
