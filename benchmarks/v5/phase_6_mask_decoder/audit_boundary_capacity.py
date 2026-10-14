"""Read-only model diagnostics; GT-assisted reconstructions are NOT predictions.

Fit 64 coefficients in [0,1] against the query GT by convex least squares.
This tests a fixed assignment representation, not maximal achievable Dice/BF1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_cdt
from scipy.optimize import minimize

from .boundary_data import load_baseline, make_loader, move_batch
from .boundary_refinement import BoundaryPixelRefiner
from .probe_boundary_refinement import metrics, summarize
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary


def fit_coefficients(assignment, truth, valid, initial):
    """Solve the bounded convex quadratic using all valid pixels, no subsampling."""
    a = assignment[0].flatten(1)[:, valid.flatten()].double()
    y = truth.flatten()[valid.flatten()].double()
    gram = (a @ a.T / y.numel()).cpu().numpy()
    rhs = (a @ y / y.numel()).cpu().numpy()
    start = initial[0].double().cpu().numpy()
    def objective(q):
        return .5 * q @ gram @ q - rhs @ q, gram @ q - rhs
    result = minimize(objective, start, jac=True, bounds=[(0., 1.)] * len(start),
                      method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
    value, grad = objective(result.x)
    projected_gradient = result.x - np.clip(result.x - grad, 0., 1.)
    kkt = float(np.abs(projected_gradient).max())
    if not result.success or kkt > 1e-6 or value > objective(start)[0] + 1e-9:
        raise RuntimeError(f'oracle optimization failed: {result.message}, KKT={kkt}')
    q = torch.as_tensor(result.x, device=assignment.device, dtype=assignment.dtype)[None]
    probability = torch.einsum('bkhw,bk->bhw', assignment, q)
    return probability, {'success': bool(result.success), 'iterations': int(result.nit),
                         'projected_gradient_max': kkt,
                         'objective_reduction': float(objective(start)[0] - value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/development_1014_172905'))
    parser.add_argument('--limit', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.limit <= 64:
        raise ValueError('limit must be 1..64')
    if Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:
        raise ValueError('output must be under FewClick artifact root')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    ref = json.loads((args.reference/'run.json').read_text())
    metadata = json.loads((Path(ref['baseline_root'])/'run_metadata.json').read_text())
    loader_args = argparse.Namespace(**{k: Path(v) if k in ('spatial_data','routing') else v for k,v in ref.items()})
    model = load_baseline(metadata, Path(ref['checkpoint']), device)
    prior = np.load(args.reference/'validation_inputs.npz')
    base_reference = prior['baseline']
    gate_model = BoundaryPixelRefiner().to(device).eval()
    records, predictions = [], {}
    manifest = {'worker_pid': os.getpid(), 'reference': str(args.reference), 'checkpoint': ref['checkpoint'],
                'limit': args.limit, 'seed': ref['seed'], 'test_images_used': False,
                'scope': 'GT-assisted development diagnostic, historical exposure; NOT model quality or a Dice upper bound',
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__), Path(__file__).with_name('boundary_data.py'),
                                   Path(__file__).with_name('probe_boundary_refinement.py')]}}
    (args.output/'run.json').write_text(json.dumps(manifest, indent=2))
    for i, batch in enumerate(make_loader(loader_args, metadata, 'val')):
        if i >= args.limit:
            break
        identity = {'patch_id': str(batch['patch_id'][0]), 'target_class': int(batch['target_class'][0]),
                    'episode_index': int(batch['episode_index'][0])}
        if identity != ref['manifest']['val'][i]:
            raise RuntimeError('reference episode mismatch')
        gpu = move_batch(batch, device)
        with torch.no_grad():
            out = model(gpu)
            base = out['pixel_probability'].float()
            truth = gpu['pixel_gt'] == gpu['target_class'][:,None,None]
            valid = gpu['pixel_gt'] != metadata['config']['data']['ignore_index']
            if not np.array_equal(truth.cpu().numpy()[0], prior['truth'][i]) or not np.array_equal(valid.cpu().numpy()[0], prior['valid'][i]):
                raise RuntimeError('GT replay mismatch')
            replay_error = float((base - torch.as_tensor(base_reference[i:i+1], device=device)).abs().max())
            q = out['logits'].sigmoid().float()
            rebuilt = torch.einsum('bkhw,bk->bhw', out['assignment'].float(), q)
            projection_error = float((rebuilt - base).abs().max())
            if max(replay_error, projection_error) > 2e-6:
                raise RuntimeError(f'baseline/projection mismatch: {replay_error}, {projection_error}')
            oracle, convergence = fit_coefficients(out['assignment'].float(), truth, valid, q)
            variants = {'baseline': base, 'oracle_fixed_assignment': oracle}
            for size in (64,128,256):
                numerator = F.interpolate((truth & valid)[:,None].float(), size=(size,size), mode='area')
                denominator = F.interpolate(valid[:,None].float(), size=(size,size), mode='area')
                variants[f'oracle_gt_grid_{size}'] = F.interpolate(numerator/denominator.clamp_min(1e-6), size=base.shape[-2:], mode='bilinear', align_corners=False)[:,0]
            variants['closed_set_semantic_argmax'] = (out['semantic_logits'].argmax(1) == gpu['target_class'][:,None,None]).float()
            pred = base >= .5
            gate = gate_model(gpu['image'], base)['gate']
            variants['oracle_gate_unbounded'] = torch.where(gate, truth.float(), base)
            # The previous refiner can change logits only within +/-2, and only inside its gate.
            logit = torch.logit(base.clamp(1e-6,1-1e-6))
            variants['oracle_gate_bounded_2'] = torch.where(gate, (logit + torch.where(truth, 2., -2.)).sigmoid(), base)
            error = ((pred != truth) & valid).cpu().numpy()[0]
            true_edge = binary_boundary(truth, valid).cpu().numpy()[0]
            distance = distance_transform_cdt(~true_edge, metric='chessboard') if true_edge.any() else np.full(error.shape, 100000)
            h,w = error.shape
            yy,xx = np.indices((h,w)); border = np.minimum.reduce([yy,xx,h-1-yy,w-1-xx])
            valid_np = valid.cpu().numpy()[0]
            row = {**identity, 'replay_max_error': replay_error, 'projection_max_error': projection_error,
                   'optimization': convergence, 'error_pixels': int(error.sum()),
                   'valid_pixels': int(valid.sum()), 'gate_pixels': int((gate & valid).sum()),
                   'error_pixels_in_gate': int((error & gate.cpu().numpy()[0]).sum()),
                   'error_within_gt_boundary': {str(r): int((error & (distance <= r)).sum()) for r in (2,4,8,16,32)},
                   'border16_errors': int((error & (border < 16)).sum()),
                   'border16_pixels': int((valid_np & (border < 16)).sum()), 'variants': {}}
            for name, probability in variants.items():
                if not torch.isfinite(probability).all():
                    raise FloatingPointError(name)
                row['variants'][name] = metrics(probability, truth, valid)
                predictions.setdefault(name, []).append(probability.cpu().numpy()[0])
        records.append(row)
        with (args.output/'episodes.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        (args.output/'status.json').write_text(json.dumps({'status':'running','completed':i+1,'total':args.limit,'worker_pid':os.getpid()}))
        if i % 8 == 0:
            print(json.dumps({'completed':i+1,'pid':os.getpid(),'replay_error':replay_error,'oracle':row['variants']['oracle_fixed_assignment']['bf1_2']}), flush=True)
    summary = {'episodes': len(records), 'variants': {}, 'test_used': False, 'oracle_not_model_performance': True}
    for name in predictions:
        # Reuse segmentation aggregation only; no prompt metric is claimed for oracles.
        value = summarize([{**r['variants'][name], 'prompt_violations': 0} for r in records])
        value.pop('prompt_violations')
        summary['variants'][name] = value
    errors = sum(r['error_pixels'] for r in records)
    summary['error_pixels'] = errors
    summary['error_fraction_within_gt_boundary'] = {str(radius): sum(r['error_within_gt_boundary'][str(radius)] for r in records)/max(errors,1) for radius in (2,4,8,16,32)}
    summary['fraction_errors_in_gate'] = sum(r['error_pixels_in_gate'] for r in records)/max(errors,1)
    summary['fraction_valid_pixels_in_gate'] = sum(r['gate_pixels'] for r in records)/sum(r['valid_pixels'] for r in records)
    border_errors = sum(r['border16_errors'] for r in records); border_pixels = sum(r['border16_pixels'] for r in records)
    summary['border16_error_rate'] = border_errors/max(border_pixels,1)
    summary['interior_error_rate'] = (errors-border_errors)/max(sum(r['valid_pixels'] for r in records)-border_pixels,1)
    summary['max_replay_error'] = max(r['replay_max_error'] for r in records)
    summary['max_projected_gradient'] = max(r['optimization']['projected_gradient_max'] for r in records)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(args.output/'diagnostic_predictions.npz', **{k:np.stack(v) for k,v in predictions.items() if k != 'baseline'})
    (args.output/'status.json').write_text(json.dumps({'status':'complete','completed':len(records),'worker_pid':os.getpid()}))
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
