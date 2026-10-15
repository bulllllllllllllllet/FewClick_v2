"""Paired, frozen-J5 boundary refinement development experiment (no test)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary, boundary_f1
from .boundary_data import load_baseline, make_loader, move_batch
from .boundary_refinement import BoundaryPixelRefiner, dilate, refinement_loss


def metrics(probability, truth, valid):
    pred = (probability >= .5) & valid
    truth = truth & valid
    tp = int((pred & truth).sum()); fp = int((pred & ~truth & valid).sum()); fn = int((~pred & truth).sum())
    row = {'dice': 2 * tp / max(2 * tp + fp + fn, 1), 'tp': tp, 'fp': fp, 'fn': fn}
    for radius in (1, 2, 4):
        # Exclude ignore interfaces plus their matching tolerance.
        safe = valid & ~dilate(~valid, radius)
        parts = boundary_f1(pred, truth, safe, radius)
        nt, npred = int(parts['true_boundary_pixels']), int(parts['predicted_boundary_pixels'])
        row[f'bf1_{radius}'] = float(parts['boundary_f1']) if nt and npred else (0.0 if nt or npred else None)
        pe = dilate(binary_boundary(pred, safe), radius) & safe
        te = dilate(binary_boundary(truth, safe), radius) & safe
        union = int((pe | te).sum())
        row[f'boundary_band_iou_{radius}'] = int((pe & te).sum()) / union if union else None
    band = dilate(binary_boundary(truth, valid), 4) & valid & ~dilate(~valid, 4)
    row['gt_boundary_band_accuracy'] = float((pred[band] == truth[band]).float().mean()) if band.any() else None
    return row


def prompt_violations(probability, batch):
    count = 0
    height, width = probability.shape[-2:]
    for prefix, sign in (('positive', True), ('negative', False)):
        xy = batch[f'{prefix}_xy'][0][batch[f'{prefix}_mask'][0]]
        # Same normalized coordinate convention as iterative decoder click_values.
        x = (xy[:, 0].clamp(0,1)*width).long().clamp(max=width-1)
        y = (xy[:, 1].clamp(0,1)*height).long().clamp(max=height-1)
        count += int(((probability[0, y, x] >= .5) != sign).sum())
    return count


def summarize(rows):
    keys = ('dice', 'bf1_1', 'bf1_2', 'bf1_4', 'boundary_band_iou_2', 'boundary_band_iou_4', 'gt_boundary_band_accuracy')
    result = {'episodes': len(rows)}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key+'_evaluable'] = len(values)
    result['micro_dice'] = 2*sum(r['tp'] for r in rows) / max(sum(2*r['tp']+r['fp']+r['fn'] for r in rows), 1)
    result['prompt_violations'] = sum(r['prompt_violations'] for r in rows)
    return result


def evaluate(model, samples, device):
    rows, probabilities = [], []
    for sample in samples:
        image, base = sample['image'].to(device), sample['base'].to(device)
        with torch.no_grad():
            out = model(image, base) if model is not None else None
            probability = out['pixel_probability'] if out else base
        truth, valid = sample['truth'].to(device), sample['valid'].to(device)
        row = metrics(probability, truth, valid)
        row.update(sample['identity'])
        row['prompt_violations'] = prompt_violations(probability.cpu(), sample['batch'])
        row['outside_gate_modified'] = int((probability[~out['gate']] != base[~out['gate']]).sum()) if out else 0
        if row['outside_gate_modified']:
            raise RuntimeError('refinement modified pixels outside inference gate')
        rows.append(row); probabilities.append(probability.cpu().numpy()[0])
    return rows, np.stack(probabilities)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-samples', type=int, default=128)
    p.add_argument('--val-samples', type=int, default=64)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--baseline-root', type=Path, default=Path('/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455'))
    p.add_argument('--spatial-data', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235'))
    p.add_argument('--routing', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/iterative_decoder/spatial_routing_0923_154545'))
    a = p.parse_args()
    if min(a.train_samples, a.val_samples, a.epochs) < 1:
        raise ValueError('counts must be positive')
    root = Path('/nfs-medical3/zyh/fewclick_v2').resolve()
    if root not in a.output.resolve().parents:
        raise ValueError('output must be under FewClick artifact root')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed)
    metadata = json.loads((a.baseline_root/'run_metadata.json').read_text())
    checkpoint = a.baseline_root/'checkpoint_epoch_004.pth'
    device = torch.device(a.device)
    model = load_baseline(metadata, checkpoint, device)
    samples = {}
    ignore = int(metadata['config']['data']['ignore_index'])
    for split in ('train', 'val'):
        samples[split] = []
        for index, batch in enumerate(make_loader(a, metadata, split)):
            gpu_batch = move_batch(batch, device)
            with torch.no_grad():
                output = model(gpu_batch)
            if not torch.isfinite(output['pixel_probability']).all():
                raise FloatingPointError('nonfinite baseline')
            sample = {'image': batch['image'], 'base': output['pixel_probability'].cpu(),
                      'truth': batch['pixel_gt'] == batch['target_class'][:,None,None],
                      'valid': batch['pixel_gt'] != ignore, 'batch': {k:batch[k] for k in ('positive_xy','negative_xy','positive_mask','negative_mask')},
                      'identity': {'patch_id': str(batch['patch_id'][0]), 'target_class': int(batch['target_class'][0]),
                                   'episode_index': int(batch['episode_index'][0])}}
            samples[split].append(sample)
            if index % 10 == 0:
                print(f'prepare {split} {index+1} assignment_low={tuple(output["assignment_low"].shape)} image={tuple(batch["image"].shape)}', flush=True)
    del model
    train_ids = {s['identity']['patch_id'] for s in samples['train']}
    val_ids = {s['identity']['patch_id'] for s in samples['val']}
    if train_ids & val_ids:
        raise RuntimeError('train/val patch overlap')
    run = {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
           'checkpoint': str(checkpoint), 'test_used': False, 'threshold': .5,
           'scope': 'developmental within-cohort spatial validation; historical encoder exposure',
           'manifest': {k:[s['identity'] for s in v] for k,v in samples.items()},
           'validation_prompts': [{prefix:s['batch'][f'{prefix}_xy'][0][s['batch'][f'{prefix}_mask'][0]].tolist()
                                   for prefix in ('positive','negative')} for s in samples['val']],
           'source_sha256': {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), Path(__file__).with_name('boundary_refinement.py'))}}
    (a.output/'run.json').write_text(json.dumps(run,indent=2))
    initial = BoundaryPixelRefiner().to(device)
    rows, probabilities = evaluate(initial, samples['val'], device)
    baseline_rows, baseline_probabilities = evaluate(None, samples['val'], device)
    if not np.allclose(probabilities, baseline_probabilities, atol=1e-6):
        raise RuntimeError('zero initialization changed baseline')
    baseline_summary = summarize(baseline_rows)
    (a.output/'baseline_rows.json').write_text(json.dumps(baseline_rows,indent=2))
    all_summaries = {'baseline': baseline_summary, 'arms': {}}
    for arm, weight in (('pixel', 0.), ('boundary', 1.)):
        refiner = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-4)
        history = []; best = None; best_score = -float('inf')
        for epoch in range(a.epochs):
            refiner.train(); losses = []
            order = np.random.default_rng(a.seed+epoch).permutation(len(samples['train']))
            for index in order:
                s = samples['train'][index]
                out = refiner(s['image'].to(device),s['base'].to(device))
                loss = refinement_loss(out,s['truth'].to(device),s['valid'].to(device),weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                optimizer.zero_grad(); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(),5.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite gradient')
                optimizer.step(); losses.append(float(loss.detach()))
            refiner.eval(); rows, probabilities = evaluate(refiner,samples['val'],device)
            summary = summarize(rows); summary.update(epoch=epoch,train_loss=float(np.mean(losses)))
            eligible = (summary['bf1_2'] is not None and baseline_summary['bf1_2'] is not None
                        and summary['boundary_band_iou_2'] is not None and baseline_summary['boundary_band_iou_2'] is not None
                        and summary['bf1_2'] > baseline_summary['bf1_2']
                        and summary['boundary_band_iou_2'] > baseline_summary['boundary_band_iou_2']
                        and summary['dice'] >= baseline_summary['dice']-.005
                        and summary['micro_dice'] >= baseline_summary['micro_dice']-.005
                        and summary['prompt_violations'] <= baseline_summary['prompt_violations'])
            summary['eligible'] = eligible
            history.append(summary)
            print(arm,json.dumps(summary),flush=True)
            torch.save({'model':refiner.state_dict(),'epoch':epoch,'summary':summary,'run':run},a.output/f'{arm}_epoch_{epoch:03}.pth')
            (a.output/f'{arm}_epoch_{epoch:03}_rows.json').write_text(json.dumps(rows,indent=2))
            score = summary['bf1_2'] if summary['bf1_2'] is not None else -float('inf')
            if score > best_score:
                best_score = score; best = {'epoch':epoch,'summary':summary}
                np.savez_compressed(a.output/f'{arm}_best_predictions.npz',probability=probabilities)
        # Strict round-trip on a real example.
        loaded = BoundaryPixelRefiner().to(device)
        loaded.load_state_dict(torch.load(a.output/f'{arm}_epoch_{a.epochs-1:03}.pth',map_location=device,weights_only=False)['model'],strict=True)
        s=samples['val'][0]
        with torch.no_grad():
            error = (loaded(s['image'].to(device),s['base'].to(device))['pixel_probability']-refiner(s['image'].to(device),s['base'].to(device))['pixel_probability']).abs().max().item()
        if error != 0:
            raise RuntimeError('checkpoint round-trip mismatch')
        all_summaries['arms'][arm] = {'history':history,'best_boundary':best,'eligible_epochs':[h['epoch'] for h in history if h['eligible']], 'reload_max_error':error}
    np.savez_compressed(a.output/'validation_inputs.npz',image=np.concatenate([s['image'].numpy() for s in samples['val']]),
                        truth=np.concatenate([s['truth'].numpy() for s in samples['val']]),valid=np.concatenate([s['valid'].numpy() for s in samples['val']]),baseline=baseline_probabilities)
    (a.output/'summary.json').write_text(json.dumps(all_summaries,indent=2))
    (a.output/'status.json').write_text(json.dumps({'status':'complete','test_used':False}))
    print('COMPLETE',json.dumps(all_summaries),flush=True)


if __name__ == '__main__':
    main()
