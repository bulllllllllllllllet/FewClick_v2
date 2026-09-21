#!/usr/bin/env python3
"""Equal-update internal head comparison, with fresh heads and fixed data seeds."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pyvips  # initialize image libraries before torch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import load_phase2
from benchmarks.v4.phase_4_cross_scale.tools.build_fine_region_labels import LabelSourceDataset
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset, collate_prompt_episodes
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import project_region_probabilities
from benchmarks.v5.phase_5_prompt_encoder.src.learnable_tissue_prototype import LearnableTissuePrototype, unprompted_bce
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import pixel_metrics, task_macro


def build_model(method):
    return PromptRegionModel() if method == 'phase5' else LearnableTissuePrototype(method)


def stratified_indices(ds, per_stratum, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in range(12):
        for size in ('point', 'small', 'large'):
            candidates = np.flatnonzero((ds.target_classes == cls) & (ds.prompt_sizes == size))
            if len(candidates) < per_stratum:
                raise ValueError('insufficient stratum')
            selected.extend(rng.choice(candidates, per_stratum, replace=False).tolist())
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp', required=True)
    p.add_argument('--split-dir', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235'))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--train-per-stratum', type=int, default=8)
    p.add_argument('--val-per-stratum', type=int, default=1)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    a = p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', a.timestamp) or min(a.steps, a.batch_size, a.train_per_stratum, a.val_per_stratum) < 1:
        raise ValueError('invalid arguments')
    out = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')/('tissue_head_comparison_'+a.timestamp)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    split = a.split_dir
    audit = json.loads((split/'audit.json').read_text())
    actual_hash = hashlib.sha256((split/'spatial_manifest.parquet').read_bytes()).hexdigest()
    if actual_hash != audit['manifest_sha256']:
        raise ValueError('split hash mismatch')
    datasets = {s: PromptEpisodeDataset(split/'cache_index.parquet', split/'labels_index.parquet',
        split/'patches_index.parquet', s, 20260922, {'point':.4, 'small':.35, 'large':.25},
        tuple(range(12)), eligibility_index=split/'eligibility_index.parquet') for s in ('train', 'val')}
    chosen = {s: stratified_indices(ds, a.train_per_stratum if s == 'train' else a.val_per_stratum, 20260922)
              for s, ds in datasets.items()}
    items = {s: [ds[i] for i in chosen[s]] for s, ds in datasets.items()}
    if set(x['patch_id'] for x in items['train']) & set(x['patch_id'] for x in items['val']):
        raise ValueError('patch leakage')
    rng = np.random.default_rng(20260922)
    order = []
    while len(order) < a.steps * a.batch_size:
        order.extend(rng.permutation(len(items['train'])).tolist())
    order = order[:a.steps*a.batch_size]
    episodes = {s: [dict(dataset_index=i, patch_id=x['patch_id'], wsi_id=x['wsi_id'],
        target_class=int(x['target_class']), prompt_size=x['prompt_size'],
        positive_xy=x['positive_xy'][x['positive_mask']].tolist(),
        negative_xy=x['negative_xy'][x['negative_mask']].tolist(),
        positive_slots=x['positive_slot_indices'][x['positive_mask']].tolist(),
        negative_slots=x['negative_slot_indices'][x['negative_mask']].tolist())
        for i,x in zip(chosen[s], items[s])] for s in items}
    manifest = dict(args={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        split_audit=audit, episodes=episodes, training_order=order,
        data_hash=hashlib.sha256(json.dumps(episodes,sort_keys=True).encode()).hexdigest(),
        protocol='GT simulated region supports, same-patch unprompted queries; NOT pixel clicks or independent test',
        budget='equal updates, batches, lr=0.001, AdamW weight_decay=0.0001; not equal parameters or FLOPs',
        initialization='fresh task heads; frozen historically exposed Phase2/3 cache',
        threshold=.5, pid=os.getpid())
    manifest['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    def log(event):
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    log(dict(event='ready', pid=os.getpid(), output=str(out)))
    def batch(xs):
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in collate_prompt_episodes(xs).items()}
    models = {}
    for seed in a.seeds:
        for method in ('phase5', 'single', 'multi'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = build_model(method).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            start = time.monotonic()
            for step in range(a.steps):
                b = batch([items['train'][i] for i in order[step*a.batch_size:(step+1)*a.batch_size]])
                optimizer.zero_grad(set_to_none=True)
                loss = unprompted_bce(model(b)['logits'], b)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                log(dict(event='step', seed=seed, method=method, step=step+1, loss=float(loss.detach()), elapsed=time.monotonic()-start))
            model.eval()
            probe = batch(items['train'][:1])
            with torch.no_grad():
                reference = model(probe)['logits'].cpu()
            path = out/f'{method}_seed{seed}_final.pth'
            torch.save(dict(model=model.cpu().state_dict(), method=method, seed=seed, steps=a.steps,
                            optimizer=optimizer.state_dict(), data_hash=manifest['data_hash']), path)
            restored = build_model(method).to(device)
            restored.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'], strict=True)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(reference, restored(probe)['logits'].cpu(), rtol=1e-5, atol=1e-6)
            models[(seed, method)] = restored.cpu()
            log(dict(event='checkpoint_verified', seed=seed, method=method, parameters=sum(p.numel() for p in model.parameters())))
            del optimizer, model, restored
    root = Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240')
    cfg = load_config(root/'config_snapshot.json')
    phase2 = load_phase2(cfg, root/'checkpoint_epoch_028.pth', device)
    table = pd.read_parquet(split/'patches_index.parquet').set_index('patch_id')
    rows = [dict(table.loc[x['patch_id']].to_dict(), patch_id=x['patch_id']) for x in items['val']]
    source = LabelSourceDataset(rows, cfg['data']['class_map'], 255)
    results = []
    for position, item in enumerate(items['val']):
        pixel = source[position]
        gt = pixel['mask'].numpy()
        valid = np.isin(gt, list(range(12)))
        target = gt == int(item['target_class'])
        b = batch([item])
        with torch.inference_mode():
            assignment = phase2(pixel['image'][None].to(device), return_full_assignment=True)['assignment']
            excluded = item['all_prompted_regions'].numpy()[assignment.argmax(1)[0].cpu().numpy()]
            masks = dict(gt=target, valid=valid, prompt_excluded=excluded)
            for (seed, method), model in models.items():
                model.to(device)
                prob = project_region_probabilities(assignment, model(b)['logits'])['pixel_probability'][0].cpu().numpy()
                model.cpu()
                if not np.isfinite(prob).all():
                    raise ValueError('nonfinite prediction')
                masks[f'{method}_{seed}'] = prob >= .5
                for scope, mask in [('all', valid), ('unprompted', valid & ~excluded)]:
                    results.append(dict(episode_id=position, seed=seed, method=method, scope=scope,
                        task_id=str(int(item['target_class'])), prompt_size=item['prompt_size'], **pixel_metrics(prob >= .5, target, mask)))
        np.savez_compressed(out/f'val_{position:04d}.npz', **masks)
        log(dict(event='val_episode', episode=position, patch_id=item['patch_id']))
    report = {f'{method}_{seed}': {scope: task_macro([r for r in results if r['seed']==seed and r['method']==method and r['scope']==scope])
              for scope in ('all','unprompted')} for seed, method in models}
    for seed, method in models:
        for scope in ('all', 'unprompted'):
            values = [r for r in results if r['seed']==seed and r['method']==method and r['scope']==scope]
            report[f'{method}_{seed}'][scope]['foreground'] = task_macro([r for r in values if r['task_id'] != '2'])
    (out/'episode_metrics.json').write_text(json.dumps(results, indent=2))
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    log(dict(event='complete', output=str(out)))


if __name__ == '__main__':
    main()
