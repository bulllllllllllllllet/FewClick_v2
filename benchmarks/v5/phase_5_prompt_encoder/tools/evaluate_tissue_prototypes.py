#!/usr/bin/env python3
"""Paired known-tissue diagnostic: frozen Phase-5 versus support prototypes.

Reuses historical validation ONLY as a compatibility diagnostic. Region support
sets are not clicks, and historical pretrained classes are not unseen classes.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pyvips  # Image libraries must be initialized before torch.
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import load_phase2
from benchmarks.v4.phase_4_cross_scale.tools.build_fine_region_labels import LabelSourceDataset
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset, collate_prompt_episodes
from benchmarks.v4.phase_6_mask_decoder.src.model import load_phase5_prompt_model, project_region_probabilities
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import pixel_metrics, prototype_logits, task_macro
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_tasks import tasks_from_class_map

ROOT = Path('/nfs-medical3/zyh/v4')
RUN = ROOT/'phase5/runs/phase5_prompt_encoder_20260721_130650_formal'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--per-task-size', type=int, default=1)
    p.add_argument('--max-episodes', type=int, default=36)
    a = p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', a.timestamp) or min(a.per_task_size, a.max_episodes) < 1:
        raise ValueError('timestamp and positive budgets required')
    out = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')/('tissue_prototype_eval_'+a.timestamp)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    cache = ROOT/'phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet'
    labels = ROOT/'phase4/data/fine_region_labels_20260719_230128/label_index.parquet'
    patches = ROOT/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet'
    eligibility = ROOT/'phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet'
    ds = PromptEpisodeDataset(cache, labels, patches, 'val', 20260921,
        {'point': .4, 'small': .35, 'large': .25}, tuple(range(12)), eligibility_index=eligibility)
    indices = []
    selection_rng = np.random.default_rng(20260921)
    for cls in range(12):
        for size in ('point', 'small', 'large'):
            candidates = np.flatnonzero((ds.target_classes == cls) & (ds.prompt_sizes == size))
            indices.extend(selection_rng.choice(candidates, size=min(a.per_task_size, len(candidates)), replace=False).tolist())
    indices = indices[:a.max_episodes]
    if not indices: raise ValueError('no eligible tasks')
    cfgpath = ROOT/'phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json'
    p2path = cfgpath.parent/'checkpoint_epoch_028.pth'
    cfg = load_config(cfgpath)
    tasks = {int(c['id']): task for c, task in zip(cfg['data']['class_map'], tasks_from_class_map(cfg['data']['class_map']))}
    p5path = Path(json.loads((RUN/'best_checkpoint.json').read_text())['path'])
    model, _ = load_phase5_prompt_model(str(p5path), 256, 4, 2, .1)
    model = model.to(device).eval()
    phase2 = load_phase2(cfg, p2path, device)
    table = pd.read_parquet(patches).set_index('patch_id')
    source_rows = []
    for i in indices:
        row = table.loc[ds.patch_ids[i]].to_dict()
        row.update(patch_id=ds.patch_ids[i], split='val')
        source_rows.append(row)
    source = LabelSourceDataset(source_rows, cfg['data']['class_map'], 255)
    manifest = {'schema': 'known_tissue_region_support_diagnostic_v1',
        'training_ready': False, 'split': 'historical_val', 'new_split_created': False,
        'pretraining_class_exposure': 'all 12 classes; no novel-class claim',
        'prompt_protocol': 'GT-simulated region support; size name point is NOT a pixel click',
        'selection': 'seed 20260921, uniform within each tissue/prompt-size stratum',
        'threshold': .5, 'prototype_temperature': .1,
        'baseline_scope': 'frozen Phase5 matcher, common soft projection, no Phase6 graph',
        'task_definitions': {str(k): dict(task_id=t.task_id, positive_ids=t.positive_ids, negative_ids=t.negative_ids) for k,t in tasks.items()},
        'sources': {k: str(v) for k, v in dict(cache=cache, labels=labels, patches=patches,
                    eligibility=eligibility, phase2=p2path, phase5=p5path).items()}, 'episodes': []}
    results = []
    for position, index in enumerate(indices):
        item = ds[index]
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collate_prompt_episodes([item]).items()}
        pixel = source[position]
        gt = pixel['mask'].numpy()
        task = tasks[int(item['target_class'])]
        target, valid = task.masks(gt)
        with torch.inference_mode():
            assignment = phase2(pixel['image'][None].to(device), return_full_assignment=True)['assignment']
            if assignment.shape[1] != batch['fine_tokens'].shape[1]:
                raise ValueError('cached token / assignment slot count mismatch')
            owner = assignment.argmax(1)[0].cpu().numpy()
            excluded = item['all_prompted_regions'].numpy()[owner]
            predictions = {'phase5': model(batch)['logits'],
                           'single': prototype_logits(batch, 'single'),
                           'multi': prototype_logits(batch, 'multi')}
            identity = f'{position:04d}'
            masks = {'gt': target, 'valid': valid, 'prompt_excluded': excluded}
            for method, logits in predictions.items():
                prob = project_region_probabilities(assignment, logits)['pixel_probability'][0].cpu().numpy()
                if not np.isfinite(prob).all(): raise ValueError('non-finite predictions')
                masks[method] = prob >= .5
                for scope, mask in [('all', valid), ('unprompted', valid & ~excluded)]:
                    results.append(dict(episode_id=identity, method=method, scope=scope,
                        task_id=task.task_id, prompt_size=item['prompt_size'],
                        **pixel_metrics(prob >= .5, target, mask)))
        np.savez_compressed(out/f'episode_{identity}.npz', **masks)
        rgb = pixel['image'].numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])
        Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)).save(out/f'episode_{identity}_he.png')
        entry = dict(episode_id=identity, dataset_index=index, patch_id=item['patch_id'],
            wsi_id=item['wsi_id'], target_class=int(item['target_class']), prompt_size=item['prompt_size'],
            positive_slots=item['positive_slot_indices'][item['positive_mask']].tolist(),
            negative_slots=item['negative_slot_indices'][item['negative_mask']].tolist(),
            positive_xy=item['positive_xy'][item['positive_mask']].tolist(),
            negative_xy=item['negative_xy'][item['negative_mask']].tolist(),
            source_box_level0=ds.patch_boxes[index].tolist())
        manifest['episodes'].append(entry)
        # Persist progress; a failed later slide does not erase completed evidence.
        with (out/'completed.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
        print(json.dumps({'event': 'episode_done', **entry}), flush=True)
    report = {'status': 'complete', 'episodes': len(indices), 'results': {}}
    for method in ('phase5', 'single', 'multi'):
        report['results'][method] = {}
        for scope in ('all', 'unprompted'):
            values = [r for r in results if r['method']==method and r['scope']==scope]
            report['results'][method][scope] = task_macro(values)
            report['results'][method][scope]['foreground'] = task_macro([r for r in values if r['task_id'] != 'tissue.background'])
            report['results'][method][scope]['by_prompt_size'] = {size: task_macro([r for r in values if r['prompt_size']==size]) for size in ('point','small','large')}
    manifest['episode_hash'] = hashlib.sha256(json.dumps(manifest['episodes'],sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'episode_metrics.json').write_text(json.dumps(results, indent=2))
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event':'complete','output':str(out),'report':report}), flush=True)


if __name__ == '__main__': main()
