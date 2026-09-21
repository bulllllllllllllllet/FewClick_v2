#!/usr/bin/env python3
"""Paired same-slide nonoverlapping support transfer, no new training."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pyvips
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.tools.train_tissue_prototype_comparison import (
    build_model, PromptEpisodeDataset, collate_prompt_episodes, load_config,
    load_phase2, LabelSourceDataset, project_region_probabilities, pixel_metrics, task_macro, prototype_logits)
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts, transfer_support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp', required=True)
    p.add_argument('--run', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_head_comparison_0922_004524'))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--max-queries', type=int, default=108)
    a = p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', a.timestamp) or a.max_queries < 1:
        raise ValueError('invalid arguments')
    out = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')/('tissue_cross_patch_'+a.timestamp)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    previous = json.loads((a.run/'manifest.json').read_text())
    split = Path(previous['args']['split_dir'])
    spatial_path = split/'spatial_manifest.parquet'
    if hashlib.sha256(spatial_path.read_bytes()).hexdigest() != previous['split_audit']['manifest_sha256']:
        raise ValueError('spatial manifest changed')
    spatial = pd.read_parquet(spatial_path).set_index('patch_id')
    ds = PromptEpisodeDataset(split/'cache_index.parquet', split/'labels_index.parquet', split/'patches_index.parquet',
        'val', 20260922, {'point':.4,'small':.35,'large':.25}, tuple(range(12)), eligibility_index=split/'eligibility_index.parquet')
    rng = np.random.default_rng(20260922)
    pairs, skipped = [], []
    fields = ['context_x0','context_y0','context_x1','context_y1']
    def box(row):
        return [float(row[k]) for k in fields]
    for query in previous['episodes']['val'][:a.max_queries]:
        qi = query['dataset_index']
        if ds.patch_ids[qi] != query['patch_id']:
            raise ValueError('query dataset identity changed')
        qr = spatial.loc[query['patch_id']]
        candidates = np.flatnonzero((ds.wsi_ids == query['wsi_id']) &
            (ds.target_classes == query['target_class']) & (ds.prompt_sizes == query['prompt_size']))
        found = None
        for si in rng.permutation(candidates):
            sr = spatial.loc[ds.patch_ids[si]]
            if qr['split'] != 'val' or sr['split'] != 'val' or sr['source_key'] != qr['source_key']:
                raise ValueError('source or split mismatch')
            if ds.patch_ids[si] != query['patch_id'] and separated_contexts(box(qr), box(sr)):
                found = dict(query=query, support_index=int(si), support_patch_id=ds.patch_ids[si],
                             source_key=qr['source_key'], query_context=box(qr), support_context=box(sr))
                break
        if found is None:
            skipped.append(dict(query=query, reason='no same-slide same-task/size support with full-context gap512'))
        else:
            pairs.append(found)
    if not pairs:
        raise ValueError('no valid pairs; do not silently relax spatial constraints')
    models = {}
    checkpoint_hashes = {}
    for seed in (17,29,43):
        for method in ('phase5','single','multi'):
            path = a.run/f'{method}_seed{seed}_step1000.pth'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['data_hash'] != previous['data_hash']:
                raise ValueError('checkpoint/data mismatch')
            model = build_model(method)
            model.load_state_dict(payload['model'], strict=True)
            models[f'{method}_{seed}'] = model.to(device).eval()
            checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfgroot = Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240')
    cfg = load_config(cfgroot/'config_snapshot.json')
    phase2 = load_phase2(cfg, cfgroot/'checkpoint_epoch_028.pth', device)
    table = pd.read_parquet(split/'patches_index.parquet').set_index('patch_id')
    source = LabelSourceDataset([dict(table.loc[x['query']['patch_id']].to_dict(), patch_id=x['query']['patch_id']) for x in pairs],cfg['data']['class_map'],255)
    manifest = dict(protocol='same-slide cross-patch transfer diagnostic, GT simulated present-target tasks',
        coordinate_policy='support xy remains support-local; query xy remains query-local; Phase5 transfer distribution shift',
        score_policy='both protocols use same query pixels excluding original local prompt regions',
        source_run=str(a.run), checkpoint_sha256=checkpoint_hashes, split_audit=previous['split_audit'],
        skipped=skipped, pairs=pairs, pid=os.getpid(), new_training=False, threshold=.5)
    def batch(item):
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in collate_prompt_episodes([item]).items()}
    results = []
    for i, pair in enumerate(pairs):
        q, s = ds[pair['query']['dataset_index']], ds[pair['support_index']]
        for sign in ('positive','negative'):
            actual = q[sign+'_slot_indices'][q[sign+'_mask']].tolist()
            if actual != pair['query'][sign+'_slots']:
                raise ValueError('original query prompts changed')
        pair['support_annotations'] = {k:s[k].tolist() for k in ('positive_slot_indices','negative_slot_indices','positive_xy','negative_xy','positive_mask','negative_mask')}
        pixel = source[i]
        gt = pixel['mask'].numpy()
        target, valid = gt==int(q['target_class']), np.isin(gt,range(12))
        with torch.inference_mode():
            assignment = phase2(pixel['image'][None].to(device),return_full_assignment=True)['assignment']
            excluded = q['all_prompted_regions'].numpy()[assignment.argmax(1)[0].cpu().numpy()]
            score_mask = valid & ~excluded
            masks = dict(gt=target, valid=score_mask)
            for protocol, item in [('local',q),('cross',transfer_support(q,s))]:
                b = batch(item)
                logits = {name:model(b)['logits'] for name,model in models.items()}
                logits.update({f'identity_{mode}':prototype_logits(b,mode) for mode in ('single','multi')})
                for name, values in logits.items():
                    prob = project_region_probabilities(assignment,values)['pixel_probability'][0].cpu().numpy()
                    if not np.isfinite(prob).all():
                        raise ValueError('nonfinite prediction')
                    masks[f'{protocol}_{name}'] = prob>=.5
                    results.append(dict(episode_id=i, protocol=protocol, model=name, task_id=str(int(q['target_class'])),
                                        **pixel_metrics(prob>=.5,target,score_mask)))
        np.savez_compressed(out/f'episode_{i:04d}.npz',**masks)
        with (out/'episode_metrics.jsonl').open('a') as f:
            for row in results[-22:]: f.write(json.dumps(row)+'\n')
        with (out/'completed_pairs.jsonl').open('a') as f: f.write(json.dumps(pair)+'\n')
        print(json.dumps(dict(event='episode', position=i, total=len(pairs),pid=os.getpid())),flush=True)
    report = {protocol:{name:task_macro([r for r in results if r['protocol']==protocol and r['model']==name])
                       for name in sorted(set(r['model'] for r in results))} for protocol in ('local','cross')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(event='complete', output=str(out), pairs=len(pairs), skipped=len(skipped))),flush=True)


if __name__ == '__main__': main()
