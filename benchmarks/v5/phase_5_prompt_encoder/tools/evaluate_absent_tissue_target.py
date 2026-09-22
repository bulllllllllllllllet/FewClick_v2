#!/usr/bin/env python3
"""Fixed-threshold rejection check: prompted tissue absent from query annotation."""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyvips
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.tools.train_tissue_prototype_comparison import (
    build_model, PromptEpisodeDataset, collate_prompt_episodes, load_config, load_phase2,
    LabelSourceDataset, project_region_probabilities)
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts, transfer_support
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import absent_target_metrics

ROOT = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')
SPLIT = ROOT/'tissue_spatial_split_0922_001235'
RUN = ROOT/'paired_tissue_training_0922_130300'
NAMES = ('tumor_epithelium','tumor_stroma','background','necrosis','normal_gland','normal_stroma',
         'submucosa_serosa','muscle','lymphocyte_aggregate','mucus','fat','blood')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp',required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--max-pairs',type=int,default=70)
    a=p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp) or a.max_pairs<1: raise ValueError('invalid arguments')
    out=ROOT/('absent_tissue_target_'+a.timestamp); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); device=torch.device(a.device)
    run=json.loads((RUN/'manifest.json').read_text())
    spatial=pd.read_parquet(SPLIT/'spatial_manifest.parquet').set_index('patch_id')
    if hashlib.sha256((SPLIT/'spatial_manifest.parquet').read_bytes()).hexdigest()!=run['split_audit']['manifest_sha256']:
        raise ValueError('spatial manifest changed')
    ds=PromptEpisodeDataset(SPLIT/'cache_index.parquet',SPLIT/'labels_index.parquet',SPLIT/'patches_index.parquet',
        'val',20260922,{'point':.4,'small':.35,'large':.25},tuple(range(12)),eligibility_index=SPLIT/'eligibility_index.parquet')
    labels={}
    def label(index):
        index=int(index)
        if index not in labels: labels[index]=np.load(ds.label_paths[index])
        return labels[index]
    def box(pid):
        r=spatial.loc[pid]; return [float(r[k]) for k in ('context_x0','context_y0','context_x1','context_y1')]
    by_wsi=defaultdict(list)
    for i,wsi in enumerate(ds.wsi_ids): by_wsi[wsi].append(i)
    pairs=[]; skipped=[]; rng=np.random.default_rng(20260924)
    for support_row in run['val_pairs']:
        support_index=int(support_row['support']['dataset_index']); support=ds[support_index]
        target=int(support_row['query']['task']); candidates=[]
        for query_index in by_wsi[support['wsi_id']]:
            if query_index==support_index or target in label(query_index): continue
            if separated_contexts(box(support['patch_id']),box(ds.patch_ids[query_index])): candidates.append(query_index)
        if candidates:
            query_index=int(rng.choice(candidates))
            pairs.append(dict(target_class=target,target_name=NAMES[target],support_index=support_index,
                support_patch_id=support['patch_id'],query_index=query_index,query_patch_id=ds.patch_ids[query_index],
                wsi_id=support['wsi_id'],support_context=box(support['patch_id']),query_context=box(ds.patch_ids[query_index])))
        else: skipped.append(dict(target_class=target,support_patch_id=support['patch_id'],reason='no same-slide absent target with context gap'))
        if len(pairs)>=a.max_pairs: break
    if not pairs: raise ValueError('no absent-target pairs; constraints were not relaxed')
    models={}
    for seed in (17,29,43):
        path=RUN/f'multi_{seed}_mixed_step1000.pth'; payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['data_hash']!=run['data_hash'] or payload['arm']!='mixed': raise ValueError('checkpoint provenance mismatch')
        model=build_model('multi'); model.load_state_dict(payload['model'],strict=True); models[seed]=model.to(device).eval()
    cfgroot=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240')
    cfg=load_config(cfgroot/'config_snapshot.json'); phase2=load_phase2(cfg,cfgroot/'checkpoint_epoch_028.pth',device)
    table=pd.read_parquet(SPLIT/'patches_index.parquet').set_index('patch_id')
    source=LabelSourceDataset([dict(table.loc[pair['query_patch_id']].to_dict(),patch_id=pair['query_patch_id']) for pair in pairs],cfg['data']['class_map'],255)
    def batch(item): return {k:v.to(device) if torch.is_tensor(v) else v for k,v in collate_prompt_episodes([item]).items()}
    rows=[]
    for position,pair in enumerate(pairs):
        support,query=ds[pair['support_index']],ds[pair['query_index']]
        if pair['target_class'] in label(pair['query_index']): raise ValueError('target unexpectedly present')
        item=transfer_support(query,support); pixel=source[position]; gt=pixel['mask'].numpy(); valid=np.isin(gt,range(12))
        with torch.inference_mode():
            assignment=phase2(pixel['image'][None].to(device),return_full_assignment=True)['assignment']
            masks=dict(valid=valid)
            for seed,model in models.items():
                prob=project_region_probabilities(assignment,model(batch(item))['logits'])['pixel_probability'][0].cpu().numpy()
                pred=prob>=.5
                masks[f'multi_mixed_{seed}']=pred
                rows.append(dict(episode_id=position,seed=seed,**pair,**absent_target_metrics(pred,valid)))
        np.savez_compressed(out/f'episode_{position:04d}.npz',**masks)
        with (out/'metrics.jsonl').open('a') as f:
            for row in rows[-3:]: f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='episode',position=position,total=len(pairs),pid=os.getpid())),flush=True)
    by_task=defaultdict(list)
    for row in rows: by_task[row['target_name']].append(row)
    by_seed={str(s):dict(activation_rate=float(np.mean([r['activated'] for r in rows if r['seed']==s])),
        false_positive_area=float(np.mean([r['false_positive_area'] for r in rows if r['seed']==s]))) for s in models}
    report=dict(protocol='same-slide spatially separated absent-target query, fixed threshold=0.5',
        source_run=str(RUN),threshold=.5,pairs=pairs,skipped=skipped,
        total_pairs=len(pairs),metric_rows=len(rows),
        aggregate=dict(activation_rate=float(np.mean([r['activated'] for r in rows])),
                       mean_false_positive_area=float(np.mean([r['false_positive_area'] for r in rows])),
                       by_seed=by_seed),
        by_target={name:dict(pairs=len(values)//3,activation_rate=float(np.mean([r['activated'] for r in values])),
              false_positive_area=float(np.mean([r['false_positive_area'] for r in values]))) for name,values in sorted(by_task.items())})
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(event='complete',output=str(out),pairs=len(pairs),skipped=len(skipped),aggregate=report['aggregate'])),flush=True)

if __name__=='__main__': main()
