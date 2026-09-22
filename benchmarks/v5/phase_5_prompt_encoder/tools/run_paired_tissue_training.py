#!/usr/bin/env python3
"""Controlled local/cross training with identical queries, order and initial states."""
import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyvips
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.tools.train_tissue_prototype_comparison import (
    build_model, PromptEpisodeDataset, collate_prompt_episodes, load_config, load_phase2,
    LabelSourceDataset, project_region_probabilities, pixel_metrics, task_macro)
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import transfer_support, separated_contexts
from benchmarks.v5.phase_5_prompt_encoder.src.learnable_tissue_prototype import unprompted_bce
from benchmarks.v5.phase_5_prompt_encoder.src.paired_training import support_counts, shuffled_order, state_hash

ROOT = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')
SPLIT = ROOT/'tissue_spatial_split_0922_001235'
FIXED = ROOT/'tissue_cross_patch_0922_005516/manifest.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp', required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--per-stratum', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', default=[17,29,43])
    p.add_argument('--val-limit', type=int, default=70)
    p.add_argument('--device', default='cuda:4')
    a = p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', a.timestamp) or min(a.steps,a.per_stratum,a.batch_size,a.val_limit)<1 or len(a.seeds)!=len(set(a.seeds)):
        raise ValueError('invalid configuration')
    out = ROOT/('paired_tissue_training_'+a.timestamp)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    def log(event):
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    log(dict(event='start',pid=os.getpid(),args=vars(a)))
    fixed = json.loads(FIXED.read_text())
    split_hash = hashlib.sha256((SPLIT/'spatial_manifest.parquet').read_bytes()).hexdigest()
    if split_hash != fixed['split_audit']['manifest_sha256']:
        raise ValueError('spatial manifest changed')
    spatial = pd.read_parquet(SPLIT/'spatial_manifest.parquet').set_index('patch_id')
    spatial_rows = spatial.to_dict('index')
    def box(row): return [row[k] for k in ('context_x0','context_y0','context_x1','context_y1')]
    def check_geometry(qid, sid, split):
        q, s = spatial_rows[qid], spatial_rows[sid]
        return (qid!=sid and q['split']==s['split']==split and q['source_key']==s['source_key']
                and separated_contexts(box(q),box(s)))
    def dataset(split):
        return PromptEpisodeDataset(SPLIT/'cache_index.parquet',SPLIT/'labels_index.parquet',SPLIT/'patches_index.parquet',
            split,20260922,{'point':.4,'small':.35,'large':.25},tuple(range(12)),eligibility_index=SPLIT/'eligibility_index.parquet')
    train, val = dataset('train'), dataset('val')
    # Cache real loader items once; all models and arms consume identical tensors.
    cache = {}
    def get(ds, index):
        key = ('train' if ds is train else 'val', int(index))
        if key not in cache: cache[key] = ds[int(index)]
        return cache[key]
    by_slide = defaultdict(list)
    for i, (w,c,z) in enumerate(zip(train.wsi_ids,train.target_classes,train.prompt_sizes)):
        by_slide[(w,int(c),z)].append(i)
    rng = np.random.default_rng(20260922)
    train_pairs = []
    rejected = Counter()
    for cls in range(12):
        for size in ('point','small','large'):
            candidates = np.flatnonzero((train.target_classes==cls)&(train.prompt_sizes==size))
            selected = 0
            for qi in rng.permutation(candidates):
                qi = int(qi)
                q = get(train,qi)
                for si in rng.permutation(by_slide[(train.wsi_ids[qi],cls,size)]):
                    si = int(si)
                    if not check_geometry(q['patch_id'],train.patch_ids[si],'train'): continue
                    s = get(train,si)
                    if support_counts(q)!=support_counts(s):
                        rejected['support_count_mismatch']+=1
                        continue
                    valid = (q['binary_target']!=255)&q['fine_active'].bool()&~q['all_prompted_regions'].bool()
                    if not valid.any(): raise ValueError('no valid query supervision')
                    train_pairs.append((qi,si))
                    selected+=1
                    break
                else:
                    rejected['query_without_pair']+=1
                    continue
                if selected==a.per_stratum: break
            if selected != a.per_stratum:
                raise ValueError(f'insufficient exact-count pairs: {cls}/{size}: {selected}')
            log(dict(event='stratum',cls=cls,size=size,selected=selected))
    val_pairs = []
    excluded = []
    for original_id, pair in enumerate(fixed['pairs']):
        qi,si = int(pair['query']['dataset_index']),int(pair['support_index'])
        q,s = get(val,qi),get(val,si)
        if q['patch_id']!=pair['query']['patch_id'] or s['patch_id']!=pair['support_patch_id'] or not check_geometry(q['patch_id'],s['patch_id'],'val'):
            raise ValueError('frozen validation identity/geometry mismatch')
        for sign in ('positive','negative'):
            if q[sign+'_slot_indices'][q[sign+'_mask']].tolist()!=pair['query'][sign+'_slots']:
                raise ValueError('query prompts changed')
            if s[sign+'_slot_indices'].tolist()!=pair['support_annotations'][sign+'_slot_indices']:
                raise ValueError('support prompts changed')
        if support_counts(q)==support_counts(s): val_pairs.append((qi,si,original_id))
        else: excluded.append(original_id)
    if len(val_pairs)!=70: raise ValueError('expected frozen exact-count subset of 70')
    val_pairs=val_pairs[:a.val_limit]
    train_ids={train.patch_ids[i] for pair in train_pairs for i in pair}
    val_ids={val.patch_ids[i] for pair in val_pairs for i in pair[:2]}
    if train_ids & val_ids: raise ValueError('train/val patch overlap')
    def describe(ds,qi,si):
        def item(i):
            x=get(ds,i); r=spatial_rows[x['patch_id']]
            return dict(dataset_index=i,patch_id=x['patch_id'],wsi_id=x['wsi_id'],case_id=r['patient_id'],
                task=int(x['target_class']),prompt_size=x['prompt_size'],counts=support_counts(x),
                source_key=r['source_key'],context=box(r),
                **{k:x[k].tolist() for k in ('positive_slot_indices','negative_slot_indices','positive_xy','negative_xy','positive_mask','negative_mask')})
        return dict(query=item(qi),support=item(si))
    order=shuffled_order(len(train_pairs),a.steps*a.batch_size)
    manifest=dict(args=vars(a),pid=os.getpid(),split_audit=fixed['split_audit'],
        frozen_val_source=str(FIXED),frozen_val_sha256=hashlib.sha256(FIXED.read_bytes()).hexdigest(),
        train_pairs=[describe(train,*pair) for pair in train_pairs],
        val_pairs=[dict(original_episode_id=orig,**describe(val,qi,si)) for qi,si,orig in val_pairs],
        excluded_count_mismatch=excluded,rejected_training_candidates=dict(rejected),order=order,
        initialization={},scope='within-cohort seen tissues, historical encoder exposure, simulated region supports',
        loss='same unprompted query regions for both arms, per-episode BCE; support-local xy',
        budget='same query/order/update/optimizer/actual support counts; patch input content differs by treatment')
    manifest['data_hash']=hashlib.sha256(json.dumps([manifest['train_pairs'],manifest['val_pairs'],order],sort_keys=True).encode()).hexdigest()
    manifest['code_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),Path('benchmarks/v5/phase_5_prompt_encoder/src/paired_training.py'))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    log(dict(event='ready',train_pairs=len(train_pairs),val_pairs=len(val_pairs),data_hash=manifest['data_hash']))
    def batch(items): return {k:v.to(device) if torch.is_tensor(v) else v for k,v in collate_prompt_episodes(items).items()}
    models={}
    for seed in a.seeds:
        for method in ('phase5','single','multi'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            initial=build_model(method)
            initial_hash=state_hash(initial)
            for arm in ('local','cross'):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model=copy.deepcopy(initial).to(device).train()
                if state_hash(model)!=initial_hash: raise ValueError('different initialization')
                name=f'{method}_{seed}_{arm}'
                manifest['initialization'][name]=initial_hash
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                for step in range(a.steps):
                    selected=[]
                    for index in order[step*a.batch_size:(step+1)*a.batch_size]:
                        qi,si=train_pairs[index]; q,s=get(train,qi),get(train,si)
                        selected.append(q if arm=='local' else transfer_support(q,s))
                    b=batch(selected)
                    optimizer.zero_grad(set_to_none=True)
                    loss=unprompted_bce(model(b)['logits'],b)
                    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    log(dict(event='step',name=name,step=step+1,loss=float(loss.detach())))
                model.eval()
                probe=batch([get(train,train_pairs[0][0])])
                with torch.no_grad(): expected=model(probe)['logits'].cpu()
                path=out/f'{name}_step{a.steps}.pth'
                torch.save(dict(model=model.cpu().state_dict(),optimizer=optimizer.state_dict(),
                    data_hash=manifest['data_hash'],initialization_hash=initial_hash,method=method,seed=seed,arm=arm,steps=a.steps),path)
                restored=build_model(method).to(device).eval()
                payload=torch.load(path,map_location=device,weights_only=False)
                if payload['data_hash']!=manifest['data_hash']: raise ValueError('checkpoint provenance')
                restored.load_state_dict(payload['model'],strict=True)
                with torch.no_grad(): torch.testing.assert_close(expected,restored(probe)['logits'].cpu(),rtol=1e-5,atol=1e-6)
                models[name]=restored.cpu()
                log(dict(event='checkpoint_verified',name=name,hash=hashlib.sha256(path.read_bytes()).hexdigest()))
                del model,optimizer,payload
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cfgroot=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240')
    cfg=load_config(cfgroot/'config_snapshot.json')
    phase2=load_phase2(cfg,cfgroot/'checkpoint_epoch_028.pth',device)
    table=pd.read_parquet(SPLIT/'patches_index.parquet').set_index('patch_id')
    source=LabelSourceDataset([dict(table.loc[val.patch_ids[qi]].to_dict(),patch_id=val.patch_ids[qi]) for qi,_,_ in val_pairs],cfg['data']['class_map'],255)
    results=[]
    for i,(qi,si,original_id) in enumerate(val_pairs):
        q,s=get(val,qi),get(val,si); pixel=source[i]; gt=pixel['mask'].numpy()
        with torch.inference_mode():
            assignment=phase2(pixel['image'][None].to(device),return_full_assignment=True)['assignment']
            excluded_mask=q['all_prompted_regions'].numpy()[assignment.argmax(1)[0].cpu().numpy()]
            target=gt==int(q['target_class']); valid=np.isin(gt,range(12))&~excluded_mask
            masks=dict(gt=target,valid=valid)
            episode_rows=[]
            for name,model in models.items():
                method,seed,arm=name.split('_'); model.to(device)
                for protocol,item in (('local',q),('cross',transfer_support(q,s))):
                    logits=model(batch([item]))['logits']
                    prob=project_region_probabilities(assignment,logits)['pixel_probability'][0].cpu().numpy()
                    if not np.isfinite(prob).all(): raise ValueError('nonfinite prediction')
                    pred=prob>=.5; masks[f'{name}_{protocol}']=pred
                    metric=pixel_metrics(pred,target,valid)
                    episode_rows.append(dict(episode_id=i,original_episode_id=original_id,name=name,method=method,seed=int(seed),
                        arm=arm,protocol=protocol,task_id=str(int(q['target_class'])),**metric))
                model.cpu()
        mask_path=out/f'episode_{i:04d}.npz'
        np.savez_compressed(mask_path,**masks)
        with np.load(mask_path) as saved:
            for row in episode_rows:
                actual=pixel_metrics(saved[f"{row['name']}_{row['protocol']}"],saved['gt'],saved['valid'])
                if any(row[k]!=v for k,v in actual.items()): raise ValueError('saved-mask metric mismatch')
        with (out/'episode_metrics.jsonl').open('a') as f:
            for row in episode_rows: f.write(json.dumps(row)+'\n')
        results.extend(episode_rows)
        log(dict(event='val_episode',episode=i,total=len(val_pairs),verified_rows=len(episode_rows)))
    report={pr:{name:task_macro([r for r in results if r['protocol']==pr and r['name']==name]) for name in models} for pr in ('local','cross')}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    summary=[]
    for method in ('phase5','single','multi'):
        for pr in ('local','cross'):
            values={arm:[report[pr][f'{method}_{seed}_{arm}']['task_macro_dice'] for seed in a.seeds] for arm in ('local','cross')}
            summary.append(dict(method=method,evaluation=pr,by_seed=values,
                local_training_mean=float(np.mean(values['local'])),cross_training_mean=float(np.mean(values['cross'])),
                paired_delta_mean=float(np.mean(np.array(values['cross'])-values['local']))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    log(dict(event='complete',output=str(out),verified_metric_rows=len(results),summary=summary))


if __name__=='__main__': main()
