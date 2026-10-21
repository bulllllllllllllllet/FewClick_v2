"""Legacy-input compatibility check; preserved points have region-centroid semantics.

The centroid-to-pixel rows intentionally measure compatibility with historical
coordinates. They are not a same-semantics comparison of real pixel clicks.
Use diagnose_regression.py for separately locked, matched interior-click tasks.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import numpy as np
import pandas as pd
import torch

from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import collate_joint_pixel_episodes
from benchmarks.v5.phase_6_mask_decoder.boundary_data import make_loader, move_batch
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN, IMAGE_KEYS, PROMPT_KEYS, ROOT, sha256
from benchmarks.v5.phase_6_mask_decoder.evaluate_crc_resolution import select_positions
from benchmarks.v5.phase_6_mask_decoder.probe_boundary_refinement import metrics, summarize, prompt_violations
from .model import load_model
from .scoring import split_target_tasks,scoring_policy


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def paired(rows, name):
    old={r['position']:r for r in rows if r['model']=='retained_centroid'}
    candidate=[r for r in rows if r['model']==name]
    result={}
    for key in ('dice','bf1_2'):
        valid=[r for r in candidate if r[key] is not None and old[r['position']][key] is not None]
        groups=sorted({r['spatial_group'] for r in valid})
        if not groups:
            result[key]={'evaluable_pairs':0,'status':'no evaluable tissue-target pairs'}
            continue
        values=[[r[key]-old[r['position']][key] for r in valid if r['spatial_group']==g] for g in groups]
        sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
        if not groups:
            result[key]={'pairs':0};continue
        draws=np.random.default_rng(20261021).integers(0,len(groups),(2000,len(groups)))
        bootstrap=sums[draws].sum(1)/counts[draws].sum(1)
        result[key]={'pairs':len(valid),'spatial_blocks':len(groups),'new_minus_retained':float(sums.sum()/counts.sum()),
                     'spatial_block_bootstrap95':np.percentile(bootstrap,[2.5,97.5]).tolist()}
    return result


def report(rows, refs):
    subset=set(select_positions(refs,8));result={};primary_rows,_=split_target_tasks(rows,dataset='crc')
    for name in sorted({r['model'] for r in rows}):
        rr=[r for r in rows if r['model']==name]
        primary,excluded=split_target_tasks(rr,dataset='crc')
        result[name]={'overall':summarize(primary),'legacy_all_targets':summarize(rr),'excluded_background_target_tasks':summarize(excluded),
                      'per_class':{str(c):summarize([r for r in primary if r['target_class']==c]) for c in sorted({r['target_class'] for r in primary})},
                      'paired_to_retained':paired(primary_rows,name),'original_96_subset':summarize([r for r in primary if r['position'] in subset]),
                      'legacy_all_targets_original_96_subset':summarize([r for r in rr if r['position'] in subset]),
                      'prompt_semantics':'historical_region_centroid' if name=='retained_centroid' else 'historical_centroid_coordinates_interpreted_as_pixel_clicks'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--merge',nargs='+',type=Path)
    a=p.parse_args()
    if ROOT not in a.output.resolve().parents or not 0<=a.shard<a.shards:raise ValueError('output/shard')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    plan=json.loads(a.plan.read_text());plan_sha=sha256(a.plan);code_sha=sha256(Path(__file__))
    source=json.loads((DENSE_RUN/'run.json').read_text());refs=source['refs']['val']
    if len(refs)!=1024 or sha256(DENSE_RUN/'run.json')!=plan['legacy_run_sha256']:raise ValueError('legacy reference changed')
    try:
        if a.merge:
            rows=[];replays=[]
            for folder in a.merge:
                run=json.loads((folder/'run.json').read_text());status=json.loads((folder/'status.json').read_text())
                if status['status']!='complete' or run['smoke'] or run['plan_sha256']!=plan_sha or run['code_sha256']!=code_sha:raise ValueError('incomplete/mismatched shard')
                rows.extend(json.loads((folder/'rows.json').read_text()));replays.extend(json.loads((folder/'replay.json').read_text()))
            names=['retained_centroid','retained_pixel',*plan['checkpoints']]
            expected={(i,n) for i in range(1024) for n in names}
            if len(rows)!=len(expected) or {(r['position'],r['model']) for r in rows}!=expected:raise ValueError('missing/duplicate query-model')
            if len(replays)!=1024 or {r['position'] for r in replays}!=set(range(1024)):raise ValueError('replay coverage')
            result=report(rows,refs)
            if abs(result['retained_centroid']['legacy_all_targets']['dice']-.803416)>1e-6 or abs(result['retained_centroid']['legacy_all_targets']['bf1_2']-.557040)>1e-6:raise ValueError('historical aggregate not reproduced')
            dump(a.output/'rows.json',rows);dump(a.output/'summary.json',{'models':result,'plan_sha256':plan_sha,'source_revision':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                 'maximum_replay_error':max(r['max_error'] for r in replays),'test_used':False,'puma_used':False,'legacy_cells_unchanged':True,'same_GT_and_points':True,
                 'same_prompt_semantics':False,'comparison_kind':'historical_coordinate_compatibility',
                 'scoring_policy':scoring_policy(),
                 'scope':'CRC spatial development with historical initialization exposure; original input compatibility across different prompt readers; not a matched real-click comparison; FP32 threshold0.5',
                 'shards':[str(f) for f in a.merge]})
        else:
            torch.set_num_threads(4);device=torch.device('cuda:0')
            original=json.loads((Path(source['source_cache'])/'run.json').read_text())
            metadata=json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
            da=argparse.Namespace(**original['args']);da.spatial_data=Path(da.spatial_data);da.routing=Path(da.routing)
            dataset=make_loader(da,metadata,'val').dataset.dataset
            spatial=pd.read_parquet(da.spatial_data/'spatial_manifest.parquet').set_index('patch_id')
            positions=select_positions(refs,1) if a.smoke else list(range(a.shard,len(refs),a.shards))
            model,weights=load_model(device,mapper='centroid');model.eval().requires_grad_(False)
            models={'retained_centroid':model}
            for name,info in plan['checkpoints'].items():
                if sha256(Path(info['path']))!=info['sha256']:raise ValueError('candidate checkpoint changed')
                candidate=copy.deepcopy(model);payload=torch.load(info['path'],map_location='cpu',weights_only=False)
                if payload['step']!=info['step'] or payload['mapper']!='pixel':raise ValueError('checkpoint identity')
                candidate.load_state_dict(payload['model'],strict=True);candidate.mapper='pixel';candidate.eval().requires_grad_(False)
                models[name]=candidate;del payload
            dump(a.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                code_sha256=code_sha,plan_sha256=plan_sha,weights=weights,positions=positions,smoke=a.smoke,shard=a.shard,shards=a.shards,
                model_names=['retained_centroid','retained_pixel',*plan['checkpoints']],precision='FP32',threshold=.5,test_used=False,puma_used=False))
            dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=0,cases=len(positions)))
            rows=[];replays=[];(a.output/'masks').mkdir()
            for number,position in enumerate(positions):
                ref=refs[position];identity=ref['identity'];batch=collate_joint_pixel_episodes([dataset[int(identity['episode_index'])]],max_cells=255)
                if str(batch['patch_id'][0])!=identity['patch_id'] or int(batch['target_class'][0])!=identity['target_class']:raise ValueError('episode identity')
                truth=(batch['pixel_gt']==identity['target_class']).to(device);valid=(batch['pixel_gt']!=255).to(device)
                inputs=move_batch({k:batch[k] for k in IMAGE_KEYS},device);prompts=move_batch({k:batch[k] for k in PROMPT_KEYS},device)
                old=torch.load(ref['file'],map_location=device,weights_only=False)
                if not torch.equal(truth,old['truth']) or not torch.equal(valid,old['valid']):raise ValueError('GT replay changed')
                for key in ['positive_xy','negative_xy','positive_mask','negative_mask']:
                    if not torch.equal(prompts[key],old[key]):raise ValueError('point replay changed')
                arrays={'truth':truth.cpu().numpy()[0],'valid':valid.cpu().numpy()[0]};current=[]
                with torch.inference_mode():
                    for name in ['retained_centroid','retained_pixel',*plan['checkpoints']]:
                        current_model=model if name.startswith('retained_') else models[name]
                        current_model.mapper='centroid' if name=='retained_centroid' else 'pixel'
                        probability=current_model(inputs,inputs,prompts,local=True,shared=True,with_support=False)['query']['pixel_probability']
                        if not torch.isfinite(probability).all():raise FloatingPointError('nonfinite prediction')
                        if name=='retained_centroid':
                            error=float((probability-model.dense(old)['pixel_probability']).abs().max())
                            if error>2e-6:raise ValueError(f'original exact replay failed {position}: {error}')
                            replays.append(dict(position=position,max_error=error))
                        row=dict(position=position,model=name,**identity,spatial_group=str(spatial.loc[identity['patch_id'],'spatial_group_id']),
                                 **metrics(probability,truth,valid),prompt_violations=prompt_violations(probability,prompts))
                        current.append(row);arrays[name]=(probability>=.5).cpu().numpy()[0]
                path=a.output/'masks'/f'{position:04d}.npz';np.savez_compressed(path,**arrays)
                with np.load(path) as saved:
                    v=saved['valid'];t=saved['truth']&v
                    for row in current:
                        pred=saved[row['model']]&v
                        if (int((pred&t).sum()),int((pred&~t&v).sum()),int((~pred&t).sum()))!=(row['tp'],row['fp'],row['fn']):raise ValueError('saved mask count')
                rows.extend(current)
                with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(position=position,completed=number+1,seconds=time.time()-started,rows=current))+'\n')
                if (number+1)%10==0 or number==0:
                    dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=number+1,cases=len(positions),seconds=time.time()-started))
                    print('CASE',number+1,'/',len(positions),'seconds',round(time.time()-started,1),flush=True)
            dump(a.output/'rows.json',rows);dump(a.output/'replay.json',replays)
            dump(a.output/'summary.json',dict(models=report(rows,refs),maximum_replay_error=max(r['max_error'] for r in replays),saved_masks_verified=True,seconds=time.time()-started))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
