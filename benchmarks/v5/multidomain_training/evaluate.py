"""Frozen development tasks, physical boundary scores and explicit empty queries."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts,boundary_f1
from benchmarks.v5.multidomain_data.prepare import dump
from .episodes import DATASETS,KINDS
from .scoring import scoring_policy,split_target_tasks


def validation_plans(corpus):
    plans=[];number=10000000
    policy=getattr(corpus,'training_policy',None)
    datasets=policy['datasets'] if policy else DATASETS
    variants=[('original','0.8'),('0.8','original'),('1.0','0.8'),('0.8','1.0')]
    for dataset in datasets:
        for kind in KINDS:
            for repetition in range(4):
                original=corpus.plan(number,split='val',dataset=dataset,kind=kind,quality_aug=False,fixed_clicks=True);number+=1
                plans.append(original|dict(quality_condition='original',pair_id=number))
                if policy is None:
                    sq,qq=variants[repetition]
                    plans.append(original|dict(support_quality=sq,query_quality=qq,quality_condition='degraded',pair_id=number))
    return plans


def summarize(rows,include_background_targets=False):
    if not include_background_targets:rows,_=split_target_tasks(rows)
    present=[r for r in rows if r['positive']>0];absent=[r for r in rows if r['positive']==0]
    boundary=[r['bf1_um2'] for r in present if r['bf1_um2'] is not None]
    return dict(episodes=len(rows),present=len(present),absent=len(absent),
        dice=None if not present else float(np.mean([r['dice'] for r in present])),
        all_foreground_dice=None if not present else float(np.mean([r['all_foreground_dice'] for r in present])),
        bf1_um2=None if not boundary else float(np.mean(boundary)),bf1_evaluable=len(boundary),
        absent_false_positive_fraction=None if not absent else float(np.mean([r['false_positive_fraction'] for r in absent])),
        absent_over_1percent=None if not absent else float(np.mean([r['false_positive_fraction']>=.01 for r in absent])))


def evaluate(model,corpus,plans,device,output,amp=True,save_examples=0,anchor_count=0):
    if not 0<=anchor_count<=len(plans):raise ValueError('invalid validation anchor length')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);model.eval();rows=[]
    with torch.inference_mode():
        for i,plan in enumerate(plans):
            inputs,supervision=corpus.materialize(plan,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):result=model(**inputs,with_support=False)
            p=result['query']['pixel_probability'].float();truth=supervision['truth'];valid=supervision['valid'];prediction=p>=.5
            if not torch.isfinite(p).all():raise FloatingPointError('non-finite validation prediction')
            if not inputs['local'] and result['query_prompt_maps'].count_nonzero():raise ValueError('remote query contains source-coordinate clicks')
            counts={k:int(v[0]) for k,v in binary_counts(prediction,truth,valid).items()}
            tolerance=int(np.ceil(2/min(supervision['mpp'])))
            bf=boundary_f1(prediction,truth,valid,tolerance)['boundary_f1'][0]
            old=boundary_f1(prediction,truth,valid,2)['boundary_f1'][0]
            den=2*counts['tp']+counts['fp']+counts['fn']
            row=dict(plan,**counts,dice=(2*counts['tp']/den if counts['positive'] else None),
                all_foreground_dice=2*counts['positive']/(counts['valid']+counts['positive']),
                false_positive_fraction=counts['fp']/counts['valid'],bf1_um2=float(bf) if torch.isfinite(bf) else None,
                bf1_px2=float(old) if torch.isfinite(old) else None)
            rows.append(row)
            if i<save_examples:
                folder=output/f'case_{i:03d}';folder.mkdir()
                Image.fromarray((prediction[0].cpu().numpy()*255).astype(np.uint8)).save(folder/'prediction.png')
                Image.fromarray((truth[0].cpu().numpy()*255).astype(np.uint8)).save(folder/'target_GT.png')
                Image.fromarray((valid[0].cpu().numpy()*255).astype(np.uint8)).save(folder/'valid.png');dump(folder/'episode.json',row)
    groups=defaultdict(list)
    for r in rows:
        for label in ['dataset','protocol','quality_condition','query_organ']:groups[label+'|'+r[label]].append(r)
        groups[f'class|{r["dataset"]}|{r["target"]}'].append(r)
    _,excluded=split_target_tasks(rows)
    report=dict(overall=summarize(rows),legacy_all_targets=summarize(rows,include_background_targets=True),
        excluded_background_target_tasks=summarize(excluded,include_background_targets=True),
        strata={k:summarize(v) for k,v in sorted(groups.items())},threshold=.5,scoring_policy=scoring_policy(),
        scope='fixed GT-simulated clicks; spatial/development validation with historical CRC initialization exposure; no heldout PUMA or test')
    if anchor_count:report.update(anchor=summarize(rows[:anchor_count]),expanded_only=summarize(rows[anchor_count:]),anchor_episodes=anchor_count)
    dump(output/'rows.json',rows);dump(output/'summary.json',report)
    return report
