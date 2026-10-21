"""Same-task click-policy audit and five-group sensitivity on fixed C3 weights."""
import argparse
from collections import defaultdict
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import numpy as np
from scipy.ndimage import label, distance_transform_edt
import torch

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts,boundary_f1
from benchmarks.v5.multidomain_data.prepare import dump,sha
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import query_features
from .catalog import interior_labels
from .click_policy import repoint,POLICY
from .episodes import EpisodeCorpus
from .evaluate import summarize
from .model import load_model
from .scoring import BACKGROUND_TARGETS
from .visualize_random import select_plans

ROOT=Path('/nfs-medical3/zyh/fewclick_v2')
CONDITIONS=('legacy','group0','group1','group2','group3','group4','ensemble','oracle','worst')


def point_quality(labels,plan,mpp):
    inside=interior_labels(labels,mpp);cache={};rows=[]
    tissue_negative=any(c not in (plan['target'],255,*BACKGROUND_TARGETS[plan['dataset']]) for c in np.unique(inside))
    for sign in ('positive','negative'):
        for x,y in plan[sign+'_xy']:
            xx,yy=int(x*labels.shape[1]),int(y*labels.shape[0]);code=int(labels[yy,xx])
            if code not in cache:
                mask=labels==code;components,_=label(mask);areas=np.bincount(components.ravel());areas[0]=0
                distance=distance_transform_edt(np.pad(mask,1),sampling=mpp[::-1])[1:-1,1:-1]
                cache[code]=(components,areas,distance)
            components,areas,distance=cache[code];fraction=float(areas[components[yy,xx]]/areas.max())
            rows.append(dict(sign=sign,code=code,component_fraction=fraction,tiny_fragment=fraction<.1,
                depth_um=float(distance[yy,xx]),frame_um=float(min((xx+.5)*mpp[0],(labels.shape[1]-xx-.5)*mpp[0],(yy+.5)*mpp[1],(labels.shape[0]-yy-.5)*mpp[1])),
                background_despite_tissue=sign=='negative' and code in BACKGROUND_TARGETS[plan['dataset']] and tissue_negative))
    return rows


def audit(corpus,training_run,output):
    training=[]
    with (training_run/'training_episodes.jsonl').open() as f:
        for i,line in enumerate(f):
            if i%200==0:training.append(json.loads(line))
    if len(training)!=400:raise ValueError('expected400 evenly spaced actual training episodes')
    validation=[p for p in json.loads((training_run/'validation_plans.json').read_text()) if p['quality_condition']=='original']
    rows=[]
    for split,plans in [('train',training),('val',validation)]:
        for i,plan in enumerate(plans):
            if split=='train' and corpus.plan(plan['number'])!=plan:raise ValueError('actual old training click replay mismatch')
            source=corpus.index[plan['support_id']];labels=corpus.labels(source);mpp=json.loads(corpus.rows[source]['input_mpp_xy_json'])
            new=repoint(corpus,plan,0)
            for name,p in [('legacy',plan),('representative',new)]:
                rows.append(dict(split=split,number=plan['number'],dataset=plan['dataset'],policy=name,points=point_quality(labels,p,mpp)))
            if (i+1)%50==0:
                dump(output/'status.json',dict(status='auditing',worker_pid=os.getpid(),split=split,completed=i+1,total=len(plans)));print('AUDIT',split,i+1,flush=True)
    summary={}
    for split in ('train','val'):
        for policy in ('legacy','representative'):
            cases=[r for r in rows if r['split']==split and r['policy']==policy]
            points=[p for r in cases for p in r['points']];positive=[p for p in points if p['sign']=='positive'];negative=[p for p in points if p['sign']=='negative']
            summary[split+'|'+policy]=dict(episodes=len(cases),points=len(points),positive_points=len(positive),negative_points=len(negative),
                positive_tiny_fragment_fraction=float(np.mean([p['tiny_fragment'] for p in positive])),
                first_positive_in_largest_fraction=float(np.mean([next(p for p in r['points'] if p['sign']=='positive')['component_fraction']==1 for r in cases])),
                all_points_near_frame_8um=float(np.mean([p['frame_um']<8 for p in points])),
                median_depth_um=float(np.median([p['depth_um'] for p in points])),
                negative_background_despite_tissue_fraction=float(np.mean([p['background_despite_tissue'] for p in negative])))
    dump(output/'audit_rows.json',rows);dump(output/'summary.json',dict(audit=summary,actual_training_replay_exact=True,training_episodes=400,validation_base_episodes=len(validation)))


def metric(prob,supervision,plan):
    truth,valid=supervision['truth'],supervision['valid'];pred=prob>=.5
    counts={k:int(v[0]) for k,v in binary_counts(pred,truth,valid).items()}
    den=2*counts['tp']+counts['fp']+counts['fn'];radius=int(np.ceil(2/min(supervision['mpp'])))
    bf=boundary_f1(pred,truth,valid,radius)['boundary_f1'][0]
    return dict(plan,**counts,dice=2*counts['tp']/den if counts['positive'] else None,
        all_foreground_dice=2*counts['positive']/(counts['valid']+counts['positive']),
        false_positive_fraction=counts['fp']/counts['valid'],bf1_um2=float(bf) if torch.isfinite(bf) else None)


def build_summary(rows):
    result={}
    for condition in CONDITIONS:
        rr=[r for r in rows if r['condition']==condition]
        result[condition]=dict(overall=summarize(rr),by_dataset={d:summarize([r for r in rr if r['dataset']==d]) for d in sorted({r['dataset'] for r in rr})})
    return dict(conditions=result,oracle_uses_query_GT=True,worst_uses_query_GT=True,ensemble_uses_query_GT=False,
        fixed_group_selected_before_scores=True,groups=5,clicks_per_group='same positive/negative budget as legacy',
        five_groups_require_extra_prompts=True,training_performed=False,puma_used=False,test_used=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--checkpoint-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--merge',type=Path,nargs='+');a=p.parse_args()
    if ROOT not in a.output.resolve().parents or not 0<=a.shard<a.shards:raise ValueError('invalid output/shard')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    try:
        run=json.loads((a.training_run/'run.json').read_text());plans=json.loads((a.training_run/'validation_plans.json').read_text())
        if sha((a.training_run/'validation_plans.json').read_bytes())!=run['validation_plans_sha256']:raise ValueError('validation changed')
        files=[Path(__file__),Path(inspect.getfile(repoint)),Path(inspect.getfile(EpisodeCorpus)),Path(inspect.getfile(load_model))]
        contract=dict(hashes={str(f):sha(f.read_bytes()) for f in files},plans_sha256=run['validation_plans_sha256'],checkpoint_plan_sha256=sha(a.checkpoint_plan.read_bytes()))
        if a.merge:
            rows=[];cases=[]
            for folder in a.merge:
                source=json.loads((folder/'run.json').read_text());status=json.loads((folder/'status.json').read_text())
                if status['status']!='complete' or source['contract']!=contract or source['smoke']:raise ValueError('incomplete/mismatched source')
                rows.extend(json.loads((folder/'rows.json').read_text()));cases.extend(json.loads((folder/'cases.json').read_text()))
            expected={(i,c) for i in range(len(plans)) for c in CONDITIONS}
            if len(cases)!=len(plans) or {c['index'] for c in cases}!=set(range(len(plans))):raise ValueError('case coverage')
            if len(rows)!=len(expected) or {(r['index'],r['condition']) for r in rows}!=expected:raise ValueError('prediction coverage')
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',build_summary(rows))
            dump(a.output/'validation_plans.json',[c['new_plans'][0] for c in sorted(cases,key=lambda c:c['index'])])
            dump(a.output/'run.json',dict(contract=contract,shards=[str(f) for f in a.merge]))
        else:
            corpus=EpisodeCorpus(Path(run['args']['catalog']),run['args']['seed'],prompt_policy='legacy')
            if sha((corpus.catalog/'tiles.parquet').read_bytes())!=run['catalog_sha256']:raise ValueError('catalog changed')
            if a.audit:
                audit(corpus,a.training_run,a.output)
                dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));return
            positions=[i for i,p in enumerate(plans) if p['quality_condition']=='original' and p['kind']=='local'][:4] if a.smoke else list(range(len(plans)))[a.shard::a.shards]
            if a.smoke:positions=[next(i for i,p in enumerate(plans) if p['quality_condition']=='original' and p['dataset']==d and p['kind']=='local') for d in ('crc','tiger','ignite','ocelot')]
            info=json.loads(a.checkpoint_plan.read_text())['checkpoints']['C3'];checkpoint=Path(info['path'])
            if checkpoint.parent!=a.training_run or sha(checkpoint.read_bytes())!=info['sha256']:raise ValueError('checkpoint changed')
            model,_=load_model(torch.device('cuda:0'));saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['step']!=info['step'] or saved['mapper']!='pixel':raise ValueError('checkpoint identity')
            model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);del saved
            reference={(r['number'],r['quality_condition']):r for r in json.loads((a.training_run/'validation_step_20000/rows.json').read_text())}
            gallery={(s['plan']['number'],s['plan']['quality_condition']) for s in select_plans(plans,20261021)}
            dump(a.output/'run.json',dict(worker_pid=os.getpid(),contract=contract,source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),smoke=a.smoke,indices=positions))
            rows=[];cases=[];(a.output/'masks').mkdir();(a.output/'gallery_arrays').mkdir()
            for count,index in enumerate(positions):
                plan=plans[index];new_plans=[repoint(corpus,plan,g) for g in range(5)]
                inputs,supervision=corpus.materialize(plan,'cuda:0');arrays={k:supervision[k].cpu().numpy()[0] for k in ('truth','valid','support_truth','support_valid')};probabilities=[];current=[]
                # Freeze all five source click sets before looking at predictions.
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not run['args']['no_amp']):
                    support=model.encode_image(inputs['support']);query=support if inputs['shared'] else model.encode_image(inputs['query'])
                    for condition,current_plan in [('legacy',plan),*[(f'group{g}',q) for g,q in enumerate(new_plans)]]:
                        prompts={'prompt_size_id':inputs['prompts']['prompt_size_id']}
                        for sign in ('positive','negative'):
                            prompts[sign+'_xy']=torch.tensor([current_plan[sign+'_xy']],device='cuda:0',dtype=torch.float32)
                            prompts[sign+'_mask']=inputs['prompts'][sign+'_mask']
                        task=model.encode_support(support,prompts)
                        features=query_features(model.j5,query,task,prompts,query_id='same' if inputs['local'] else 'remote',support_id='same')
                        probability=model.dense(features)['pixel_probability'].float()
                        if not torch.isfinite(probability).all():raise FloatingPointError('nonfinite prediction')
                        r=metric(probability,supervision,plan)|dict(index=index,condition=condition)
                        if condition=='legacy':
                            ref=reference[plan['number'],plan['quality_condition']]
                            if any(r[k]!=ref[k] for k in ('tp','fp','fn','tn','positive','valid')):raise ValueError('legacy validation replay mismatch')
                        current.append(r);probabilities.append(probability)
                    average=torch.stack(probabilities[1:]).mean(0);current.append(metric(average,supervision,plan)|dict(index=index,condition='ensemble'))
                candidates=current[1:6];ordered=sorted(candidates,key=lambda r:r['dice'] if r['positive'] else -r['false_positive_fraction'])
                current+=[ordered[-1]|dict(condition='oracle',selected_group=ordered[-1]['condition'],selection_uses_query_GT=True),
                          ordered[0]|dict(condition='worst',selected_group=ordered[0]['condition'],selection_uses_query_GT=True)]
                probs=[*probabilities,average];masks=np.stack([(p[0]>=.5).cpu().numpy() for p in probs])
                path=a.output/'masks'/f'{index:04d}.npz';np.savez_compressed(path,masks=np.packbits(masks,axis=-1),truth=np.packbits(arrays['truth'],axis=-1),valid=np.packbits(arrays['valid'],axis=-1))
                with np.load(path) as saved_masks:
                    restored=np.unpackbits(saved_masks['masks'],axis=-1).astype(bool);truth=np.unpackbits(saved_masks['truth'],axis=-1).astype(bool);valid=np.unpackbits(saved_masks['valid'],axis=-1).astype(bool)
                    for r,pred in zip(current[:7],restored):
                        t=truth&valid;p=pred&valid
                        if (int((p&t).sum()),int((p&~t&valid).sum()),int((~p&t).sum()))!=tuple(r[k] for k in ('tp','fp','fn')):raise ValueError('saved mask count mismatch')
                if (plan['number'],plan['quality_condition']) in gallery:
                    from PIL import Image
                    for role in ('support','query'):
                        source=corpus.rows[corpus.index[plan[role+'_id']]];variant=json.loads(source['variants_json'])[plan[role+'_quality']]
                        arrays[role+'_rgb']=np.asarray(Image.open(variant['image']).convert('RGB'))
                    arrays['probabilities']=np.stack([p[0].cpu().numpy() for p in probs]);np.savez_compressed(a.output/'gallery_arrays'/f'{index:04d}.npz',**arrays)
                case=dict(index=index,legacy_plan=plan,new_plans=new_plans,oracle_group=ordered[-1]['condition'],unique_new_click_sets=len({json.dumps([p['positive_xy'],p['negative_xy']]) for p in new_plans}))
                cases.append(case);rows.extend(current)
                with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,rows=current))+'\n')
                dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=count+1,total=len(positions),seconds=time.time()-started))
                if count%10==0:print('CASE',count+1,'/',len(positions),round(time.time()-started,1),flush=True)
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',build_summary(rows))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
