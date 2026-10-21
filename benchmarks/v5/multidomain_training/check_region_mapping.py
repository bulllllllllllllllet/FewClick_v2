"""Same-click inference control: nearest centroid, membership argmax, soft mixture.

All three readers use whole region tokens. This probe never changes a checkpoint
or the production reader. Ground truth is used only for replay and audit.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import remap_prompt_tokens
from benchmarks.v5.phase_6_mask_decoder.boundary_data import make_loader, move_batch
from benchmarks.v5.phase_6_mask_decoder.evaluate_crc_resolution import input_rgb, select_positions
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN, IMAGE_KEYS, PROMPT_KEYS, ROOT, sha256
from benchmarks.v5.phase_6_mask_decoder.probe_boundary_refinement import metrics, prompt_violations, summarize
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import query_features
from .diagnose_regression import make_interior_prompts, paired_delta, point_audit
from .model import load_model, pixel_prompt_tokens
from .regression import dump
from .scoring import scoring_policy, split_target_tasks

MODES=('centroid','hard','soft')
WEIGHTS=('retained','C0','C3')


def membership(encoded, xy):
    weights=sample_assignment_at_cells(encoded['assignment_low'].float(),xy.float())
    return weights/weights.sum(1,keepdim=True).clamp_min(1e-8)


def hard_prompt_tokens(encoded, xy, valid):
    """Choose a single region by membership at the click; never consult GT."""
    if xy.shape[1]==0:return encoded['tokens'].new_zeros(xy.shape[0],0,encoded['tokens'].shape[-1]).float()
    indices=membership(encoded,xy).argmax(1)
    tokens=encoded['tokens'].float().gather(1,indices[...,None].expand(-1,-1,encoded['tokens'].shape[-1]))
    return tokens*valid[:,:,None]


def encode_task(model, encoded, prompts, mode):
    if mode not in MODES:raise ValueError('unknown reader')
    values={'prompt_size_id':prompts['prompt_size_id']}
    for sign in ('positive','negative'):
        xy,valid=prompts[sign+'_xy'],prompts[sign+'_mask']
        if mode=='hard':tokens=hard_prompt_tokens(encoded,xy,valid)
        elif mode=='soft':tokens=pixel_prompt_tokens(encoded,xy,valid)
        else:tokens=remap_prompt_tokens(encoded['tokens'],encoded['xy'],xy,valid)[0]
        values.update({sign+'_tokens':tokens,sign+'_xy':xy,sign+'_mask':valid})
    return model.j5.decoder.prompt_model.encode_prompt_task(values)


def audit_regions(encoded, prompts, truth, valid):
    """GT purity is a post-hoc diagnostic and never enters encode_task."""
    assignment=encoded['assignment_low'].float()
    size=assignment.shape[-2:]
    target=F.interpolate((truth&valid)[:,None].float(),size=size,mode='area')
    known=F.interpolate(valid[:,None].float(),size=size,mode='area')
    purity=(assignment*target).sum((2,3))/(assignment*known).sum((2,3)).clamp_min(1e-8)
    result=[]
    for sign in ('positive','negative'):
        xy=prompts[sign+'_xy'];mask=prompts[sign+'_mask'][0]
        if xy.shape[1]==0:continue
        weights=membership(encoded,xy)[0]
        hard=weights.argmax(0)
        centroid=torch.cdist(xy.float(),encoded['xy'].float()).argmin(2)[0]
        entropy=-(weights*weights.clamp_min(1e-30).log()).sum(0)
        for i in torch.where(mask)[0].tolist():
            result.append(dict(sign=sign,xy=xy[0,i].tolist(),centroid_region=int(centroid[i]),hard_region=int(hard[i]),
                same_region=bool(centroid[i]==hard[i]),top1_membership=float(weights[hard[i],i]),
                effective_regions=float(entropy[i].exp()),centroid_target_fraction=float(purity[0,centroid[i]]),
                hard_target_fraction=float(purity[0,hard[i]]),soft_target_fraction=float((weights[:,i]*purity[0]).sum())))
    return result


def report(rows, cases):
    buckets=defaultdict(list)
    for row in rows:buckets[(row['weight'],row['mapping'])].append(row)
    conditions={f'{w}|{m}':dict(overall=summarize(rr),per_class={str(c):summarize([r for r in rr if r['target_class']==c])
        for c in sorted({r['target_class'] for r in rr})}) for (w,m),rr in buckets.items()}
    contrasts={f'{w}|{m}-soft':{key:paired_delta(buckets[(w,m)],buckets[(w,'soft')],key) for key in ('dice','bf1_2')}
               for w,m in buckets if m!='soft'}
    audits={}
    for w in WEIGHTS:
        points=[p for c in cases for p in c['audits'][w]]
        audits[w]=dict(points=len(points),centroid_hard_agreement=float(np.mean([p['same_region'] for p in points])),
                       mean_top1_membership=float(np.mean([p['top1_membership'] for p in points])),
                       mean_effective_regions=float(np.mean([p['effective_regions'] for p in points])))
        for sign in ('positive','negative'):
            pp=[p for p in points if p['sign']==sign]
            audits[w][sign]={k:float(np.mean([p[k] for p in pp])) if pp else None
                            for k in ('centroid_target_fraction','hard_target_fraction','soft_target_fraction')}
    return dict(cases=len(cases),predictions=len(rows),conditions=conditions,paired_contrasts=contrasts,region_audits=audits,
        replay_masks_exact=all(c['replay_masks_exact'] for c in cases),scoring_policy=scoring_policy(),
        scope='CRC tissue-target spatial development, fixed GT-interior clicks; legacy RGB/cells/FOV; not WSI or unseen-cancer test',
        training_performed=False,production_reader_changed=False,puma_used=False,test_used=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--merge',type=Path,nargs='+');a=p.parse_args()
    if ROOT not in a.output.resolve().parents or not 0<=a.shard<a.shards:raise ValueError('invalid output/shard')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    try:
        plan=json.loads(a.plan.read_text());source=json.loads((DENSE_RUN/'run.json').read_text());refs=source['refs']['val']
        if len(refs)!=1024 or sha256(DENSE_RUN/'run.json')!=plan['legacy_run_sha256']:raise ValueError('legacy source changed')
        prior=a.prior_root/'full_merged_1021_175042';prior_rows=json.loads((prior/'rows.json').read_text())
        prior_cases={c['position']:c for c in json.loads((prior/'cases.json').read_text())}
        prior_rows={(r['position'],r['weight'],r['prompt']):r for r in prior_rows if r['cells']=='legacy'}
        primary,_=split_target_tasks([dict(position=i,**r['identity']) for i,r in enumerate(refs)],dataset='crc')
        all_positions=[r['position'] for r in primary]
        if len(all_positions)!=931:raise ValueError('tissue-target identity count changed')
        dependencies=[Path(inspect.getfile(x)) for x in (load_model,query_features,metrics,make_interior_prompts,scoring_policy)]
        contract=dict(source_sha256=sha256(Path(__file__)),plan_sha256=sha256(a.plan),
            prior_rows_sha256=sha256(prior/'rows.json'),prior_cases_sha256=sha256(prior/'cases.json'),
            dependency_sha256={str(f):sha256(f) for f in dependencies})
        if a.merge:
            rows=[];cases=[]
            for folder in a.merge:
                run=json.loads((folder/'run.json').read_text());status=json.loads((folder/'status.json').read_text())
                if status['status']!='complete' or run['smoke'] or run['contract']!=contract:raise ValueError('mismatched/incomplete shard')
                rows.extend(json.loads((folder/'rows.json').read_text()));cases.extend(json.loads((folder/'cases.json').read_text()))
            expected={(i,w,m) for i in all_positions for w in WEIGHTS for m in MODES}
            if len(cases)!=931 or {c['position'] for c in cases}!=set(all_positions):raise ValueError('case coverage mismatch')
            if len(rows)!=len(expected) or {(r['position'],r['weight'],r['mapping']) for r in rows}!=expected:raise ValueError('prediction coverage mismatch')
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',report(rows,cases))
            dump(a.output/'run.json',dict(contract=contract,shards=[str(f) for f in a.merge]))
        else:
            positions=[i for i in select_positions(refs,1) if i in all_positions] if a.smoke else all_positions[a.shard::a.shards]
            # All controls reuse precisely the previous locked clicks, not new GT-guided choices.
            original=json.loads((Path(source['source_cache'])/'run.json').read_text())
            metadata=json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
            da=argparse.Namespace(**original['args']);da.spatial_data=Path(da.spatial_data);da.routing=Path(da.routing)
            dataset=make_loader(da,metadata,'val').dataset.dataset;device=torch.device('cuda:0')
            retained,initial=load_model(device,mapper='pixel');retained.eval().requires_grad_(False);models={'retained':retained}
            for name in WEIGHTS[1:]:
                info=plan['checkpoints'][name]
                if sha256(Path(info['path']))!=info['sha256']:raise ValueError('candidate changed')
                saved=torch.load(info['path'],map_location='cpu',weights_only=False)
                if saved['step']!=info['step'] or saved['mapper']!='pixel':raise ValueError('checkpoint identity')
                model=copy.deepcopy(retained);model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);models[name]=model;del saved
            previous_shards=json.loads((a.prior_root/'full_selection.json').read_text());case_paths={}
            for shard in previous_shards:
                folder=Path(shard['output'])
                if json.loads((folder/'status.json').read_text())['status']!='complete':raise ValueError('prior incomplete')
                for c in json.loads((folder/'cases.json').read_text()):case_paths[c['position']]=folder/'cases'/f"{c['position']:04d}"
            dump(a.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                contract=contract,positions=positions,smoke=a.smoke,weights=initial,threshold=.5,precision='FP32',modes=MODES))
            rows=[];cases=[];(a.output/'masks').mkdir()
            for number,position in enumerate(positions):
                identity=refs[position]['identity'];batch=collate_joint_pixel_episodes([dataset[int(identity['episode_index'])]],max_cells=255)
                if str(batch['patch_id'][0])!=identity['patch_id'] or int(batch['target_class'][0])!=identity['target_class']:raise ValueError('case identity changed')
                labels=batch['pixel_gt'][0].numpy();truth=(batch['pixel_gt']==identity['target_class']).to(device);valid=(batch['pixel_gt']!=255).to(device)
                inputs=move_batch({k:batch[k] for k in IMAGE_KEYS},device);historical=move_batch({k:batch[k] for k in PROMPT_KEYS},device)
                previous=prior_cases[position];rgb=input_rgb(inputs['image'])
                if hashlib.sha256(rgb.tobytes()).hexdigest()!=previous['RGB_sha256']:raise ValueError('RGB changed')
                prompts,feasible=make_interior_prompts(labels,identity['target_class'],previous['input_mpp'],historical,position,device)
                if feasible['status']!='available' or int(prompts['prompt_size_id'][0])!=previous['prompt_size_id']:raise ValueError('prompt condition changed')
                audit=point_audit(prompts,labels==identity['target_class'],labels!=255)
                if audit!=previous['prompts']['interior']:raise ValueError('locked point replay changed')
                with np.load(case_paths[position]/'predictions.npz') as npz:old={k:npz[k] for k in npz.files if k.endswith(('interior_centroid','interior_pixel')) or k in ('truth','valid')}
                if not np.array_equal(truth.cpu().numpy()[0],old['truth']) or not np.array_equal(valid.cpu().numpy()[0],old['valid']):raise ValueError('GT changed')
                arrays={'truth':old['truth'],'valid':old['valid']};current=[];audits={}
                with torch.inference_mode():
                    for name,model in models.items():
                        encoded=model.encode_image(inputs);audits[name]=audit_regions(encoded,prompts,truth,valid)
                        for mode in MODES:
                            task=encode_task(model,encoded,prompts,mode)
                            features=query_features(model.j5,encoded,task,prompts,query_id='same',support_id='same')
                            probability=model.dense(features)['pixel_probability']
                            if not torch.isfinite(probability).all():raise FloatingPointError('nonfinite prediction')
                            prediction=(probability>=.5).cpu().numpy()[0]
                            result=dict(position=position,**identity,weight=name,mapping=mode,
                                spatial_group=prior_rows[(position,name,'interior_pixel')]['spatial_group'],
                                **metrics(probability,truth,valid),prompt_violations=prompt_violations(probability,prompts))
                            if mode!='hard':
                                kind='interior_pixel' if mode=='soft' else 'interior_centroid'
                                if not np.array_equal(prediction,old[name+'|legacy|'+kind]):raise ValueError('prior mask replay mismatch')
                                reference=prior_rows[(position,name,kind)]
                                if any(result[k]!=reference[k] for k in ('tp','fp','fn','dice','bf1_2','prompt_violations')):raise ValueError('prior score mismatch')
                            current.append(result);arrays[name+'|'+mode]=prediction
                path=a.output/'masks'/f'{position:04d}.npz';np.savez_compressed(path,**arrays)
                with np.load(path) as npz:
                    v=npz['valid'];t=npz['truth']&v
                    for r in current:
                        pred=npz[r['weight']+'|'+r['mapping']]&v
                        if (int((pred&t).sum()),int((pred&~t&v).sum()),int((~pred&t).sum()))!=(r['tp'],r['fp'],r['fn']):raise ValueError('saved mask count mismatch')
                case=dict(position=position,identity=identity,prompts=audit,audits=audits,replay_masks_exact=True,prior_case=str(case_paths[position]))
                rows.extend(current);cases.append(case)
                with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(completed=number+1,case=case,rows=current))+'\n')
                dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=number+1,total=len(positions),seconds=time.time()-started))
                if number%10==0:print('CASE',number+1,'/',len(positions),'seconds',round(time.time()-started,1),flush=True)
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',report(rows,cases))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
