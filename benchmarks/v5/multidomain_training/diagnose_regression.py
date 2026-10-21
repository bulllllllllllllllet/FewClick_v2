"""Separate legacy region prompts, pixel clicks, cell recipe and weight effects."""
import argparse
from collections import defaultdict
import copy
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import collate_joint_pixel_episodes
from benchmarks.v5.multidomain_data.fast_nuclei import CachedCenterCropper, InstanSegDetector, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.multidomain_data.prepare import tiff_mpp
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.phase_6_mask_decoder.boundary_data import make_loader, move_batch
from benchmarks.v5.phase_6_mask_decoder.evaluate_crc_resolution import input_rgb, select_positions
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN, ROOT, IMAGE_KEYS, PROMPT_KEYS, sha256
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS
from benchmarks.v5.phase_6_mask_decoder.probe_boundary_refinement import metrics, summarize, prompt_violations
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import query_features
from .catalog import interior_labels
from .episodes import spread_clicks
from .model import load_model
from .regression import dump

PROMPTS=('historical_centroid','historical_pixel','interior_centroid','interior_pixel')


def prompt_kinds(size_one):
    return PROMPTS+('historical_pixel_id1','interior_pixel_id1') if size_one else PROMPTS


def point_audit(prompts, truth, valid):
    report={}
    for sign,expected in [('positive',True),('negative',False)]:
        xy=prompts[sign+'_xy'][0][prompts[sign+'_mask'][0]].cpu().numpy()
        if len(xy) and (not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any()):raise ValueError('invalid normalized point')
        pixel=np.minimum((xy*np.array(truth.shape[::-1])).astype(int),np.array(truth.shape[::-1])-1)
        x,y=pixel[:,0],pixel[:,1];known=valid[y,x];correct=known&(truth[y,x]==expected)
        report[sign]=dict(count=len(xy),correct=int(correct.sum()),ignored=int((~known).sum()),incorrect=int((known&~correct).sum()),xy=xy.tolist())
    return report


def make_interior_prompts(labels,target,mpp,historical,position,device):
    inside=interior_labels(labels,mpp);result={'prompt_size_id':historical['prompt_size_id'].clone()}
    rng=np.random.default_rng(20261021+300000000+position)
    for sign,mask in [('positive',inside==target),('negative',(inside!=target)&(inside!=255))]:
        count=int(historical[sign+'_mask'].sum())
        if int(mask.sum())<count:return None,dict(status='infeasible',sign=sign,available=int(mask.sum()),requested=count,margin_um=2.)
        xy=spread_clicks(mask,count,rng) if count else []
        result[sign+'_xy']=torch.tensor(xy,device=device,dtype=torch.float32).reshape(1,count,2)
        result[sign+'_mask']=torch.ones(1,count,device=device,dtype=torch.bool)
    return result,dict(status='available',margin_um=2.)


def unknown_cells(inputs):
    cells=inputs['cells'].clone();cells[...,3:10]=0;cells[...,3]=1
    if not torch.equal(cells[...,:3],inputs['cells'][...,:3]) or not torch.equal(cells[...,10:],inputs['cells'][...,10:]):raise ValueError('unknown ablation changed other features')
    return inputs|{'cells':cells}


def cell_stats(inputs):
    x=inputs['cells'][0][inputs['cell_valid'][0]].float().cpu().numpy()
    return dict(retained=len(x),total=float(inputs['total_cell_count'][0]),
                class_counts=np.bincount(x[:,3:10].argmax(1),minlength=7).tolist() if len(x) else [0]*7,
                mean_log_area=float(x[:,2].mean()) if len(x) else None,reg64_mean=float(x[:,10:].mean()) if len(x) else None,
                reg64_std=float(x[:,10:].std()) if len(x) else None)


def paired_delta(candidate,reference,key):
    old={r['position']:r for r in reference}
    eligible=[r for r in candidate if r['position'] in old and r[key] is not None and old[r['position']][key] is not None]
    groups=defaultdict(list)
    for r in eligible:groups[r['spatial_group']].append(r[key]-old[r['position']][key])
    if not groups:return dict(pairs=0)
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    draws=np.random.default_rng(20261021).integers(0,len(groups),(2000,len(groups)))
    boot=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(pairs=len(eligible),spatial_blocks=len(groups),candidate_minus_reference=float(sums.sum()/counts.sum()),bootstrap95=np.percentile(boot,[2.5,97.5]).tolist())


def build_summary(rows):
    buckets=defaultdict(list)
    for r in rows:buckets[(r['weight'],r['cells'],r['prompt'])].append(r)
    conditions={'|'.join(k):dict(overall=summarize(v),nonbackground_diagnostic=summarize([r for r in v if r['target_class']!=2]),per_class={str(c):summarize([r for r in v if r['target_class']==c]) for c in sorted({r['target_class'] for r in v})}) for k,v in buckets.items()}
    contrasts={}
    def compare(label,candidate,reference):
        if candidate in buckets and reference in buckets:
            contrasts[label]={key:paired_delta(buckets[candidate],buckets[reference],key) for key in ['dice','bf1_2']}
    for w,c,p in buckets:
        if w!='retained':compare(f'weights:{w}-retained|{c}|{p}',(w,c,p),('retained',c,p))
        if c!='legacy':compare(f'cells:{c}-legacy|{w}|{p}',(w,c,p),(w,'legacy',p))
        if p=='historical_pixel':compare(f'reader:pixel-centroid|{w}|{c}|historical',(w,c,p),(w,c,'historical_centroid'))
        if p=='interior_pixel':compare(f'reader:pixel-centroid|{w}|{c}|interior',(w,c,p),(w,c,'interior_centroid'))
        if p.endswith('_id1'):compare(f'size_id:1-native|{w}|{c}|{p}',(w,c,p),(w,c,p.removesuffix('_id1')))
    return dict(conditions=conditions,paired_contrasts=contrasts)


@lru_cache(maxsize=128)
def native_mpp(path):return tiff_mpp(Path(path))[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--full',action='store_true')
    p.add_argument('--size-one',action='store_true',help='Add a same-coordinate control using the size condition1 used by dynamic training')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--cells',nargs='+',choices=['legacy','unknown','instanseg'],default=['legacy','unknown','instanseg'])
    p.add_argument('--weights',nargs='+',choices=['C0','C1','C2','C3'],default=['C0','C3'])
    p.add_argument('--merge',nargs='+',type=Path);a=p.parse_args()
    if ROOT not in a.output.resolve().parents or not 0<=a.shard<a.shards or (a.smoke and a.full):raise ValueError('invalid scope/output/shard')
    if 'legacy' not in a.cells or len(set(a.cells))!=len(a.cells) or len(set(a.weights))!=len(a.weights):raise ValueError('unique conditions and legacy replay control required')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    plan=json.loads(a.plan.read_text());source=json.loads((DENSE_RUN/'run.json').read_text());refs=source['refs']['val']
    if sha256(DENSE_RUN/'run.json')!=plan['legacy_run_sha256'] or len(refs)!=1024:raise ValueError('legacy source changed')
    source_hash=sha256(Path(__file__));plan_hash=sha256(a.plan)
    try:
        if a.merge:
            rows=[];cases=[];runs=[]
            for folder in a.merge:
                status=json.loads((folder/'status.json').read_text());run=json.loads((folder/'run.json').read_text())
                if status['status']!='complete' or run['source_sha256']!=source_hash or run['plan_sha256']!=plan_hash or run['smoke']:raise ValueError('incomplete/mismatched shard')
                rows.extend(json.loads((folder/'rows.json').read_text()));cases.extend(json.loads((folder/'cases.json').read_text()));runs.append(run)
            signature=lambda r:(r['full'],r['cells'],r['weights'],r.get('size_one',False),r['dependency_sha256'])
            if any(signature(r)!=signature(runs[0]) for r in runs):raise ValueError('different contracts across shards')
            positions=list(range(1024)) if runs[0]['full'] else select_positions(refs,8)
            if len(cases)!=len(positions) or {r['position'] for r in cases}!=set(positions):raise ValueError('missing/duplicate case')
            expected={(c['position'],w,cell,prompt) for c in cases for w in ['retained',*runs[0]['weights']] for cell in runs[0]['cells'] for prompt in prompt_kinds(runs[0].get('size_one',False)) if prompt.startswith('historical') or c['interior']['status']=='available'}
            if len(rows)!=len(expected) or {(r['position'],r['weight'],r['cells'],r['prompt']) for r in rows}!=expected:raise ValueError('condition coverage mismatch')
            summary=build_summary(rows);summary.update(cases=len(cases),predictions=len(rows),maximum_replay_error=max(c['replay_error'] for c in cases),
                interior_infeasible=[c['position'] for c in cases if c['interior']['status']!='available'],plan_sha256=plan_hash,puma_used=False,test_used=False,
                scope='CRC spatial development; historical region prompts and GT-interior pixel prompts reported separately; same original RGB/GT/FOV across cell conditions')
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',summary)
        else:
            device=torch.device('cuda:0');original=json.loads((Path(source['source_cache'])/'run.json').read_text())
            metadata=json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
            da=argparse.Namespace(**original['args']);da.spatial_data=Path(da.spatial_data);da.routing=Path(da.routing)
            dataset=make_loader(da,metadata,'val').dataset.dataset
            spatial=pd.read_parquet(da.spatial_data/'spatial_manifest.parquet').set_index('patch_id')
            positions=select_positions(refs,1) if a.smoke else list(range(1024)) if a.full else select_positions(refs,8)
            positions=positions if a.smoke else positions[a.shard::a.shards]
            retained,weights=load_model(device,mapper='centroid');retained.eval().requires_grad_(False);models={'retained':retained};drift={}
            for name in a.weights:
                info=plan['checkpoints'][name]
                if sha256(Path(info['path']))!=info['sha256']:raise ValueError('checkpoint changed')
                model=copy.deepcopy(retained);saved=torch.load(info['path'],map_location='cpu',weights_only=False)
                if saved['step']!=info['step'] or saved['mapper']!='pixel':raise ValueError('checkpoint identity')
                model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);models[name]=model;del saved
                drift[name]={}
                for prefix in ['j5.phase2.backbone.layer3.','j5.phase2.backbone.layer4.','j5.phase2.embedding.','j5.phase2.assignment.','j5.phase2.semantic.','j5.cell.','j5.decoder.prompt_model.','j5.decoder.','dense.']:
                    old=dict(retained.named_parameters());new=dict(model.named_parameters());keys=[k for k in old if k.startswith(prefix)]
                    delta=sum(float((new[k]-old[k]).double().square().sum()) for k in keys);norm=sum(float(old[k].double().square().sum()) for k in keys)
                    drift[name][prefix]=dict(parameters=sum(old[k].numel() for k in keys),relative_L2=(delta/max(norm,1e-30))**.5)
            detector=ctp=xcell=cropper=None
            if 'instanseg' in a.cells:
                detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device);ctp,xcell,feature_audit=load_feature_models(device);cropper=CachedCenterCropper()
                dump(a.output/'cell_recipe.json',dict(detector_sha256=detector.weight_sha256,feature_weights={k:sha256(v) for k,v in WEIGHTS.items() if k!='cpsam'},feature_audit=feature_audit,RGB='exact original loader tensor reconstructed to uint8',MPP='native source MPP times actual1024to512 ratio',tissue_GT_used=False))
            dependencies={str(f):sha256(f) for f in [Path(inspect.getfile(load_model)),Path(inspect.getfile(prepare_variant)),Path(inspect.getfile(InstanSegDetector)),Path(inspect.getfile(load_feature_models))]}
            dump(a.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),source_sha256=source_hash,dependency_sha256=dependencies,
                plan_sha256=plan_hash,positions=positions,smoke=a.smoke,full=a.full,cells=a.cells,weights=a.weights,size_one=a.size_one,initial_weights=weights,threshold=.5,precision='FP32',puma_used=False,test_used=False))
            dump(a.output/'weight_drift.json',drift);dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=0,total=len(positions)))
            (a.output/'cases').mkdir();rows=[];cases=[]
            for number,position in enumerate(positions):
                ref=refs[position];identity=ref['identity'];pid=identity['patch_id'];batch=collate_joint_pixel_episodes([dataset[int(identity['episode_index'])]],max_cells=255)
                if str(batch['patch_id'][0])!=pid or int(batch['target_class'][0])!=identity['target_class']:raise ValueError('episode identity')
                labels=batch['pixel_gt'][0].numpy();target=identity['target_class'];truth=(batch['pixel_gt']==target).to(device);valid=(batch['pixel_gt']!=255).to(device)
                inputs=move_batch({k:batch[k] for k in IMAGE_KEYS},device);historical=move_batch({k:batch[k] for k in PROMPT_KEYS},device)
                cache=torch.load(ref['file'],map_location=device,weights_only=False)
                if not torch.equal(cache['truth'],truth) or not torch.equal(cache['valid'],valid):raise ValueError('GT replay changed')
                for k in ['positive_xy','negative_xy','positive_mask','negative_mask']:
                    if not torch.equal(cache[k],historical[k]):raise ValueError('legacy points changed')
                row=dataset.rows[pid];native=native_mpp(str(row['wsi_path']));mpp=[native[0]*row['width_level0']/512,native[1]*row['height_level0']/512]
                interior,feasibility=make_interior_prompts(labels,target,mpp,historical,position,device)
                audits={'historical':point_audit(historical,labels==target,labels!=255)}
                if interior is not None:
                    audits['interior']=point_audit(interior,labels==target,labels!=255)
                    if any(v['ignored'] or v['incorrect'] for v in audits['interior'].values()):raise ValueError('interior click quality')
                folder=a.output/'cases'/f'{position:04d}';folder.mkdir();rgb=input_rgb(inputs['image']);Image.fromarray(rgb).save(folder/'image.png');Image.fromarray(labels.astype(np.uint8)).save(folder/'labels.png')
                cell_inputs={}
                if 'legacy' in a.cells:cell_inputs['legacy']=inputs
                if 'unknown' in a.cells:cell_inputs['unknown']=unknown_cells(inputs)
                if 'instanseg' in a.cells:
                    geometry=dict(input_to_image_affine=[1.,0,0,0,1.,0],source_image_size=[512,512])
                    arrays,details=prepare_variant(rgb,mpp,geometry,detector,ctp,xcell,device,cropper)
                    np.savez_compressed(folder/'matched_cells.npz',**arrays)
                    with np.load(folder/'matched_cells.npz') as saved:
                        if any(not np.array_equal(saved[k],v) for k,v in arrays.items()):raise ValueError('matched cell roundtrip')
                    f=torch.from_numpy(arrays['features'])[None].to(device)
                    cell_inputs['instanseg']=inputs|dict(cells=f,cell_valid=torch.ones(f.shape[:2],device=device,dtype=torch.bool),total_cell_count=torch.tensor([int(arrays['total'])],device=device,dtype=torch.float32))
                    dump(folder/'cell_details.json',details)
                current=[];masks={'truth':truth.cpu().numpy()[0],'valid':valid.cpu().numpy()[0]};replay=None
                with torch.inference_mode():
                    for cell,ci in cell_inputs.items():
                        for name,model in models.items():
                            encoded=model.encode_image(ci)
                            for kind in prompt_kinds(a.size_one):
                                prompts=historical if kind.startswith('historical') else interior
                                if prompts is None:continue
                                if kind.endswith('_id1'):prompts=prompts|{'prompt_size_id':torch.ones_like(prompts['prompt_size_id'])}
                                model.mapper='centroid' if '_centroid' in kind else 'pixel';task=model.encode_support(encoded,prompts)
                                features=query_features(model.j5,encoded,task,prompts,query_id='same',support_id='same')
                                probability=model.dense(features)['pixel_probability']
                                if not torch.isfinite(probability).all():raise FloatingPointError('nonfinite prediction')
                                if name=='retained' and cell=='legacy' and kind=='historical_centroid':
                                    replay=float((probability-retained.dense(cache)['pixel_probability']).abs().max())
                                    if replay>2e-6:raise ValueError('legacy replay failed')
                                result=dict(position=position,**identity,weight=name,cells=cell,prompt=kind,prompt_size_id=int(prompts['prompt_size_id'][0]),spatial_group=str(spatial.loc[pid,'spatial_group_id']),
                                            **metrics(probability,truth,valid),prompt_violations=prompt_violations(probability,prompts))
                                current.append(result);masks[name+'|'+cell+'|'+kind]=(probability>=.5).cpu().numpy()[0]
                if 'legacy' in a.cells and replay is None:raise ValueError('missing retained replay')
                np.savez_compressed(folder/'predictions.npz',**masks)
                with np.load(folder/'predictions.npz') as saved:
                    t=saved['truth']&saved['valid'];v=saved['valid']
                    for r in current:
                        pred=saved[r['weight']+'|'+r['cells']+'|'+r['prompt']]&v
                        if (int((pred&t).sum()),int((pred&~t&v).sum()),int((~pred&t).sum()))!=(r['tp'],r['fp'],r['fn']):raise ValueError('saved mask count')
                case=dict(position=position,identity=identity,input_mpp=mpp,prompt_size_id=int(historical['prompt_size_id'][0]),interior=feasibility,prompts=audits,cell_stats={k:cell_stats(v) for k,v in cell_inputs.items()},replay_error=replay,
                          RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),cell_geometry='same original RGB/FOV',matched_cells_tissue_GT_used=False)
                dump(folder/'case.json',case);rows.extend(current);cases.append(case)
                with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(position=position,completed=number+1,seconds=time.time()-started,case=case,rows=current))+'\n')
                dump(a.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),completed=number+1,total=len(positions),seconds=time.time()-started))
                if number%5==0:print('CASE',number+1,'/',len(positions),'seconds',round(time.time()-started,1),flush=True)
            dump(a.output/'rows.json',rows);dump(a.output/'cases.json',cases);dump(a.output/'summary.json',build_summary(rows))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
