#!/usr/bin/env python3
"""Fixed-manifest Panel1 transfer: pixel segmentation and actual pixel-click curves.

Historical validation compatibility diagnostic, frozen Phase2 and Phase5/6,
no Phase4 context, no independent-test or new-class generalization claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import pyvips  # Initialize libvips before image backbone imports.
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder, project_region_probabilities
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import assignment_geometry
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss
from benchmarks.v5.phase_3_cell_region.src.panel1_controlled import (
    common_cells, fingerprint, spread, overlaps, interior_click, next_click, confusion_scores,
)
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import pixel_metrics, task_macro

V4 = Path('/nfs-medical3/zyh/v4')
P2 = V4/'phase2/runs/phase_2_region_encoder_train_20260717_161240'
P3 = V4/'phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth'
P6 = V4/'phase6/runs/phase6_mask_decoder_20260721_135527/checkpoint_epoch_001.pth'
CONTRACT = Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_token_contract_0922_201100')
ARMS = ('seg_only', 'joint_003', 'joint_010', 'panel1_pretrain', 'seg_budget')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def event(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def crc_data(args, device):
    meta = json.loads((args.contract/'metadata.json').read_text())
    manifest_path = Path(meta['manifest_dir'])/'manifest.json'
    if meta['status'] != 'complete' or sha(manifest_path) != meta['manifest_hash']:
        raise ValueError('incomplete or mismatched CRC manifest')
    index = json.loads((args.contract/'index.json').read_text())
    source = {r['sample_id']:r for r in json.loads(manifest_path.read_text())['samples']}
    groups = {}
    # Audit groups across all splits without loading test labels/features.
    for row in index:
        key = row['spatial_group_id']
        if key in groups and groups[key] != row['split']:
            raise ValueError('CRC spatial group crosses splits')
        groups[key] = row['split']
    loaded, selected = {}, {}
    for split, n in [('train', args.crc_train_per_case), ('val', args.crc_val_per_case)]:
        pool = []
        for row in index:
            if row['split'] != split:
                continue
            with np.load(args.contract/row['contract_path']) as x:
                counts = x['panel1_counts']; w = x['panel1_coverage']; obs = x['panel1_observed_cells']
                valid = (counts.sum(-1)>0) & (w>0) & (obs>=1)
                if not valid.any():
                    continue
                means = np.average(counts[valid]/counts[valid].sum(-1,keepdims=True), axis=0, weights=w[valid])
            pool.append(dict(row, case_id=row['sample_id'].split(':')[0], dominant=int(means.argmax())))
        chosen = []
        for case in sorted({r['case_id'] for r in pool}):
            keys = ('dominant','spatial_group_id') if split == 'train' else ('spatial_group_id',)
            chosen += spread([r for r in pool if r['case_id']==case], n, keys, args.data_seed)
        selected[split] = chosen
        loaded[split] = []
        for row in chosen:
            with np.load(args.contract/row['contract_path']) as x:
                item = {k:torch.from_numpy(x[k].copy()).to(device) for k in x.files}
            item['cells'] = common_cells(item['cells'])
            item['tokens'] = item['region_tokens'][None]
            item['mass'] = item['assignment_at_cells'][None]
            item['cells'] = item['cells'][None]
            item['valid'] = torch.ones(item['cells'].shape[:2], dtype=torch.bool, device=device)
            item['total'] = item['total_cell_count'].reshape(1)
            item['sample_id'] = row['sample_id']
            loaded[split].append(item)
    # Verify selected actual physical footprints, not only spatial group names.
    buffer = float(json.loads(manifest_path.read_text()).get('split_buffer_um', 256))
    for tr in selected['train']:
        for va in selected['val']:
            if tr['case_id'] != va['case_id']:
                continue
            a = source[tr['sample_id']]['context_footprint']; b = source[va['sample_id']]['context_footprint']
            if a[0]<b[2]+buffer and b[0]<a[2]+buffer and a[1]<b[3]+buffer and b[1]<a[3]+buffer:
                raise ValueError('selected CRC cross-split footprint overlap')
    coverage = {}
    for split, items in loaded.items():
        count = torch.zeros(5, device=device)
        for x in items:
            count += (x['panel1_counts'] > 0).sum(0)
        coverage[split] = count.cpu().tolist()
    return loaded, {'selected':selected, 'class_nonzero_token_counts':coverage,
                    'manifest_sha256':sha(manifest_path), 'buffer_um':buffer}


def gdph_data(args, cfg, device, out):
    patch = V4/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet'
    slic = V4/'phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet'
    datasets = {s:CellRegionDataset(patch, slic, V4/f'phase3/data/xcell_hybrid_manifest_{s}_20260719_014500_spatial/feature_routing.parquet', cfg, s) for s in ('train','val')}
    rows = {s:[dict(r, source_index=i) for i,r in enumerate(ds.region.rows)] for s,ds in datasets.items()}
    val = spread(rows['val'], args.val_patches, ('sampling_group','wsi_id'), args.data_seed)
    candidates = [r for r in rows['train'] if not any(overlaps(r,v,args.buffer) for v in val)]
    train = spread(candidates, args.train_patches, ('sampling_group','wsi_id'), args.data_seed)
    if {r['patch_id'] for r in train} & {r['patch_id'] for r in val}:
        raise ValueError('GDPH patch IDs overlap')
    selected = {'train':train, 'val':val}
    manifest = {s:[{k:r[k] for k in ('patch_id','wsi_id','source_index','sampling_group','x_level0','y_level0','width_level0','height_level0')} for r in v] for s,v in selected.items()}
    manifest.update(buffer_level0=args.buffer, excluded_train_candidates=len(rows['train'])-len(candidates),
                    scope='fine-only footprint; historical pretrained exposure not ruled out')
    p2 = DeepRegionEncoder(cfg, len(cfg['data']['class_map']), 64, 256).to(device)
    p2.load_state_dict(torch.load(P2/'checkpoint_epoch_028.pth', map_location=device, weights_only=False)['model'])
    p2.eval().requires_grad_(False)
    data = {}
    for split, chosen in selected.items():
        data[split] = []
        for i,row in enumerate(chosen):
            original = datasets[split][row['source_index']]
            batch = collate_cell_region([original], 255)
            with torch.no_grad():
                z = p2(batch['image'].to(device), False, True)
            cells = common_cells(batch['cells'][0]).to(device)[None]
            mask = batch['mask'][0].numpy()
            item = dict(tokens=z['region_tokens'], mass=sample_assignment_at_cells(z['assignment_low'],cells),
                        cells=cells, valid=batch['cell_valid'].to(device), total=batch['total_cell_count'].to(device),
                        assignment=z['assignment_low'], labels=region_labels(z['assignment_low'],batch['mask'].to(device),12,255),
                        mask=mask, patch_id=row['patch_id'], wsi_id=row['wsi_id'])
            if not (item['labels'] != 255).any():
                raise ValueError('GDPH patch has no supervised region')
            data[split].append(item)
            if split == 'val':
                rgb = batch['image'][0].numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])
                Image.fromarray((rgb.clip(0,1)*255).astype('uint8')).save(out/f'val_{i:03d}_he.png')
            if i % 16 == 0:
                event(event='prepared',split=split,patch=i+1,total=len(chosen))
    del p2
    return data, manifest


def fuse(cell,x):
    return cell(x['tokens'],x['mass'],x['cells'],x['valid'],x['total'])['fused_tokens']


def aux(cell, evidence, x):
    logits = evidence(fuse(cell,x))['structure_observed_logits'][0]
    return observed_composition_loss(logits,x['panel1_counts'],x['panel1_observed_cells'],x['panel1_coverage'])


def load_decoder(device):
    payload = torch.load(P6,map_location='cpu',weights_only=False)
    c = payload['config']['model']
    prompt = PromptRegionModel(c['region_dim'],c['prompt_heads'],c['prompt_set_layers'],c['prompt_dropout'])
    model = ContextAwareMaskDecoder(prompt,c['region_dim'],c['graph_heads'],c['graph_layers'],c['graph_neighbours'],c['graph_dropout'],True,c.get('residual_limit',1.))
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def click_batch(tokens, assignment, xy, area, clicks):
    h,w = assignment.shape[-2:]
    batch = dict(fine_tokens=tokens,region_xy=xy,region_area=area,fine_active=area>0,
                 prompt_size_id=torch.zeros(1,dtype=torch.long,device=tokens.device))
    for name,sign in [('positive',1),('negative',0)]:
        points = [(y,x) for y,x,s in clicks if s==sign]
        # Coordinates refer to real interior pixels, not old cached centroids.
        weights = torch.stack([assignment[0,:,y,x] for y,x in points])
        batch[name+'_tokens'] = (weights @ tokens[0])[None]
        batch[name+'_xy'] = tokens.new_tensor([[(x+.5)/w,(y+.5)/h] for y,x in points])[None]
        batch[name+'_mask'] = torch.ones((1,len(points)),dtype=torch.bool,device=tokens.device)
    return batch


@torch.no_grad()
def evaluate(cell, head, evidence, data, crc, decoder, out, cfg):
    cell.eval(); head.eval(); evidence.eval()
    confusion = np.zeros((12,12),dtype=np.int64)
    episodes = []; segmentation = []; noc = []
    background = set(cfg['data'].get('background_class_ids', []))
    out.mkdir(parents=True,exist_ok=False)
    for i,x in enumerate(data):
        tokens = fuse(cell,x)
        a = F.interpolate(x['assignment'],size=x['mask'].shape,mode='bilinear',align_corners=False)
        a = a/a.sum(1,keepdim=True).clamp_min(1e-8)
        probabilities = torch.einsum('bkhw,bkc->bchw',a,head(tokens).softmax(-1))
        pred = probabilities.argmax(1)[0].cpu().numpy()
        gt = x['mask']; valid = gt!=255
        cm = np.bincount(gt[valid]*12+pred[valid],minlength=144).reshape(12,12)
        confusion += cm
        np.savez_compressed(out/f'patch_{i:03d}.npz',prediction=pred.astype('uint8'),gt=gt.astype('uint8'),valid=valid)
        # Saved predictions are the numerical source of truth.
        with np.load(out/f'patch_{i:03d}.npz') as saved:
            np.testing.assert_array_equal(cm,np.bincount(saved['gt'][valid].astype('int64')*12+saved['prediction'][valid],minlength=144).reshape(12,12))
        segmentation.append(dict(patch_id=x['patch_id'],wsi_id=x['wsi_id'],**confusion_scores(cm)))
        xy, area = assignment_geometry(x['assignment'])
        for cls in sorted(set(np.unique(gt).tolist())-background-{255}):
            target = gt==cls
            if target.sum()<64 or (valid & ~target).sum()<64:
                continue
            py,px = interior_click(target & valid); ny,nx = interior_click(~target & valid)
            clicks = [(py,px,1),(ny,nx,0)]
            curve = []; masks = {}; points = {}; no_error = False
            for budget in range(2,7):
                if budget>2 and not no_error:
                    nxt = next_click(click_pred,target,valid,clicks)
                    if nxt is None:
                        no_error = True
                    else:
                        clicks.append(nxt)
                batch = click_batch(tokens,a,xy,area,clicks)
                log = decoder(batch)['logits']
                prob = project_region_probabilities(a,log)['pixel_probability'][0].cpu().numpy()
                if not np.isfinite(prob).all():
                    raise ValueError('non-finite click prediction')
                click_pred = prob>=.5
                score_valid = valid.copy()
                for y,xx,_ in clicks:
                    score_valid[y,xx] = False
                metrics = pixel_metrics(click_pred,target,score_valid)
                curve.append(dict(task_id=str(cls),patch_id=x['patch_id'],wsi_id=x['wsi_id'],budget=budget,actual_clicks=len(clicks),**metrics))
                masks[f'pred_{budget}'] = click_pred
                points[str(budget)] = [list(p) for p in clicks]
            file = out/f'click_{i:03d}_{cls}.npz'
            np.savez_compressed(file,gt=target,valid=valid,**masks)
            with np.load(file) as saved:
                for row in curve:
                    score_valid = saved['valid'].copy()
                    for y,xx,_ in points[str(row['budget'])]:
                        score_valid[y,xx] = False
                    checked = pixel_metrics(saved[f"pred_{row['budget']}"],saved['gt'],score_valid)
                    if any(checked[k]!=row[k] for k in ('tp','fp','fn','dice','iou')):
                        raise AssertionError('saved click mask metric mismatch')
            write(out/f'click_{i:03d}_{cls}.json',dict(patch_id=x['patch_id'],target_class=cls,clicks_y_x_sign=points,curve=curve))
            episodes.extend(curve)
            for threshold in (.8,.9):
                reached = [r['actual_clicks'] for r in curve if r['dice']>=threshold]
                noc.append(dict(task_id=str(cls),threshold=threshold,noc=min(reached) if reached else 7,failed=not bool(reached)))
        event(event='evaluated_patch',output=str(out),patch=i+1,total=len(data))
    if not episodes:
        raise ValueError('no valid foreground click tasks')
    av=0.; an=0
    for x in crc:
        loss,n=aux(cell,evidence,x); av+=float(loss)*n; an+=n
    summary = dict(segmentation=confusion_scores(confusion),crc_val_ce=av/an,
                   clicks={str(b):task_macro([r for r in episodes if r['budget']==b]) for b in (2,4,6)},noc={})
    for threshold in (.8,.9):
        rows=[r for r in noc if r['threshold']==threshold]
        by_class={str(c):{'mean_noc':float(np.mean([r['noc'] for r in rows if r['task_id']==c])),
                           'failure_rate':float(np.mean([r['failed'] for r in rows if r['task_id']==c]))} for c in sorted({r['task_id'] for r in rows})}
        summary['noc'][str(threshold)]={'by_class':by_class,'task_macro_noc':float(np.mean([r['mean_noc'] for r in by_class.values()])),
                                       'task_macro_failure_rate':float(np.mean([r['failure_rate'] for r in by_class.values()]))}
    write(out/'summary.json',summary); write(out/'segmentation.json',segmentation); write(out/'click_metrics.json',episodes)
    return summary


def optimizer(*models):
    return torch.optim.AdamW([p for m in models for p in m.parameters()],lr=1e-4)


def update(loss,opt,models):
    if not torch.isfinite(loss):
        raise ValueError('non-finite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for m in models for p in m.parameters()],5.,error_if_nonfinite=True)
    opt.step()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp',required=True); p.add_argument('--device',default='cuda:1')
    p.add_argument('--contract',type=Path,default=CONTRACT); p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--steps',type=int,default=200); p.add_argument('--pretrain-steps',type=int,default=200)
    p.add_argument('--train-patches',type=int,default=96); p.add_argument('--val-patches',type=int,default=48)
    p.add_argument('--crc-train-per-case',type=int,default=24); p.add_argument('--crc-val-per-case',type=int,default=12)
    p.add_argument('--data-seed',type=int,default=20260923); p.add_argument('--buffer',type=int,default=128)
    args=p.parse_args()
    if min(args.steps,args.pretrain_steps,args.train_patches,args.val_patches,args.crc_train_per_case,args.crc_val_per_case)<1:
        raise ValueError('positive budgets required')
    out=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')/f'panel1_controlled_{args.timestamp}'
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); device=torch.device(args.device)
    event(event='start',pid=os.getpid(),output=str(out))
    write(out/'status.json',{'status':'preparing','pid':os.getpid()})
    cfg=load_config(P2/'config_snapshot.json')
    crc,crc_manifest=crc_data(args,device)
    gdph,gdph_manifest=gdph_data(args,cfg,device,out)
    manifest={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'crc':crc_manifest,'gdph':gdph_manifest,
              'sources':{str(f):sha(f) for f in (P3,P6,P2/'checkpoint_epoch_028.pth')},
              'runner_sha256':sha(Path(__file__)),'helper_sha256':sha(Path(__file__).parents[1]/'src/panel1_controlled.py'),
              'metadata_policy':'zero channels 2:10 for both cohorts; x/y and HE reg64 retained',
              'checkpoint_selection':'fixed final step; no validation selection',
              'evaluation':'historical-val; frozen Phase5/6 fine-only compatibility; oracle actual pixel clicks, initial 1 positive+1 negative, max 6 total, NoC failure=7; no test labels/features loaded',
              'class_map':cfg['data']['class_map']}
    manifest['fixed_manifest_hash']=fingerprint({'crc':crc_manifest,'gdph':gdph_manifest})
    write(out/'manifest.json',manifest)
    # Train-only prior: uniform selected patches, coverage-weighted targets within patch.
    prior=[]
    for x in crc['train']:
        c=x['panel1_counts']; w=x['panel1_coverage']; k=(c.sum(-1)>0)&(w>0)&(x['panel1_observed_cells']>=1)
        prior.append(((c[k]/c[k].sum(-1,keepdim=True))*w[k,None]).sum(0)/w[k].sum())
    prior=torch.stack(prior).mean(0)
    pv=0.; pn=0
    for x in crc['val']:
        loss,n=observed_composition_loss(prior.clamp_min(1e-8).log().expand_as(x['panel1_counts']),x['panel1_counts'],x['panel1_observed_cells'],x['panel1_coverage'])
        pv+=float(loss)*n; pn+=n
    write(out/'crc_prior.json',dict(train_prior=prior.cpu().tolist(),val_ce=pv/pn,val_tokens=pn))
    write(out/'status.json',{'status':'training_and_evaluating','pid':os.getpid()})
    decoder=load_decoder(device); base=torch.load(P3,map_location=device,weights_only=False)
    results=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        main_schedule=rng.integers(0,len(gdph['train']),args.steps).tolist()
        warm_schedule=rng.integers(0,len(gdph['train']),args.pretrain_steps).tolist()
        crc_schedule=rng.integers(0,len(crc['train']),args.steps+args.pretrain_steps).tolist()
        write(out/f'schedule_{seed}.json',dict(main=main_schedule,extra_seg=warm_schedule,crc=crc_schedule))
        for arm in args.arms:
            torch.manual_seed(seed)
            cell=CellToRegionAttention(256,74).to(device); cell.load_state_dict(base['cell'])
            head=torch.nn.Linear(256,12).to(device); head.load_state_dict(base['head'])
            evidence=ObservedRegionEvidenceHeads().to(device)
            armout=out/f'{seed}_{arm}'; armout.mkdir()
            init_hash=state_hash(cell); trace=[]
            if not results:
                initial=evaluate(cell,head,evidence,gdph['val'],crc['val'],decoder,out/'initial',cfg)
                write(out/'initial_metrics.json',initial)
            if arm in ('panel1_pretrain','seg_budget'):
                warm_models=(cell,evidence) if arm=='panel1_pretrain' else (cell,head)
                opt=optimizer(*warm_models)
                for step in range(args.pretrain_steps):
                    cell.train(); head.train(); evidence.train(); opt.zero_grad(set_to_none=True)
                    if arm=='panel1_pretrain':
                        value,_=aux(cell,evidence,crc['train'][crc_schedule[step]])
                    else:
                        x=gdph['train'][warm_schedule[step]]
                        value=F.cross_entropy(head(fuse(cell,x)).flatten(0,1),x['labels'].flatten(),ignore_index=255)
                    update(value,opt,warm_models)
                    trace.append(dict(stage='pretrain' if arm=='panel1_pretrain' else 'extra_seg',step=step+1,loss=float(value.detach())))
                torch.save({'cell':cell.state_dict(),'head':head.state_dict(),'evidence':evidence.state_dict(),'metadata_policy':manifest['metadata_policy']},armout/'pretrain.pth')
            opt=optimizer(cell,head,evidence)
            weight={'joint_003':.03,'joint_010':.1}.get(arm,0.)
            for step,index in enumerate(main_schedule):
                cell.train(); head.train(); evidence.train(); opt.zero_grad(set_to_none=True)
                x=gdph['train'][index]
                seg=F.cross_entropy(head(fuse(cell,x)).flatten(0,1),x['labels'].flatten(),ignore_index=255)
                auxiliary=seg*0
                if weight:
                    auxiliary,_=aux(cell,evidence,crc['train'][crc_schedule[args.pretrain_steps+step]])
                value=seg+weight*auxiliary
                update(value,opt,(cell,head,evidence))
                trace.append(dict(stage='main',step=step+1,seg=float(seg.detach()),aux=float(auxiliary.detach()),loss=float(value.detach())))
                if (step+1)%50==0: event(event='train',seed=seed,arm=arm,step=step+1)
            write(armout/'trace.json',trace)
            trained_hash=state_hash(cell)
            torch.save(dict(cell=cell.state_dict(),head=head.state_dict(),evidence=evidence.state_dict(),seed=seed,arm=arm,steps=args.steps,
                            fixed_manifest_hash=manifest['fixed_manifest_hash'],metadata_policy=manifest['metadata_policy']),armout/'final.pth')
            reloaded=torch.load(armout/'final.pth',map_location=device,weights_only=False)
            cell.load_state_dict(reloaded['cell']); head.load_state_dict(reloaded['head']); evidence.load_state_dict(reloaded['evidence'])
            if state_hash(cell)!=trained_hash: raise AssertionError('checkpoint reload mismatch')
            metrics=evaluate(cell,head,evidence,gdph['val'],crc['val'],decoder,armout/'evaluation',cfg)
            row=dict(seed=seed,arm=arm,initial_cell_hash=init_hash,final_cell_hash=trained_hash,main_schedule_hash=fingerprint(main_schedule),metrics=metrics)
            results.append(row); write(out/'results.json',results)
            event(event='arm_complete',seed=seed,arm=arm,seg_dice=metrics['segmentation']['macro_dice'],click6=metrics['clicks']['6']['task_macro_dice'])
    write(out/'status.json',dict(status='complete',pid=os.getpid(),arms=len(results),test_evaluated=False))
    event(event='complete',output=str(out),arms=len(results))


if __name__=='__main__':
    main()
