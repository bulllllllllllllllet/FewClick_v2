"""Frozen-model, area-stratified prompt/region diagnostic; no model selection."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import torch
import torch.nn.functional as F

from .diagnose_local_errors import read, write, sha
from .adapted_models import build
from .local_data import LocalData, fewclick_inputs
from .metrics import score
from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells


ROOT=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')


def band(f):
    return 'small' if f<.05 else 'medium_small' if f<.2 else 'medium_large' if f<.5 else 'large'


def select(data):
    pools={}
    for role,plans in [('train',data.training),('val',data.validation)]:
        unique={}
        for p in plans:
            k=(p['dataset'],p['query_id'],p['target'])
            if k in unique or len(p['positive_xy'])<3 or len(p['negative_xy'])<2:continue
            counts=__import__('json').loads(data.rows[p['query_id']]['counts_json'])
            fraction=counts[str(p['target'])]/sum(v for k,v in counts.items() if k!='255')
            unique[k]=p|dict(positive_xy=p['positive_xy'][:3],negative_xy=p['negative_xy'][:2],
                click_budget='3+2',role=role,area_band=band(fraction),GT_fraction=fraction)
        grouped=defaultdict(list)
        for p in unique.values():grouped[p['dataset'],p['area_band']].append(p)
        pools[role]=[]
        for k,values in sorted(grouped.items()):
            values.sort(key=lambda p:hashlib.sha256(f"prompt-region-v1|{p['dataset']}|{p['query_id']}|{p['target']}".encode()).hexdigest())
            pools[role].extend(values[:4])
    assert not {p['query_id'] for p in pools['train']}&{p['query_id'] for p in pools['val']}
    return pools


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261025)
    data=LocalData(ROOT/'tasks_1025_125400',ROOT/'realized_train_1025_125928')
    selected=select(data);write(a.output/'plans.json',selected)
    best=read(ROOT/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
    assert sha(best['path'])==best['checkpoint_sha256']
    write(a.output/'contract.json',dict(worker_pid=os.getpid(),device='CPU FP32',threads=2,checkpoint=best,
        plans_sha256=sha(a.output/'plans.json'),source_sha256=sha(__file__),task_contract_sha256=sha(data.tasks/'contract.json'),
        training_sha256=data.realized_sha,threshold=.5,selection='domain-area SHA order; 4 unique pairs/cell; no predictions',
        purity='GT spatial fraction under soft assignment on64x64 feature grid, not semantic feature purity',
        inference_GT=False))
    model=build('fewclick_m3d_xcell',torch.device('cpu'))
    saved=torch.load(best['path'],map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    model.eval();model.requires_grad_(False);rows=[];gates={};captured={}
    def capture(module,args):captured['base']=args[0]['base'].detach().clone()
    handle=model.core.dense.register_forward_pre_hook(capture)
    for role,plans in selected.items():
        for p in plans:
            index=len(rows);raw,prompts,sup=data.materialize(p,torch.device('cpu'),True,True)
            with torch.inference_mode():
                enc=model.core.encode_image(fewclick_inputs(raw));task=model.core.support_task(enc,prompts)
                out,_=model.core.decode(enc,task,prompts,True);base=captured['base'].clone()
                if p['dataset'] not in gates:
                    direct=model(raw,prompts)['pixel_probability']
                    assert torch.equal(direct,out['pixel_probability'])
                    gates[p['dataset']]=dict(case=index,probability_exact=True)
                assignment=enc['assignment_low'].float()
                truth=sup['truth'][0].numpy();valid=sup['valid'][0].numpy()
                target_grid=F.avg_pool2d(sup['truth'][:,None].float(),8)
                valid_grid=F.avg_pool2d(sup['valid'][:,None].float(),8)
                mass=(assignment*valid_grid).sum((-2,-1))
                region_purity=(assignment*target_grid).sum((-2,-1))/mass.clamp_min(1e-8)
                values={};weights={}
                for sign in ('positive','negative'):
                    w=sample_assignment_at_cells(assignment,prompts[sign+'_xy']);w=w/w.sum(1,keepdim=True).clamp_min(1e-8)
                    purity=torch.einsum('bkp,bk->bp',w,region_purity)
                    weights[sign]=w[0].numpy();values[sign+'_purity']=purity[0].tolist()
                final=out['pixel_probability'][0].numpy();base=base[0].numpy()
                metrics={name:score(prob>=.5,truth,valid,sup['mpp']) for name,prob in [('base',base),('dense',final)]}
                pos=np.array(values['positive_purity']);neg=np.array(values['negative_purity'])
                row=dict(index=index,role=role,dataset=p['dataset'],query_id=p['query_id'],target=p['target'],
                    area_band=p['area_band'],GT_fraction=p['GT_fraction'],**values,
                    purity_separation=float(pos.mean()-neg.mean()),metrics=metrics,
                    positive_click_correct=float(np.mean(final[np.floor(np.array(p['positive_xy'])[:,1]*512).astype(int),np.floor(np.array(p['positive_xy'])[:,0]*512).astype(int)]>=.5)))
                file=a.output/f'case{index:03d}.npz'
                np.savez_compressed(file,assignment=assignment[0].numpy(),truth=truth,valid=valid,
                    positive_xy=np.array(p['positive_xy']),negative_xy=np.array(p['negative_xy']),
                    positive_weights=weights['positive'],negative_weights=weights['negative'],
                    base=base,dense=final,region_purity=region_purity[0].numpy())
                row['artifact_sha256']=sha(file);rows.append(row)
            write(a.output/'rows.json',rows)
            write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(rows),total=sum(map(len,selected.values()))))
            print(f'{len(rows)}/{sum(map(len,selected.values()))} {role} {p["dataset"]} {p["area_band"]} Dice={metrics["dense"]["dice"]:.4f}',flush=True)
    handle.remove()
    groups={}
    for role in ('train','val'):
        for area in ('all','small','medium_small','medium_large','large'):
            rr=[r for r in rows if r['role']==role and (area=='all' or r['area_band']==area)]
            if not rr:continue
            groups[f'{role}|{area}']=dict(n=len(rr),base_dice=float(np.mean([r['metrics']['base']['dice'] for r in rr])),
                dense_dice=float(np.mean([r['metrics']['dense']['dice'] for r in rr])),
                positive_purity=float(np.mean([np.mean(r['positive_purity']) for r in rr])),
                negative_purity=float(np.mean([np.mean(r['negative_purity']) for r in rr])),
                purity_separation=float(np.mean([r['purity_separation'] for r in rr])))
    write(a.output/'summary.json',dict(status='complete_diagnostic_pending_independent_acceptance',worker_pid=os.getpid(),
        n=len(rows),groups=groups,native_gates=gates,seconds=time.monotonic()-started,
        rows_sha256=sha(a.output/'rows.json'),limitations='Single-seed stratified local development diagnostic; area/domain matched by sampling, not patient or full-class matched. Train uses prefix of previously seen prompts. Spatial purity is not causal proof of semantic information loss.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        write(a.output/'failure.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()));raise
