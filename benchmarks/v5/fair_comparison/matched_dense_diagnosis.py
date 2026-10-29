"""Same-class, narrow-area frozen Dense train/development comparison."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import torch

from .diagnose_local_errors import read, write, sha
from .local_data import LocalData
from .adapted_models import build
from .metrics import score

ROOT=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
PILOT=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/dense_hardnegative_repair_1029_160100/pilot_1029_160900')


def fraction(data,p):
    c=json.loads(data.rows[p['query_id']]['counts_json'])
    return c[str(p['target'])]/sum(v for k,v in c.items() if k!='255')


def selection(data):
    anchors=read(PILOT/'plans.json')['val'];unique={}
    for p in data.training:
        key=(p['dataset'],p['query_id'],p['target'])
        if key in unique or len(p['positive_xy'])<3 or len(p['negative_xy'])<2:continue
        unique[key]=p|dict(positive_xy=p['positive_xy'][:3],negative_xy=p['negative_xy'][:2],click_budget='3+2')
    used=set();pairs=[];missing=[]
    for index,v in enumerate(anchors):
        vf=fraction(data,v);candidates=[]
        for key,t in unique.items():
            if key in used or (t['dataset'],t['target'])!=(v['dataset'],v['target']):continue
            tf=fraction(data,t)
            if .8<=tf/vf<=1.25 and abs(tf-vf)<=.02:
                tie=hashlib.sha256(f"matched-dense-v1|{t['dataset']}|{t['query_id']}|{t['target']}".encode()).hexdigest()
                candidates.append((abs(tf-vf),tie,key,t,tf))
        if not candidates:
            missing.append(dict(anchor=index,dataset=v['dataset'],target=v['target'],GT_fraction=vf,reason='no_unused_same_class_training_pair_inside_frozen_calipers'));continue
        _,_,key,t,tf=min(candidates,key=lambda x:x[:2]);used.add(key)
        pairs.append(dict(pair=len(pairs),anchor=index,train=t,val=v,train_fraction=tf,val_fraction=vf))
    return dict(pairs=pairs,missing=missing,total_anchors=len(anchors))


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261025)
    data=LocalData(ROOT/'tasks_1025_125400',ROOT/'realized_train_1025_125928')
    selected=selection(data);write(a.output/'selection.json',selected)
    if a.plan_only:
        print(json.dumps(dict(pairs=len(selected['pairs']),missing=selected['missing'])));return
    assert selected['pairs']
    old_accept=PILOT.parent/'pilot_acceptance_1029_160900/summary.json'
    assert read(old_accept)['status']=='accepted_dense_hardnegative_pilot_qualified'
    best=read(ROOT/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
    assert sha(best['path'])==best['checkpoint_sha256']==read(PILOT/'contract.json')['signature']['checkpoint_sha256']
    sources=[Path(__file__),Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/phase_6_mask_decoder/dense/dense_prompt_decoder.py')]
    write(a.output/'contract.json',dict(worker_pid=os.getpid(),checkpoint=best,source_sha256={str(p):sha(p) for p in sources},
        selection_sha256=sha(a.output/'selection.json'),task_contract_sha256=sha(data.tasks/'contract.json'),training_sha256=data.realized_sha,
        original_pilot_acceptance_sha256=sha(old_accept),calipers=dict(ratio=[.8,1.25],absolute_fraction=.02),
        selection='fixed32anchors; nearest area within same class; hash tie; no training pair reuse',threshold=.5,device='CPU FP32',threads=2))
    model=build('fewclick_m3d_xcell',torch.device('cpu'));saved=torch.load(best['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);del saved;model.eval();model.requires_grad_(False)
    captured={}
    handle=model.core.dense.register_forward_pre_hook(lambda m,args:captured.update(base=args[0]['base'].detach().clone()))
    rows=[];gates={};bindings={}
    for pair in selected['pairs']:
        index=pair['pair'];v=pair['val'];anchor=pair['anchor']
        cp=PILOT/'val_cache'/f'case{anchor:02d}.pth';npz=PILOT/'original'/f'case{anchor:02d}.npz'
        bindings[str(cp)]=sha(cp);bindings[str(npz)]=sha(npz)
        cache=torch.load(cp,map_location='cpu',weights_only=False)
        assert all(cache['plan'][k]==v[k] for k in ('query_id','target','dataset','positive_xy','negative_xy'))
        with np.load(npz,allow_pickle=False) as z:
            base=cache['features']['base'][0].numpy();final=z['probability'].copy();truth=z['truth'].copy();valid=z['valid'].copy()
        assert np.array_equal(cache['truth'][0].numpy(),truth) and np.array_equal(cache['valid'][0].numpy(),valid)
        if v['dataset'] not in gates:
            raw,prompts,sup=data.materialize(v,torch.device('cpu'),True,True)
            with torch.inference_mode():direct=model(raw,prompts)['pixel_probability'][0].numpy()
            assert np.array_equal(direct,final) and np.array_equal(captured['base'][0].numpy(),base)
            gates[v['dataset']]=dict(anchor=anchor,base_exact=True,dense_exact=True)
        for role in ('val','train'):
            if role=='train':
                raw,prompts,sup=data.materialize(pair['train'],torch.device('cpu'),True,True)
                with torch.inference_mode():out=model(raw,prompts)
                final=out['pixel_probability'][0].numpy();base=captured['base'][0].numpy()
                truth=sup['truth'][0].numpy();valid=sup['valid'][0].numpy();mpp=sup['mpp']
            else:mpp=cache['mpp']
            metric={name:score(p>=.5,truth,valid,mpp) for name,p in [('base',base),('dense',final)]}
            assert np.isfinite(base).all() and np.isfinite(final).all()
            file=a.output/f'pair{index:02d}_{role}.npz';np.savez_compressed(file,base=base,dense=final,truth=truth,valid=valid)
            row=dict(pair=index,anchor=anchor,role=role,dataset=v['dataset'],target=v['target'],query_id=pair[role]['query_id'],
                GT_fraction=float(truth[valid].sum()/valid.sum()),metrics=metric,artifact_sha256=sha(file),
                dense_gain=metric['dense']['dice']-metric['base']['dice'])
            rows.append(row)
        write(a.output/'rows.json',rows);write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed_pairs=index+1,total_pairs=len(selected['pairs'])))
        print(f"{index+1}/{len(selected['pairs'])} {v['dataset']} class{v['target']} areas {pair['train_fraction']:.5f}/{pair['val_fraction']:.5f} gains train={rows[-1]['dense_gain']:.4f} val={rows[-2]['dense_gain']:.4f}",flush=True)
    handle.remove();groups={}
    for label in ('all','small','non_small'):
        eligible={p['pair'] for p in selected['pairs'] if label=='all' or (p['val_fraction']<.05)==(label=='small')}
        groups[label]={}
        for role in ('train','val'):
            rr=[r for r in rows if r['role']==role and r['pair'] in eligible]
            if rr:groups[label][role]=dict(n=len(rr),base_dice=float(np.mean([r['metrics']['base']['dice'] for r in rr])),dense_dice=float(np.mean([r['metrics']['dense']['dice'] for r in rr])),dense_gain=float(np.mean([r['dense_gain'] for r in rr])))
    write(a.output/'summary.json',dict(status='complete_matched_diagnostic_pending_independent_acceptance',worker_pid=os.getpid(),pairs=len(selected['pairs']),missing=selected['missing'],
        groups=groups,gates=gates,reused_bindings=bindings,seconds=time.monotonic()-started,rows_sha256=sha(a.output/'rows.json'),
        limitations='Fixed class-area extreme development anchors. Same class/narrow area, not matched morphology, click geometry, or patient. Training prompts derive from seen tasks. Not a blind or population overfit test.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception:
        import traceback
        write(a.output/'failure.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()));raise
