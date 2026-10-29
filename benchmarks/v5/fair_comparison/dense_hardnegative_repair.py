"""Isolated Dense-only training pilot: matched continuation versus hard negatives."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import torch
import torch.nn.functional as F

from .adapted_models import build
from .diagnose_local_errors import sha, write, read
from .local_data import LocalData
from .metrics import score
from benchmarks.v5.multidomain_training.losses import mask_loss, point_loss


def hard_negative(logits, truth, valid, fraction=.1):
    neg = logits.float()[valid.bool() & ~truth.bool()]
    if not neg.numel():
        return logits.sum()*0.
    return F.softplus(neg).topk(max(1, math.ceil(fraction*neg.numel()))).values.mean()


def state_sha(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def plans(a, data):
    training=[json.loads(line) for line in (a.root.parent/'realized_train_1025_125928/training_plans.jsonl').open()]
    rows=[json.loads(line) for line in (a.root/'evaluate_fewclick_m3d_xcell_1025_131451/rows.jsonl').open()]
    val=[r for r in rows if r['click_budget']=='3+2' and r['prompt_group']==0]
    if a.gate:
        tr=[];va=[]
        for domain in ('crc','tiger'):
            seen=set()
            for p in training:
                if p['dataset']==domain and p['query_id'] not in seen:
                    seen.add(p['query_id']);tr.append(p)
                    if len(seen)==2:break
            va.extend([p for p in val if p['dataset']==domain][:2])
    else:
        classes=sorted({(p['dataset'],p['target']) for p in val})
        tr=[];va=[]
        for key in classes:
            tr.append(next(p for p in training if (p['dataset'],p['target'])==key))
            candidates=sorted([p for p in val if (p['dataset'],p['target'])==key],key=lambda p:(p['positive']/p['valid'],p['query_id']))
            assert len(candidates)>=2
            va.extend([candidates[0],candidates[-1]])
        assert len(tr)==16 and len(va)==32
    for p in tr:assert p['split']=='train' and data.rows[p['query_id']]['split']=='train'
    for p in va:assert p['split']=='val' and data.rows[p['query_id']]['split']=='val'
    assert not {p['query_id'] for p in tr}&{p['query_id'] for p in va}
    return tr,va


def evaluate(dense, caches, output):
    rows=[]
    for index,c in enumerate(caches):
        with torch.inference_mode():result=dense(c['features'])
        prob=result['pixel_probability'][0].numpy();truth=c['truth'][0].numpy();valid=c['valid'][0].numpy()
        assert np.isfinite(prob).all()
        metric=score(prob>=.5,truth,valid,c['mpp'])
        row=dict(index=index,dataset=c['plan']['dataset'],target=c['plan']['target'],query_id=c['plan']['query_id'],
                 GT_fraction=float((truth&valid).sum()/valid.sum()),**metric)
        rows.append(row)
        np.savez_compressed(output/f'case{index:02d}.npz',probability=prob,truth=truth,valid=valid)
    write(output/'rows.json',rows)
    return rows


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.steps>0
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();started=time.perf_counter()
    def status(s,**kw):write(a.output/'status.json',dict(status=s,worker_pid=pid,**kw))
    try:
        torch.set_num_threads(2);torch.manual_seed(20261025)
        data=LocalData(a.tasks);train,val=plans(a,data)
        best=read(a.root/'train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
        assert sha(best['path'])==best['checkpoint_sha256']
        signature=dict(source_sha256=sha(__file__),checkpoint_sha256=best['checkpoint_sha256'],task_contract_sha256=sha(a.tasks/'contract.json'),
                       hard_fraction=.1,hard_coefficient=.25,lr=1e-5,weight_decay=1e-4,threshold=.5,device='CPU float32',threads=2)
        if not a.gate:
            assert a.verified_gate is not None
            gate=read(a.verified_gate/'summary.json')
            assert gate['status']=='passed_dense_hardnegative_training_gate' and gate['signature']==signature
        write(a.output/'contract.json',dict(signature=signature,worker_pid=pid,gate=a.gate,steps_per_arm=a.steps,
            note='Extra Dense-only training; posthoc development pilot, not matched original40k performance. No validation GT enters dense inference.'))
        write(a.output/'plans.json',dict(train=train,val=val))
        status('loading_original_frozen_model',train=len(train),val=len(val))
        model=build('fewclick_m3d_xcell',torch.device('cpu'))
        saved=torch.load(best['path'],map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
        model.eval();model.requires_grad_(False)
        initial=copy.deepcopy(model.core.dense);initial_sha=state_sha(initial)
        captured={}
        def hook(module,args,output):
            captured['features']={k:v.detach().clone() for k,v in args[0].items()}
            captured['probability']=output['pixel_probability'].detach().clone()
        handle=model.core.dense.register_forward_hook(hook)
        caches={'train':[],'val':[]}
        for role,selected in [('train',train),('val',val)]:
            folder=a.output/(role+'_cache');folder.mkdir()
            for i,p in enumerate(selected):
                raw,prompts,sup=data.materialize(p,torch.device('cpu'),True,True)
                with torch.inference_mode():
                    result=model(raw,prompts)
                    # Clone out of inference mode before training cached features.
                    direct=result['pixel_probability'].clone()
                features={k:v.clone() for k,v in captured['features'].items()}
                with torch.no_grad():again=initial(features)['pixel_probability']
                assert torch.equal(again,direct),'cached Dense path must equal original CPU model'
                c=dict(features=features,truth=sup['truth'].clone(),valid=sup['valid'].clone(),mpp=sup['mpp'],prompts=prompts,plan=p)
                torch.save(c,folder/f'case{i:02d}.pth');caches[role].append(c)
                status('preparing_frozen_Dense_inputs',role=role,completed=i+1,total=len(selected))
                print(json.dumps(dict(stage='cache',role=role,item=i+1,total=len(selected))),flush=True)
        handle.remove();del model
        original=a.output/'original';original.mkdir();baseline=evaluate(initial,caches['val'],original)
        all_results={'original':baseline};checks={};order=[i%len(train) for i in range(a.steps)]
        write(a.output/'training_order.json',order)
        for arm in ('continued_original_loss','hardnegative_repair'):
            dense=copy.deepcopy(initial);dense.requires_grad_(True);dense.train()
            assert state_sha(dense)==initial_sha
            opt=torch.optim.AdamW(dense.parameters(),lr=1e-5,weight_decay=1e-4)
            folder=a.output/arm;folder.mkdir();history=[]
            for step,index in enumerate(order):
                c=caches['train'][index];opt.zero_grad(set_to_none=True);output=dense(c['features'])
                loss,parts=mask_loss(output,c['truth'],c['valid'],c['mpp'])
                loss=loss+.1*point_loss(output['pixel_logits'],c['prompts'])
                hard=hard_negative(output['pixel_logits'],c['truth'],c['valid'])
                if arm=='hardnegative_repair':loss=loss+.25*hard
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(dense.parameters(),1.)
                assert torch.isfinite(norm) and norm>0
                opt.step();history.append(dict(step=step+1,train_index=index,loss=float(loss.detach()),hard=float(hard.detach()),gradient_norm=float(norm)))
                if (step+1)%8==0:
                    write(folder/'history.json',history);status('training_Dense_only',arm=arm,step=step+1,steps=a.steps)
                    print(json.dumps(dict(stage='train',arm=arm,step=step+1,loss=history[-1]['loss'])),flush=True)
            dense.eval();assert state_sha(dense)!=initial_sha
            torch.save(dict(dense=dense.state_dict(),signature=signature,arm=arm,steps=a.steps,initial_sha256=initial_sha),folder/'checkpoint.pth')
            with torch.no_grad():before=dense(caches['val'][0]['features'])['pixel_probability']
            fresh=copy.deepcopy(initial);fresh.load_state_dict(torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=False)['dense'],strict=True);fresh.eval()
            with torch.no_grad():after=fresh(caches['val'][0]['features'])['pixel_probability']
            assert torch.equal(before,after)
            rows=evaluate(fresh,caches['val'],folder);all_results[arm]=rows
            checks[arm]=dict(initial_sha256=initial_sha,final_sha256=state_sha(fresh),checkpoint_sha256=sha(folder/'checkpoint.pth'),reload_exact=True,steps=a.steps)
        result=dict(status='passed_dense_hardnegative_training_gate' if a.gate else 'complete_dense_hardnegative_pilot_unaccepted',signature=signature,
            worker_pid=pid,train_cases=len(train),val_cases=len(val),steps_per_arm=a.steps,checks=checks,
            means={arm:{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ['dice','bf1_um2','false_positive_fraction']} for arm,rows in all_results.items()},
            seconds=time.perf_counter()-started,limitations='Fixed small posthoc local development pilot; no WSI/new external/three-seed conclusion.')
        write(a.output/'summary.json',result);status(result['status']);print(json.dumps(result),flush=True)
    except Exception:
        status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--steps',type=int,default=256)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
