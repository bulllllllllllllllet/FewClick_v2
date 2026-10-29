"""Full-pool Dense continuation versus synchronized RGB-only color augmentation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import torch
import torch.nn.functional as F

from .adapted_models import build
from .diagnose_local_errors import read,write,sha
from .local_data import LocalData
from .metrics import score
from .dense_hardnegative_repair import state_sha
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.multidomain_training.losses import mask_loss,point_loss

ROOT=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
MATCHED=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/matched_dense_diagnosis_1029_181751')
ARMS=('continued_original_loss','color_augmentation')
SEED=20261029


def training_pool(data):
    pool={}
    for p in data.training:
        k=p['dataset'],p['query_id'],p['target']
        if k not in pool:pool[k]=p
    return sorted(pool.values(),key=lambda p:hashlib.sha256(f"dense-generalization-v1|{p['dataset']}|{p['query_id']}|{p['target']}".encode()).hexdigest())


def color(raw,number,identity=False):
    rng=np.random.default_rng(SEED+number)
    gain=np.ones(3) if identity else rng.uniform(.9,1.1,3)
    contrast=1. if identity else float(rng.uniform(.9,1.1))
    brightness=0. if identity else float(rng.uniform(-8,8))
    result=dict(raw);changes=[]
    for scale in range(3):
        suffix='' if scale==0 else f'_{scale}'
        image=raw['rgb'+suffix][0].permute(1,2,0).cpu().numpy()
        aug=np.rint(((image.astype(np.float64)-127.5)*contrast+127.5)*gain+brightness).clip(0,255).astype(np.uint8)
        result['image'+suffix]=he_to_tensor(aug)[None].to(raw['image'+suffix].device)
        changes.append(float(np.mean(image!=aug)))
    return result,dict(gain=gain.tolist(),contrast=contrast,brightness=brightness,changed_fraction=changes)


def objective(head,features,sup,prompts):
    out=head(features);loss,_=mask_loss(out,sup['truth'],sup['valid'],sup['mpp'])
    return loss+.1*center_point_loss(out['pixel_logits'],prompts)


def center_point_loss(logits,prompts):
    losses=[];height,width=logits.shape[-2:]
    for sign,label in [('positive',1.),('negative',0.)]:
        xy=prompts[sign+'_xy']*logits.new_tensor([width,height])-.5
        index=xy.round().long()
        assert torch.equal(xy,index.to(xy.dtype)), 'This equivalent path requires exact pixel centers'
        values=logits.flatten(1).gather(1,index[:,:,1]*width+index[:,:,0])
        valid=prompts[sign+'_mask'];assert valid.any()
        losses.append(F.binary_cross_entropy_with_logits(values[valid],torch.full_like(values[valid],label)))
    return torch.stack(losses).mean()


def optimize(head,opt,items):
    head.train();opt.zero_grad(set_to_none=True);losses=[]
    for f,s,p in items:
        loss=objective(head,f,s,p);assert torch.isfinite(loss)
        (loss/len(items)).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
    assert torch.isfinite(norm) and norm>0;opt.step()
    return dict(loss=float(np.mean(losses)),gradient_norm=float(norm))


def evaluate(model,heads,data,plans,device,output,extract):
    output.mkdir(exist_ok=False);rows=[];replay_domains=set()
    for h in heads.values():h.eval()
    for i,p in enumerate(plans):
        raw,prompts,sup=data.materialize(p,device,True,True);features,direct=extract(raw,prompts)
        if i<2 or p['dataset'] not in replay_domains:
            torch.save({k:v.detach().cpu() for k,v in features.items()},output/f'replay_features_{i:04d}.pth')
            replay_domains.add(p['dataset'])
        with torch.no_grad():
            for arm,head in heads.items():
                prob=head(features)['pixel_probability'][0].cpu().numpy()
                if arm=='original':assert np.array_equal(prob,direct[0].cpu().numpy())
                truth=sup['truth'][0].cpu().numpy();valid=sup['valid'][0].cpu().numpy()
                metric=score(prob>=.5,truth,valid,sup['mpp']);assert np.isfinite(prob).all()
                name=f'{i:04d}_{arm}.npz';np.savez_compressed(output/name,probability=prob,truth=truth,valid=valid)
                rows.append(dict(index=i,arm=arm,dataset=p['dataset'],target=p['target'],query_id=p['query_id'],
                    GT_fraction=float(truth[valid].sum()/valid.sum()),**metric,artifact_sha256=sha(output/name)))
        if (i+1)%20==0:print(f'eval {output.name} {i+1}/{len(plans)}',flush=True)
    write(output/'rows.json',rows);return rows


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0');a.output.mkdir(exist_ok=False)
    def status(stage,**kw):write(a.output/'status.json',dict(status=stage,worker_pid=os.getpid(),**kw))
    data=LocalData(ROOT/'tasks_1025_125400',ROOT/'realized_train_1025_125928');pool=training_pool(data)
    assert len(pool)==4733 and len({p['query_id'] for p in pool})==1356
    plans=pool
    if a.gate:
        plans=[]
        for domain in ('crc','tiger'):
            seen=set()
            for p in pool:
                if p['dataset']==domain and p['query_id'] not in seen:
                    plans.append(p);seen.add(p['query_id'])
                    if len(seen)==2:break
        plans=plans*4
    if not a.gate:
        receipt=read(a.gate_acceptance/'independent_acceptance.json')
        assert receipt['status']=='accepted_dense_generalization_gate'
        assert receipt['producer_source_sha256']==sha(__file__)
    best=read(ROOT/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
    assert sha(best['path'])==best['checkpoint_sha256']
    write(a.output/'training_plans.json',plans)
    contract=dict(source_sha256=sha(__file__),checkpoint=best,training_source_sha256=data.realized_sha,training_plans_sha256=sha(a.output/'training_plans.json'),
        full_pool_tasks=len(pool),full_pool_fields=1356,actual_tasks_per_arm=len(plans),updates_per_arm=(len(plans)+3)//4,gate=a.gate,
        worker_pid=os.getpid(),device='physical GPU3 FP32',threshold=.5,lr=1e-5,weight_decay=1e-4,accumulate=4,
        task_contract_sha256=sha(data.tasks/'contract.json'),seed=SEED,selection='first actual task per image-target, full pool hash order, one pass, final only',
        supervision='extra Dense-only continuation; not the original equal40k comparison',
        augmentation='synchronized RGB gain/contrast/brightness across3 views; fixed cell features, no spatial change')
    write(a.output/'contract.json',contract);status('loading_original_model')
    model=build('fewclick_m3d_xcell',device);saved=torch.load(best['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);del saved;model.eval();model.requires_grad_(False)
    frozen_sha=state_sha(model);initial=copy.deepcopy(model.core.dense);initial_sha=state_sha(initial)
    torch.save(initial.state_dict(),a.output/'initial_dense.pth')
    heads={arm:copy.deepcopy(initial).requires_grad_(True) for arm in ARMS}
    opts={arm:torch.optim.AdamW(heads[arm].parameters(),lr=1e-5,weight_decay=1e-4) for arm in ARMS}
    captured={}
    hook=model.core.dense.register_forward_pre_hook(lambda m,args:captured.update(features={k:v.detach() for k,v in args[0].items()}))
    def extract(raw,prompts):
        with torch.no_grad():direct=model(raw,prompts)['pixel_probability'].detach()
        return captured['features'],direct
    native_checks={};history=[];consumed={arm:[] for arm in ARMS}
    for start in range(0,len(plans),4):
        batches={arm:[] for arm in ARMS};augmentations=[]
        for number,p in enumerate(plans[start:start+4],start):
            raw,prompts,sup=data.materialize(p,device,True,True)
            if p['dataset'] not in native_checks:
                identity,_=color(raw,number,True)
                assert all(torch.equal(identity[k],raw[k]) for k in ('image','image_1','image_2'))
                probe=torch.linspace(-3,3,512*512,device=device).reshape(1,512,512)
                assert torch.equal(center_point_loss(probe,prompts),point_loss(probe,prompts))
                cpu_prompts={k:v.cpu() for k,v in prompts.items()}
                left=probe.cpu().requires_grad_(True);right=left.detach().clone().requires_grad_(True)
                center_point_loss(left,cpu_prompts).backward();point_loss(right,cpu_prompts).backward()
                assert torch.equal(left.grad,right.grad)
            clean,direct=extract(raw,prompts)
            with torch.no_grad():assert torch.equal(initial(clean)['pixel_probability'],direct)
            augmented,params=color(raw,number)
            assert max(params['changed_fraction'])>0
            noisy,_=extract(augmented,prompts)
            if p['dataset'] not in native_checks:
                assert not torch.equal(noisy['deep'],clean['deep'])
                native_checks[p['dataset']]=dict(identity_inputs_exact=True,cached_dense_exact=True,color_changes_deep=True,point_loss_GPU_forward_exact=True,point_loss_CPU_gradient_exact=True)
            batches[ARMS[0]].append((clean,sup,prompts));batches[ARMS[1]].append((noisy,sup,prompts))
            augmentations.append(params)
            for arm in ARMS:consumed[arm].append(p)
        step=start//4+1;item=dict(step=step,tasks=min(start+4,len(plans)),augmentations=augmentations)
        for arm in ARMS:item[arm]=optimize(heads[arm],opts[arm],batches[arm])
        history.append(item)
        if step%10==0 or a.gate:
            write(a.output/'history.json',history);status('training',step=step,total_steps=contract['updates_per_arm'],tasks=item['tasks'])
            print(json.dumps(dict(step=step,tasks=item['tasks'],seconds=time.monotonic()-started)),flush=True)
    assert state_sha(model)==frozen_sha
    checks={}
    for arm in ARMS:
        assert state_sha(heads[arm])!=initial_sha
        folder=a.output/arm;folder.mkdir();path=folder/'checkpoint.pth'
        torch.save(dict(dense=heads[arm].state_dict(),optimizer=opts[arm].state_dict(),contract=contract,arm=arm,initial_sha256=initial_sha),path)
        fresh=copy.deepcopy(initial).requires_grad_(True);ck=torch.load(path,map_location=device,weights_only=False);fresh.load_state_dict(ck['dense'],strict=True)
        fresh_opt=torch.optim.AdamW(fresh.parameters(),lr=1e-5,weight_decay=1e-4);fresh_opt.load_state_dict(ck['optimizer'])
        heads[arm].eval();fresh.eval()
        with torch.no_grad():assert torch.equal(heads[arm](batches[arm][0][0])['pixel_probability'],fresh(batches[arm][0][0])['pixel_probability'])
        if a.gate:
            optimize(heads[arm],opts[arm],[batches[arm][0]]);optimize(fresh,fresh_opt,[batches[arm][0]])
            assert state_sha(heads[arm])==state_sha(fresh)
            heads[arm].load_state_dict(ck['dense'],strict=True)
        checks[arm]=dict(checkpoint_sha256=sha(path),initial_sha256=initial_sha,reload_exact=True,optimizer_continuation_exact=a.gate)
        write(folder/'consumed.json',consumed[arm]);assert sha(folder/'consumed.json')==sha(a.output/'training_plans.json')
    write(a.output/'history.json',history);write(a.output/'native_checks.json',native_checks)
    val=data.validation
    if a.gate:val=[next(p for p in val if p['dataset']==d) for d in ('crc','tiger')]
    eval_heads={'original':initial,**heads}
    allrows=evaluate(model,eval_heads,data,val,device,a.output/'development',extract)
    train_eval=[] if a.gate else [p['train'] for p in read(MATCHED/'selection.json')['pairs']]
    if train_eval:evaluate(model,eval_heads,data,train_eval,device,a.output/'matched_training',extract)
    hook.remove();assert state_sha(model)==frozen_sha
    groups={}
    for arm in ('original',)+ARMS:
        groups[arm]={}
        for label in ('all','small','large'):
            rr=[r for r in allrows if r['arm']==arm and (label=='all' or (r['GT_fraction']<.05 if label=='small' else r['GT_fraction']>=.5))]
            if rr:groups[arm][label]=dict(n=len(rr),dice=float(np.mean([r['dice'] for r in rr])),bf1=float(np.mean([r['bf1_um2'] for r in rr if r['bf1_um2'] is not None])),
                precision=float(np.mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0 for r in rr])),recall=float(np.mean([r['tp']/(r['tp']+r['fn']) for r in rr])))
    write(a.output/'summary.json',dict(status='complete_gate_pending_acceptance' if a.gate else 'complete_training_pending_acceptance',worker_pid=os.getpid(),groups=groups,checks=checks,
        frozen_model_sha256=frozen_sha,initial_dense_sha256=initial_sha,native_checks=native_checks,seconds=time.monotonic()-started,peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    status('complete_execution_pending_independent_acceptance');print('COMPLETE',time.monotonic()-started,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--gate-acceptance',type=Path);a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        a.output.mkdir(exist_ok=True);write(a.output/'failure.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()));raise
