"""One preregistered Phikon frozen-vs-last-block training and evaluation recipe."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .continuous_data import sha
from .continuous_prompt import mask_loss, tree_sha
from .gate_phikon_joint import Tail
from .joint_data import JointData
from .shared_prompt import SharedPromptDecoder
from .train_cell_only_pilot import gate_indices
from benchmarks.v5.fair_comparison.metrics import score

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
TASK=BASE/'phikon_joint_tasks_1031_222610/data'
PREFIX=BASE/'phikon_prefix_1031_223355/full'
PARENT=BASE/'task_diagnostic_run_1031_023222/mixed/checkpoint.pt'


def sources():
    return {name:sha(Path(__file__).with_name(name)) for name in
            ['train_joint_pilot.py','joint_data.py','shared_prompt.py','continuous_prompt.py','gate_phikon_joint.py','continuous_data.py']}


def evaluation(data, plans, gate):
    if not gate:
        return list(data.validation())
    result=[('gate_train','recorded',p) for p in plans]
    seen=set()
    for scope,budget,p in data.validation():
        key=(scope,budget,p['dataset'])
        if key not in seen:
            seen.add(key); result.append((scope,budget,p))
    assert len(seen)==12
    return result


def initialize(arm):
    state=torch.load(PARENT,map_location='cpu',weights_only=True)
    assert state['updates']==2000
    full=AutoModel.from_pretrained('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2',local_files_only=True)
    tail=Tail(full).cuda().eval().requires_grad_(arm=='joint'); del full
    head=SharedPromptDecoder(1024).cuda(); head.load_state_dict(state['model'],strict=True)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    opt.load_state_dict(state['optimizer'])
    assert tree_sha(opt.state_dict())==tree_sha(state['optimizer'])
    for group in opt.param_groups: group['lr']=1e-4
    if arm=='joint': opt.add_param_group(dict(params=list(tail.parameters()),lr=1e-5,weight_decay=1e-4))
    return head,tail,opt


def update(data,head,tail,opt,plans):
    opt.zero_grad(set_to_none=True); losses=[]
    for p in plans:
        logits=head(**data.model_inputs(p,tail,'cuda:0'))
        truth,valid=data.supervision(p,'cuda:0')
        loss=mask_loss(logits,truth,valid)
        assert torch.isfinite(loss)
        (loss/len(plans)).backward(); losses.append(float(loss.detach()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    if any(p.requires_grad for p in tail.parameters()):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tail.parameters())
    else: assert all(p.grad is None for p in tail.parameters())
    opt.step(); return losses


def accept(directory):
    report=json.loads((directory/'report.json').read_text())
    assert report['sources']==sources() and report['checkpoint_sha256']==sha(directory/'checkpoint.pt')
    assert report['parent_sha256']==sha(PARENT) and report['training_sha256']==sha(TASK/'training4000.jsonl')
    lines=(TASK/'training4000.jsonl').read_text().splitlines(keepends=True)
    selected=gate_indices(lines)
    used=[lines[i] for i in selected] if report['gate'] else lines
    assert (directory/'training_consumed.jsonl').read_text()==''.join(used)
    data=JointData(PREFIX); assert data.identity==report['inputs']
    expected=evaluation(data,[json.loads(s) for s in used],report['gate'])
    rows=json.loads((directory/'rows.json').read_text()); masks=np.load(directory/'packed_masks.npz',allow_pickle=False)['masks']
    assert len(rows)==len(expected)==(20 if report['gate'] else 3711)
    assert masks.shape==(len(rows),32768)
    for i,(row,(scope,budget,p)) in enumerate(zip(rows,expected)):
        assert row['scope']==scope and row['budget']==budget and row['plan']==p
        mask=np.unpackbits(masks[i],count=512*512).reshape(512,512).astype(bool)
        gt,mpp=data.labels(p['query_id'])
        assert row['metrics']==score(mask,gt==p['target'],gt!=255,mpp)
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert state['updates']==2000+(2 if report['gate'] else 1000)
    steps=[int(v['step']) for v in state['optimizer']['state'].values()]
    assert steps.count(state['updates'])==len(list(SharedPromptDecoder(1024).parameters()))
    if report['arm']=='joint': assert steps.count(state['updates']-2000)==20
    assert report['tail_changed']==(report['arm']=='joint') and report['head_changed']
    result=dict(status='accepted_joint_training_gate' if report['gate'] else 'accepted_joint_pilot',arm=report['arm'],
                worker_pid=os.getpid(),rows_replayed=len(rows),sources=sources(),inputs=data.identity,
                initial_state=report['initial_state'],initial_logits_sha256=report['initial_logits_sha256'],
                training_sha256=report['training_sha256'],checkpoint_sha256=report['checkpoint_sha256'],
                quality_claim=False,report_sha256=sha(directory/'report.json'))
    (directory/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--arm',choices=['frozen','joint'])
    p.add_argument('--gate',action='store_true'); p.add_argument('--accept',action='store_true'); p.add_argument('--gate-root',type=Path)
    args=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.accept: accept(args.output); return
    assert args.arm; args.output.mkdir(exist_ok=False)
    task_proof=json.loads((TASK/'independent_acceptance.json').read_text())
    assert task_proof['training_sha256']==sha(TASK/'training4000.jsonl')
    lines=(TASK/'training4000.jsonl').read_text().splitlines(keepends=True)
    selected=gate_indices(lines); train=[lines[i] for i in selected] if args.gate else lines
    data=JointData(PREFIX)
    if not args.gate:
        assert args.gate_root
        gates=[json.loads((args.gate_root/a/'independent_acceptance.json').read_text()) for a in ['frozen','joint']]
        for g in gates: assert g['status']=='accepted_joint_training_gate' and g['sources']==sources() and g['inputs']==data.identity
        assert gates[0]['initial_state']==gates[1]['initial_state'] and gates[0]['initial_logits_sha256']==gates[1]['initial_logits_sha256']
    head,tail,opt=initialize(args.arm)
    original_head=tree_sha(head.state_dict()); original_tail=tree_sha(tail.state_dict())
    initial=[]
    with torch.no_grad():
        for i in selected: initial.append(head(**data.model_inputs(json.loads(lines[i]),tail,'cuda:0')).cpu())
    initial_logits=tree_sha(initial)
    updates=2 if args.gate else 1000
    with (args.output/'training_consumed.jsonl').open('w') as stream:
        for step in range(updates):
            batch=train[step*4:step*4+4]; assert len(batch)==4
            losses=update(data,head,tail,opt,[json.loads(s) for s in batch]); stream.writelines(batch)
            if step%25==0 or step+1==updates:
                stream.flush(); status=dict(status='training',worker_pid=os.getpid(),updates=step+1,losses=losses)
                (args.output/'status.json').write_text(json.dumps(status)); print(json.dumps(status),flush=True)
    state=dict(model=head.state_dict(),tail=tail.state_dict(),optimizer=opt.state_dict(),updates=2000+updates,seed=20261031,arm=args.arm)
    torch.save(state,args.output/'checkpoint.pt')
    fresh,ft,fo=initialize(args.arm); saved=torch.load(args.output/'checkpoint.pt',map_location='cuda:0',weights_only=True)
    fresh.load_state_dict(saved['model']); ft.load_state_dict(saved['tail']); fo.load_state_dict(saved['optimizer'])
    assert tree_sha(fo.state_dict())==tree_sha(opt.state_dict())
    rows=[]; masks=[]
    with torch.no_grad():
        for scope,budget,plan in evaluation(data,[json.loads(s) for s in train],args.gate):
            logits=fresh(**data.model_inputs(plan,ft,'cuda:0'))
            if args.gate: assert torch.equal(logits,head(**data.model_inputs(plan,tail,'cuda:0')))
            assert torch.isfinite(logits).all()
            mask=(logits[0]>=0).cpu().numpy(); gt,mpp=data.labels(plan['query_id'])
            rows.append(dict(scope=scope,budget=budget,plan=plan,metrics=score(mask,gt==plan['target'],gt!=255,mpp)))
            masks.append(np.packbits(mask))
            if len(rows)%100==0: print(json.dumps(dict(status='scoring',rows=len(rows))),flush=True)
    np.savez_compressed(args.output/'packed_masks.npz',masks=np.stack(masks))
    (args.output/'rows.json').write_text(json.dumps(rows)+'\n')
    if args.gate:
        batch=[json.loads(s) for s in train[:4]]
        assert update(data,head,tail,opt,batch)==update(data,fresh,ft,fo,batch)
        assert tree_sha(head.state_dict())==tree_sha(fresh.state_dict()) and tree_sha(tail.state_dict())==tree_sha(ft.state_dict())
        assert tree_sha(opt.state_dict())==tree_sha(fo.state_dict())
    report=dict(arm=args.arm,gate=args.gate,sources=sources(),inputs=data.identity,worker_pid=os.getpid(),
                initial_state=dict(head=original_head,tail=original_tail),initial_logits_sha256=initial_logits,
                head_changed=tree_sha(saved['model'])!=original_head,tail_changed=tree_sha(saved['tail'])!=original_tail,
                parent_sha256=sha(PARENT),training_sha256=sha(TASK/'training4000.jsonl'),
                checkpoint_sha256=sha(args.output/'checkpoint.pt'),exact_reload=True,exact_continuation=args.gate,
                gate_root=str(args.gate_root) if args.gate_root else None,threshold=.5,scoring_rows=len(rows))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
