# Adapted from architecture_bridge/functional_repaired.py: only model/data factories change.
"""Fixed train-only local/remote/absent learning gate with fresh final replay."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.v5.architecture_bridge.data import BASE
from benchmarks.v5.architecture_bridge.repaired_model import RepairedData, RepairedContinuous
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score

BOUNDS=dict(present_group_Dice_min=.2,absent_group_FPA_max=.1,
            present_group_mean_std_min=.02,present_group_median_swap_mean_min=.001)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['sam_native'],required=True);p.add_argument('--seed',type=int,default=20261103);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(exist_ok=False)
    source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl'
    plans=list(map(json.loads,source.read_text().splitlines()));selected=[]
    for d in ['crc','tiger']:
        for kind,n in [('local',8),('cross_present',4),('cross_absent',4)]:
            group=[r for r in plans if r['dataset']==d and r['kind']==kind][:n];assert len(group)==n;selected+=group
    if a.arm:
        from .model import DecoderData as SpatialData,DecoderModel as SpatialModel
        data=SpatialData(a.cache,a.arm,a.seed)
        build=lambda:SpatialModel(a.arm,a.seed)
    else:
        data=RepairedData(a.cache)
        build=lambda:RepairedContinuous(a.seed)
    assert all(data.rows[r['support_id']]['split']==data.rows[r['query_id']]['split']=='train' for r in selected)
    (a.output/'contract.json').write_text(json.dumps(dict(tasks=selected,source_sha256=sha(source),cache=data.identity,bounds=BOUNDS,updates=256,batch=4,seed=a.seed,arm=a.arm),indent=2))
    started=time.monotonic()
    model=build();model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    def update(m,o,batch):
        o.zero_grad(set_to_none=True)
        for plan in batch:
            inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(m(**inputs),data,plan,inputs['prompts'])
            assert torch.isfinite(loss);(loss/4).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);o.step()
    # Exact fresh optimizer continuation before proceeding with the budget.
    for step in range(2):update(model,opt,selected[step*4:step*4+4])
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'engineering.pt')
    update(model,opt,selected[8:12]);expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    fresh=build();saved=torch.load(a.output/'engineering.pt',weights_only=True)
    fresh.load_state_dict(saved['model']);fresh.train();fresh_opt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);fresh_opt.load_state_dict(saved['optimizer'])
    update(fresh,fresh_opt,selected[8:12])
    from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
    assert tree_sha(expected)==tree_sha(fresh.state_dict()) and tree_sha(expected_opt)==tree_sha(fresh_opt.state_dict())
    (a.output/'optimizer_acceptance.json').write_text(json.dumps(dict(exact_optimizer_continuation=True,arm=a.arm,seed=a.seed)))
    del fresh,fresh_opt,expected,expected_opt,saved
    for step in range(3,256):
        batch=[selected[(step*4+j)%32] for j in range(4)];update(model,opt,batch)
        if step%16==0:
            status=dict(status='learning',arm=a.arm,updates=step+1,total_updates=256,worker_pid=os.getpid(),seconds=time.monotonic()-started)
            (a.output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'checkpoint.pt')
    model.eval();rows=[];reference=[]
    with torch.no_grad():
        for plan in selected:
            inputs=data.model_inputs(plan,'xcell','cuda:0');prob=model(**inputs)['query']['pixel_probability'];gt,mpp=data.labels(plan['query_id']);valid=torch.from_numpy(gt!=255).cuda()
            prompts=dict(inputs['prompts'])
            for k in ['xy','mask']:prompts['positive_'+k],prompts['negative_'+k]=prompts['negative_'+k],prompts['positive_'+k]
            other=model(**dict(inputs,prompts=prompts))['query']['pixel_probability']
            rows.append(dict(plan=plan,metrics=score((prob[0]>=.5).cpu().numpy(),gt==plan['target'],gt!=255,mpp),valid_spatial_std=float(prob[0][valid].std()),swap_valid_mean=float((prob-other)[0][valid].abs().mean())))
            reference.append(prob.cpu())
    # Reload a new model from the actual final checkpoint, then replay all32.
    del model;torch.cuda.empty_cache();fresh=build();fresh.load_state_dict(torch.load(a.output/'checkpoint.pt',weights_only=True)['model']);fresh.eval()
    with torch.no_grad():
        for plan,expected in zip(selected,reference):assert torch.equal(fresh(**data.model_inputs(plan,'xcell','cuda:0'))['query']['pixel_probability'].cpu(),expected)
    np.savez_compressed(a.output/'masks.npz',masks=np.stack([np.packbits((x[0]>=.5).numpy()) for x in reference]))
    groups={};passed=True
    for d in ['crc','tiger']:
        for kind in ['local','cross_present','cross_absent']:
            g=[r for r in rows if r['plan']['dataset']==d and r['plan']['kind']==kind]
            if kind=='cross_absent':
                values=dict(FPA=float(np.mean([r['metrics']['false_positive_fraction'] for r in g])));ok=values['FPA']<=BOUNDS['absent_group_FPA_max']
            else:
                values=dict(Dice=float(np.mean([r['metrics']['dice'] for r in g])),spatial_std=float(np.mean([r['valid_spatial_std'] for r in g])),median_swap_mean=float(np.median([r['swap_valid_mean'] for r in g])))
                ok=values['Dice']>=BOUNDS['present_group_Dice_min'] and values['spatial_std']>=BOUNDS['present_group_mean_std_min'] and values['median_swap_mean']>=BOUNDS['present_group_median_swap_mean_min']
            groups[d+'|'+kind]=dict(n=len(g),values=values,passed=ok);passed=passed and ok
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_repaired_functional_gate' if passed else 'failed_repaired_functional_gate',arm=a.arm,seed=a.seed,groups=groups,rows=rows,bounds=BOUNDS,exact_optimizer_continuation=True,fresh_final_predictions_exact=True,train_only=True,source_sha256=sha(__file__),seconds=time.monotonic()-started,checkpoint_sha256=sha(a.output/'checkpoint.pt')),indent=2))
    if not passed:raise RuntimeError('Repaired learning gate failed; stop without recipe/threshold retries')


if __name__=='__main__':main()
