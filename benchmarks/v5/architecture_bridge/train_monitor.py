"""Fixed train-only diagnostics; never a checkpoint or threshold selector."""
import json
from pathlib import Path
import numpy as np
import torch


def probability_diagnostics(probability,truth,valid,swapped):
    assert probability.shape==truth.shape==valid.shape==swapped.shape
    assert torch.isfinite(probability).all() and torch.isfinite(swapped).all() and valid.any()
    p=probability[valid];t=truth[valid].bool();other=swapped[valid]
    assert ((p>=0)&(p<=1)).all() and ((other>=0)&(other<=1)).all()
    positive=int(t.sum());pred=p>=.5
    dice=float(2*(pred&t).sum()/(pred.sum()+t.sum())) if positive else None
    return dict(valid_pixels=len(p),positive_pixels=positive,mean=float(p.mean()),std=float(p.std(unbiased=False)),
                positive_mean=float(p[t].mean()) if t.any() else None,negative_mean=float(p[~t].mean()) if (~t).any() else None,
                Dice=dice,absent_FPA=float(pred.float().mean()) if not positive else None,
                empty_prediction=not bool(pred.any()),full_prediction=bool(pred.all()),
                predicted_fraction=float(pred.float().mean()),swap_valid_mean=float((p-other).abs().mean()))


def fixed_train_plans(source):
    plans=list(map(json.loads,Path(source).read_text().splitlines()));selected=[]
    for d in ['crc','tiger']:
        for kind,n in [('local',8),('cross_present',4),('cross_absent',4)]:
            rows=[r for r in plans if r['dataset']==d and r['kind']==kind][:n];assert len(rows)==n;selected+=rows
    return selected


@torch.no_grad()
def evaluate_train_monitor(model,data,plans,device,step,output):
    assert len(plans)==32 and step in [0,250,500,1000,2000]
    assert all(data.rows[p['support_id']]['split']==data.rows[p['query_id']]['split']=='train' for p in plans)
    was_training=model.training;cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []
    model.eval();rows=[]
    try:
        for plan in plans:
            inputs=data.model_inputs(plan,'xcell',device);prob=model(**inputs)['query']['pixel_probability'][0]
            prompts=dict(inputs['prompts'])
            for k in ['xy','mask']:prompts['positive_'+k],prompts['negative_'+k]=prompts['negative_'+k],prompts['positive_'+k]
            other=model(**dict(inputs,prompts=prompts))['query']['pixel_probability'][0]
            # Labels are accessed only after both inference passes.
            gt,_=data.labels(plan['query_id']);truth=torch.from_numpy(gt==plan['target']).to(device);valid=torch.from_numpy(gt!=255).to(device)
            rows.append(dict(plan=plan,diagnostics=probability_diagnostics(prob,truth,valid,other)))
        groups={}
        for d in ['crc','tiger']:
            for kind in ['local','cross_present','cross_absent']:
                values=[r['diagnostics'] for r in rows if r['plan']['dataset']==d and r['plan']['kind']==kind]
                groups[d+'|'+kind]={key:float(np.mean([r[key] for r in values if r[key] is not None])) if any(r[key] is not None for r in values) else None for key in values[0]}
        result=dict(status='complete_fixed_train_diagnostics',step=step,rows=rows,groups=groups,selection_used=False,threshold=.5)
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
    finally:model.train(was_training)
    assert torch.equal(torch.get_rng_state(),cpu_rng)
    if cuda_rng:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    return result
