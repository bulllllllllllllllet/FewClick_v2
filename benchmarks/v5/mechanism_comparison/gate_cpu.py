"""Actual CPU training/resume gates for new cell and capacity interventions."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import pyvips
import torch

from .models import VARIANTS,build,filter_cells
from benchmarks.v5.fair_comparison.audit import write,digest
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.train_adapted import loss_for,grad_norm
from benchmarks.v5.multidomain_training.distributed_utils import optimizer_digest


def hashes(model):
    return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name,value in model.named_parameters()}


def run(tasks,realized,variant,output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261025)
    device=torch.device('cpu');started=time.time();data=LocalData(tasks,realized);plans=data.training[:4]
    if len(plans)!=4 or {p['dataset'] for p in plans}!={'crc','tiger'}:raise ValueError('gate requires four real two-domain training tasks')
    model=build(variant,device);groups=model.parameter_groups()
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)):raise ValueError('optimizer contains duplicate shared weights')
    shared=VARIANTS[variant].get('shared',False)
    if shared and not all(e is model.core.encoders[0] for e in model.core.encoders):raise ValueError('encoder sharing contract failed')
    total=sum(p.numel() for p in model.parameters());optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    initial=hashes(model);history=[];resumed=False;before_joint=None;consumed=[]
    write(output/'run.json',dict(variant=variant,worker_pid=os.getpid(),device='cpu',parameters=total,
        optimizer_unique_parameters=len(ids),initialization=model.prior,training_plans_sha256=data.realized_sha,
        steps=4,warmup=2,accumulate=1,scope='real CPU intervention gate; formal GPU training still requires native trainer gate'))
    for step,plan in enumerate(plans):
        model.set_stage(step>=2);model.train();optimizer.zero_grad(set_to_none=True)
        raw,prompts,supervision=data.materialize(plan,device,model.contexts,model.cells)
        filtered=filter_cells(raw,model.spec['cell'])
        if model.spec['cell']=='morphology':
            for key,value in raw.items():
                if key=='cells' or key.startswith('cells_'):
                    if not torch.equal(filtered[key][...,:10],value[...,:10]) or filtered[key][...,10:].count_nonzero():raise ValueError('morphology filter changed geometry or retained XCell')
        result=model(raw,prompts);loss=loss_for(model,result,supervision,prompts)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite mechanism loss')
        loss.backward();norms={g['name']:grad_norm(g['params']) for g in optimizer.param_groups}
        if norms['heads']<=0 or (step>=2 and norms['backbone']<=0):raise ValueError('missing mechanism gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();consumed.append(plan)
        history.append(dict(step=step+1,loss=float(loss.detach()),gradient_norms=norms))
        print(json.dumps(history[-1]),flush=True)
        if step==1:
            model.eval();probe_raw,probe_prompts,_=data.materialize(plans[0],device,model.contexts,model.cells)
            with torch.inference_mode():expected=model(probe_raw,probe_prompts)['pixel_probability'].clone()
            before_joint=hashes(model)
            checkpoint=output/'checkpoint_step_00002.pth'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_sha=optimizer_digest(optimizer.state_dict()),
                rng=torch.get_rng_state(),step=2,initialization=model.prior),checkpoint)
            del model,optimizer,groups,result,loss;gc.collect()
            model=build(variant,device);groups=model.parameter_groups();optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['initialization']!=model.prior:raise ValueError('mechanism source/init changed while resuming')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            if optimizer_digest(optimizer.state_dict())!=saved['optimizer_sha']:raise ValueError('optimizer resume mismatch')
            torch.set_rng_state(saved['rng']);del saved;model.eval()
            with torch.inference_mode():actual=model(probe_raw,probe_prompts)['pixel_probability']
            if not torch.equal(expected,actual):raise ValueError('fresh CPU model replay mismatch')
            resumed=True
    final=hashes(model);changed=[k for k in final if final[k]!=initial[k]]
    backbone=[k for k in final if 'backbone.' in k and final[k]!=before_joint[k]]
    heads=[k for k in changed if 'backbone.' not in k]
    if not backbone or not heads:raise ValueError('no actual parameter update')
    ids=[id(p) for g in model.parameter_groups() for p in g['params']]
    if len(ids)!=len(set(ids)):raise ValueError('shared duplicate optimizer parameters after resume')
    model.eval();raw,prompts,_=data.materialize(plans[0],device,model.contexts,model.cells)
    invariance=None
    if model.spec['cell'] in ('none','morphology'):
        changed_inputs=dict(raw)
        for key,value in raw.items():
            if key=='cells' or key.startswith('cells_'):
                modified=value.clone()
                if model.spec['cell']=='none':modified.normal_()
                else:modified[...,10:]=torch.randn_like(modified[...,10:])
                changed_inputs[key]=modified
        with torch.inference_mode():
            expected=model(raw,prompts)['pixel_probability'];actual=model(changed_inputs,prompts)['pixel_probability']
        if not torch.equal(expected,actual):raise ValueError('excluded cell appearance affected prediction')
        invariance=True
    torch.save(dict(model=model.state_dict(),step=4,initialization=model.prior),output/'checkpoint_step_00004.pth')
    source=output/'consumed.jsonl'
    with source.open('x') as handle:
        for plan in consumed:handle.write(json.dumps(plan,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(status='passed',worker_pid=os.getpid(),variant=variant,parameters=total,history=history,
        fresh_model_optimizer_resume=resumed,actual_changed_tensors=len(changed),backbone_changed_tensors=len(backbone),
        head_changed_tensors=len(heads),excluded_feature_perturbation_invariant=invariance,
        shared_encoder=shared,optimizer_parameters_unique=True,consumed_sha256=digest(source),seconds=time.time()-started,
        scope='four-update CPU intervention gate; no convergence or formal GPU performance claim')
    write(output/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--realized',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.tasks,a.realized,a.variant,a.output)
