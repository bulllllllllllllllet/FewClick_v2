"""Matched four-arm training with locked tasks, full state checkpoints and loud failures."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import time

import pyvips
import numpy as np
import torch

from .data import MultiscaleCorpus, validation_plans
from .model import load_model
from benchmarks.v5.multidomain_data.prepare import dump, sha
from benchmarks.v5.multidomain_training.evaluate import evaluate
from benchmarks.v5.multidomain_training.losses import episode_loss
from benchmarks.v5.multidomain_training.longrun import learning_rates
from benchmarks.v5.multidomain_training.distributed_utils import rng_state, restore_rng, optimizer_digest

ARMS={'single_region':(False,False),'multi_region':(True,False),
      'single_dense':(False,True),'multi_dense':(True,True)}


def atomic_checkpoint(path,payload):
    temp=path.with_suffix('.partial');torch.save(payload,temp);temp.rename(path)


def norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def identity(catalog,plans,arm,weights,args):
    return dict(catalog_sha256=sha((catalog/'tiles.parquet').read_bytes()),
                validation_sha256=sha(json.dumps(plans,sort_keys=True).encode()),arm=arm,weights=weights,
                budget={k:getattr(args,k) for k in ('steps','warmup','accumulate','eval_every','seed')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--accumulate',type=int,default=4);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261025);p.add_argument('--validation-plans',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--gate',action='store_true')
    a=p.parse_args();end=a.stop_after or a.steps
    if min(a.steps,a.accumulate,a.eval_every)<1 or not 0<=a.warmup<a.steps or not 0<end<=a.steps:
        raise ValueError('invalid training budget')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);device=torch.device('cuda:0')
    try:
        corpus=MultiscaleCorpus(a.catalog,a.seed)
        plans=json.loads(a.validation_plans.read_text()) if a.validation_plans else validation_plans(corpus,2 if a.gate else 12)
        dump(a.output/'validation_plans.json',plans)
        model,weights=load_model(device,*ARMS[a.arm]);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        contract=identity(a.catalog,plans,a.arm,weights,a);start=0;history=[];best=-float('inf');counts=Counter()
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            if saved['contract']!=contract:raise ValueError('resume data/model/budget identity mismatch')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            if optimizer_digest(optimizer.state_dict())!=optimizer_digest(saved['optimizer']):raise ValueError('optimizer restore mismatch')
            restore_rng(saved['rng'],device);start=saved['step'];history=saved['history'];best=saved['best'];counts.update(saved['counts'])
            if not start<end:raise ValueError('resume end does not follow saved step')
        run=dict(contract=contract,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                 worker_pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 sources={str(f):sha(f.read_bytes()) for f in sorted(Path(__file__).parent.glob('*.py'))},
                 protocol_policy={str(k):list(v[0]) for k,v in corpus.protocol_policy.items()},
                 historical_initialization_exposure=True,test_used=False,puma_used=False,GT_in_forward=False)
        run['parameters_total']=sum(p.numel() for p in model.parameters())
        dump(a.output/'run.json',run)
        if start==0:
            baseline=evaluate(model,corpus,plans,device,a.output/'baseline_1025',amp=True)
            dump(a.output/'baseline_summary.json',baseline)
        with (a.output/'progress.jsonl').open('w') as progress,(a.output/'training_episodes.jsonl').open('w') as episodes:
            for step in range(start,end):
                tick=time.monotonic();joint=step>=a.warmup;model.set_stage(joint);model.train();optimizer.zero_grad(set_to_none=True)
                rates=learning_rates(step,a.steps,a.warmup,min(200,max(1,a.steps//10)),.1,'warmup_cosine')
                for group in optimizer.param_groups:group['lr']=rates[group['name']]
                losses=[]
                for micro in range(a.accumulate):
                    plan=corpus.plan(step*a.accumulate+micro,quality_aug=False)
                    inputs,supervision=corpus.materialize(plan,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**inputs);loss,parts=episode_loss(output,supervision,inputs['prompts'],inputs['local'])
                        loss=loss+output['assignment_regularization']
                    if not torch.isfinite(loss):raise FloatingPointError(f'loss step={step} task={plan}')
                    (loss/a.accumulate).backward();losses.append(float(loss.detach()))
                    counts[plan['dataset']+'|'+plan['kind']]+=1;episodes.write(json.dumps(plan)+'\n')
                norms=dict(cell=norm(model.cell),prompt=norm(model.decoder.prompt_model),graph=norm(model.decoder),
                           hierarchy=norm(model.hierarchy) if model.hierarchy is not None else 0.,
                           encoders=[norm(e) for e in model.encoders],dense=norm(model.dense) if model.dense is not None else 0.)
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError(f'gradient step={step} norms={norms}')
                if joint and any(v<=0 for v in norms['encoders']):raise ValueError('a scale encoder received no joint gradient')
                clipped=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(clipped) or clipped<=0:raise FloatingPointError('invalid global gradient')
                optimizer.step();torch.cuda.synchronize()
                row=dict(step=step+1,loss=float(np.mean(losses)),joint=joint,seconds=time.monotonic()-tick,
                         gradient_norms=norms,learning_rates=rates)
                progress.write(json.dumps(row)+'\n');progress.flush();episodes.flush()
                if step==start or (step+1)%10==0:
                    dump(a.output/'status.json',dict(status='training',worker_pid=os.getpid(),step=step+1,total=a.steps,arm=a.arm,seconds=time.time()-started))
                    print('TRAIN',json.dumps(row),flush=True)
                if (step+1)%a.eval_every==0 or step+1==end:
                    report=evaluate(model,corpus,plans,device,a.output/f'validation_{step+1:05d}_1025',amp=True,save_examples=4 if step+1==end else 0)
                    overall=report['overall'];present=overall['dice'];absent=overall['absent_false_positive_fraction']
                    score=float(present)-.25*float(absent or 0.)
                    history.append(dict(step=step+1,metrics=report,score=score));best=max(best,score)
                    path=a.output/f'checkpoint_step_{step+1:05d}.pth'
                    payload=dict(step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(device),
                                 history=history,best=best,counts=dict(counts),contract=contract)
                    atomic_checkpoint(path,payload)
                    # Strict reload must preserve the actual fixed-task probability.
                    model.eval();real,_=corpus.materialize(plans[0],device)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(**real,with_support=False)['query']['pixel_probability'].clone()
                    saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(**real,with_support=False)['query']['pixel_probability']
                    if not torch.equal(expected,actual):raise ValueError('checkpoint prediction replay mismatch')
                    if score>=best:dump(a.output/'best_checkpoint.json',dict(path=str(path),step=step+1,score=score))
                    dump(a.output/'last_checkpoint.json',dict(path=str(path),step=step+1))
                    dump(a.output/'summary.json',dict(status='training',history=history,counts=dict(counts),checkpoint_reload_exact=True))
                    print('VALIDATION',step+1,json.dumps(overall),flush=True)
        status='complete' if end==a.steps else 'stopped_at_requested_boundary'
        summary=dict(status=status,worker_pid=os.getpid(),step=end,history=history,counts=dict(counts),
                     checkpoint_reload_exact=True,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(a.output/'summary.json',summary);dump(a.output/'status.json',dict(status=status,worker_pid=os.getpid(),step=end,total=a.steps,seconds=time.time()-started))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
