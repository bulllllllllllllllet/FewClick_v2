"""Train matched dynamic CRC/multi-domain and original/multi-quality arms."""
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

from benchmarks.v5.multidomain_data.prepare import dump,sha
from .episodes import EpisodeCorpus
from .evaluate import evaluate,validation_plans
from .losses import episode_loss
from .model import load_model
from .longrun import learning_rates,Coverage


def validate_legacy_plans(primary,legacy):
    """Only click positions may differ in this compatibility control."""
    if len(primary)!=len(legacy):raise ValueError('legacy validation task count mismatch')
    keys=('number','support_id','query_id','dataset','target','kind','protocol','split',
          'support_quality','query_quality','quality_condition','pair_id')
    for p,q in zip(primary,legacy):
        if any(p.get(k)!=q.get(k) for k in keys):raise ValueError('legacy validation task identity mismatch')
        if q.get('prompt_policy') is not None:raise ValueError('compatibility manifest must use historical legacy clicks')
        if any(len(p[k])!=len(q[k]) for k in ('positive_xy','negative_xy')):raise ValueError('legacy validation click budget mismatch')


def grad_norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['C0','C1','C2','C3'],required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--warmup',type=int,default=200)
    p.add_argument('--accumulate',type=int,default=4);p.add_argument('--eval-every',type=int,default=500);p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--validation-plans',type=Path);p.add_argument('--no-amp',action='store_true');p.add_argument('--benchmark',action='store_true')
    p.add_argument('--legacy-validation-plans',type=Path,help='Same task identities with historical clicks; report-only, never used for selection')
    p.add_argument('--validation-anchor-count',type=int,default=0)
    p.add_argument('--prompt-policy',choices=['legacy','representative'],default='representative')
    p.add_argument('--lr-schedule',choices=['constant','warmup_cosine'],default='constant');p.add_argument('--lr-ramp-steps',type=int,default=200);p.add_argument('--min-lr-ratio',type=float,default=.1)
    a=p.parse_args()
    if min(a.steps,a.accumulate,a.eval_every)<1 or a.warmup<0:raise ValueError('invalid training budget')
    learning_rates(0,a.steps,a.warmup,a.lr_ramp_steps,a.min_lr_ratio,a.lr_schedule)
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0');started=time.time()
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    try:
        corpus=EpisodeCorpus(a.catalog,a.seed,prompt_policy=a.prompt_policy);dump(a.output/'coverage.json',corpus.coverage)
        plans=json.loads(a.validation_plans.read_text()) if a.validation_plans else validation_plans(corpus)
        expected_policy='representative_1021_214456' if a.prompt_policy=='representative' else None
        if any(plan.get('prompt_policy')!=expected_policy for plan in plans):raise ValueError('training/validation click policy mismatch; explicitly regenerate plans or use --prompt-policy legacy')
        legacy_plans=json.loads(a.legacy_validation_plans.read_text()) if a.legacy_validation_plans else None
        if legacy_plans is not None:
            validate_legacy_plans(plans,legacy_plans);dump(a.output/'legacy_validation_plans.json',legacy_plans)
        dump(a.output/'validation_plans.json',plans);model,weights=load_model(device);amp=not a.no_amp
        run=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},weights=weights,worker_pid=os.getpid(),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),catalog_sha256=sha((a.catalog/'tiles.parquet').read_bytes()),
            validation_plans_sha256=sha((a.output/'validation_plans.json').read_bytes()),
            source_sha256={str(f):sha(f.read_bytes()) for f in sorted(Path(__file__).parent.glob('*.py'))},
            effective_batch=a.accumulate,query_GT_in_model=False,gt_simulated_clicks=True,puma_used=False,test_used=False)
        run.update(source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            legacy_validation_plans_sha256=None if legacy_plans is None else sha((a.output/'legacy_validation_plans.json').read_bytes()),
            selection_validation='primary fixed single-group plans only; legacy is report-only')
        dump(a.output/'run.json',run)
        baseline=evaluate(model,corpus,plans,device,a.output/'baseline',amp=amp,anchor_count=a.validation_anchor_count)
        legacy_baseline=evaluate(model,corpus,legacy_plans,device,a.output/'baseline_legacy',amp=amp) if legacy_plans is not None else None
        coverage=Coverage(corpus,datasets=['crc'] if a.arm in ['C0','C1'] else None)
        optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4);history=[];counts=Counter();max_grad=Counter();best_score=-float('inf');step_times=[]
        with (a.output/'progress.jsonl').open('w') as log,(a.output/'training_episodes.jsonl').open('w') as episode_log:
            for step in range(a.steps):
                tick=time.perf_counter();joint=step>=a.warmup;model.set_stage(joint);model.train();optimizer.zero_grad(set_to_none=True);losses=[]
                rates=learning_rates(step,a.steps,a.warmup,a.lr_ramp_steps,a.min_lr_ratio,a.lr_schedule)
                for group in optimizer.param_groups:group['lr']=rates[group['name']]
                for micro in range(a.accumulate):
                    number=step*a.accumulate+micro
                    plan=corpus.plan(number,crc_only=a.arm in ['C0','C1'],quality_aug=a.arm in ['C1','C3'])
                    inputs,supervision=corpus.materialize(plan,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        output=model(**inputs);loss,parts=episode_loss(output,supervision,inputs['prompts'],inputs['local'])
                    (loss/a.accumulate).backward();losses.append(float(loss.detach()))
                    coverage.observe(plan)
                    counts[plan['dataset']+'|'+plan['kind']]+=1;episode_log.write(json.dumps(plan)+'\n')
                norms=dict(prompt=grad_norm(model.j5.decoder.prompt_model),cell=grad_norm(model.j5.cell),dense=grad_norm(model.dense),
                    assignment=grad_norm(model.j5.phase2.assignment),layer3=grad_norm(model.j5.phase2.backbone.layer3),layer4=grad_norm(model.j5.phase2.backbone.layer4))
                if not all(np.isfinite(v) for v in norms.values()):raise FloatingPointError('non-finite parameter gradient')
                for k,v in norms.items():max_grad[k]=max(max_grad[k],v)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm) or float(norm)<=0:raise FloatingPointError('invalid global gradient')
                optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick;step_times.append(seconds)
                item=dict(step=step+1,joint=joint,loss=float(np.mean(losses)),seconds=seconds,gradient_norms=norms,learning_rates=rates)
                log.write(json.dumps(item)+'\n');log.flush();episode_log.flush()
                if (step+1)%10==0 or step==0:
                    dump(a.output/'status.json',dict(status='training',worker_pid=os.getpid(),step=step+1,total_steps=a.steps,arm=a.arm,stage='joint' if joint else 'warmup',seconds=time.time()-started))
                    print('TRAIN',a.arm,json.dumps(item),flush=True)
                if (step+1)%a.eval_every==0 or step+1==a.steps:
                    report=evaluate(model,corpus,plans,device,a.output/f'validation_step_{step+1:05d}',amp=amp,save_examples=8 if step+1==a.steps else 0,anchor_count=a.validation_anchor_count)
                    legacy_report=evaluate(model,corpus,legacy_plans,device,a.output/f'legacy_validation_step_{step+1:05d}',amp=amp) if legacy_plans is not None else None
                    history.append(dict(step=step+1,metrics=report,legacy_metrics=legacy_report));overall=report['overall']
                    # Development-only matched score; never consult heldout disease/test.
                    selection=float(overall['dice'])-.25*float(overall['absent_false_positive_fraction'])
                    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,mapper=model.mapper,run=run,metrics=report,
                        legacy_metrics=legacy_report,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    path=a.output/f'checkpoint_step_{step+1:05d}.pth';torch.save(payload,path)
                    if selection>best_score:best_score=selection;dump(a.output/'best_checkpoint.json',dict(path=str(path),step=step+1,development_score=selection))
                    dump(a.output/'summary.json',dict(status='training',baseline=baseline,legacy_baseline=legacy_baseline,history=history,episode_counts=dict(counts),max_gradient_norms=dict(max_grad),coverage=coverage.summary()))
                    model.train()
        if a.steps>a.warmup and any(max_grad[k]<=0 for k in ['prompt','cell','dense','assignment','layer3','layer4']):raise ValueError('a required trainable module received no gradient')
        model.eval();real,_=corpus.materialize(plans[0],device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):expected=model(**real,with_support=False)['query']['pixel_probability'].clone()
        saved=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(saved['model'],strict=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):actual=model(**real,with_support=False)['query']['pixel_probability']
        if not torch.equal(expected,actual):raise ValueError('training checkpoint reload changed output')
        dump(a.output/'summary.json',dict(status='complete',baseline=baseline,legacy_baseline=legacy_baseline,history=history,episode_counts=dict(counts),max_gradient_norms=dict(max_grad),
            steps=a.steps,effective_batch=a.accumulate,mean_step_seconds=float(np.mean(step_times)),median_step_seconds=float(np.median(step_times)),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_reload_exact=True,seconds=time.time()-started,test_used=False,puma_used=False,coverage=coverage.summary()))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),step=a.steps,total_steps=a.steps,seconds=time.time()-started));print('COMPLETE',a.arm,flush=True)
    except Exception as exc:dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
