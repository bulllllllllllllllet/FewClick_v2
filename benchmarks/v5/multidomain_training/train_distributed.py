"""Multi-GPU continuation with fixed global batch, explicit NCCL gradient sum."""
import argparse
from collections import Counter
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import time

import pyvips
import numpy as np
import torch
import torch.distributed as dist

from benchmarks.v5.multidomain_data.prepare import dump,sha
from .distributed_utils import task_numbers,synchronize_gradients,rng_state,restore_rng,state_digest,optimizer_digest,merge_rows
from .episodes import EpisodeCorpus
from .evaluate import evaluate
from .longrun import learning_rates,Coverage
from .losses import episode_loss
from .model import load_model
from .train import grad_norm,validate_legacy_plans


def collect(value,rank,world):
    result=[None]*world if rank==0 else None
    dist.gather_object(value,result,dst=0)
    return result


def distributed_evaluate(model,corpus,plans,device,output,rank,world,amp,anchor_count=0):
    subset=plans[rank::world]
    evaluate(model,corpus,subset,device,output/f'rank_{rank}',amp=amp)
    dist.barrier();report=None
    if rank==0:
        shards=[json.loads((output/f'rank_{i}'/'rows.json').read_text()) for i in range(world)]
        rows,report=merge_rows(shards,plans,anchor_count)
        dump(output/'rows.json',rows);dump(output/'summary.json',report)
    dist.barrier()
    return report


def resume_state(saved,corpus,batch):
    if 'training_state' in saved:
        state=saved['training_state'];coverage=Coverage(corpus,datasets=['crc'] if saved['run']['args']['arm'] in ('C0','C1') else None)
        coverage.seen=state['coverage_seen'];return state,coverage
    parent=Path(saved['run']['args']['output']);summary=json.loads((parent/'summary.json').read_text())
    history=[h for h in summary['history'] if h['step']<=saved['step']]
    if not history or history[-1]['step']!=saved['step']:raise ValueError('checkpoint/history boundary mismatch')
    coverage=Coverage(corpus,datasets=['crc'] if saved['run']['args']['arm'] in ('C0','C1') else None);counts=Counter();seen=0
    with (parent/'training_episodes.jsonl').open() as f:
        for line in f:
            if seen==saved['step']*batch:break
            plan=json.loads(line);coverage.observe(plan);counts[plan['dataset']+'|'+plan['kind']]+=1;seen+=1
    if seen!=saved['step']*batch:raise ValueError('checkpoint task prefix incomplete')
    maximum=Counter()
    with (parent/'progress.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['step']>saved['step']:break
            for k,v in row['gradient_norms'].items():maximum[k]=max(maximum[k],v)
    best=max(history,key=lambda h:h['metrics']['overall']['dice']-.25*h['metrics']['overall']['absent_false_positive_fraction'])
    state=dict(baseline=summary['baseline'],legacy_baseline=summary.get('legacy_baseline'),history=history,
        counts=dict(counts),max_gradient_norms=dict(maximum),coverage_seen=coverage.seen,
        best_score=best['metrics']['overall']['dice']-.25*best['metrics']['overall']['absent_false_positive_fraction'],
        best_path=str(parent/f'checkpoint_step_{best["step"]:05d}.pth'))
    return state,coverage


def serial_gradient_check(model,corpus,plans,seeds,device,amp,optimizer):
    """Replay each rank's exact dropout state in a serial four-task reference."""
    after=rng_state(device);reference=copy.deepcopy(model);reference.zero_grad(set_to_none=True);losses=[]
    for tasks,seed_list in zip(plans,seeds):
        for plan,seed in zip(tasks,seed_list):
            inputs,sup=corpus.materialize(plan,device);restore_rng(seed,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                output=reference(**inputs);loss,_=episode_loss(output,sup,inputs['prompts'],inputs['local'])
            (loss/sum(len(p) for p in plans)).backward();losses.append(float(loss.detach()))
    difference=norm=0.;maximum=reference_max=0.
    for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
        if name!=other:raise ValueError('parameter order changed')
        if (p.grad is None)!=(q.grad is None):raise ValueError('global unused-gradient semantics mismatch: '+name)
        if p.grad is None:continue
        delta=(p.grad-q.grad).double();difference+=float(delta.square().sum());norm+=float(q.grad.double().square().sum());maximum=max(maximum,float(delta.abs().max()))
        reference_max=max(reference_max,float(q.grad.abs().max()))
    restore_rng(after,device);del reference
    relative=(difference/max(norm,1e-30))**.5
    # encode_image always stores deep/shallow/semantic features in FP16, even
    # with autocast disabled. Independent CUDA backward kernels can therefore
    # differ at FP16 rounding scale. This is not a pure FP32 comparison.
    tolerance=torch.finfo(torch.bfloat16).eps if amp else 2e-4
    maximum_relative=maximum/max(reference_max,1e-30)
    return dict(relative_gradient_L2=relative,max_gradient_abs=maximum,reference_gradient_max=reference_max,
        max_gradient_relative=maximum_relative,tolerance=tolerance,amp=amp,internal_features_dtype='float16',
        passed=relative<=tolerance and maximum_relative<=tolerance,serial_mean_loss=float(np.mean(losses)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-plans',type=Path,required=True);p.add_argument('--legacy-validation-plans',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--arm',choices=['C0','C1','C2','C3'],default='C3')
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--accumulate',type=int,default=4,help='Global task batch, not batch per rank')
    p.add_argument('--eval-every',type=int,default=2000);p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--validation-anchor-count',type=int,default=160);p.add_argument('--prompt-policy',default='representative',choices=['legacy','representative'])
    p.add_argument('--lr-schedule',default='warmup_cosine');p.add_argument('--lr-ramp-steps',type=int,default=200);p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--no-amp',action='store_true');p.add_argument('--stop-after',type=int,help='Absolute step for a controlled gate, keeps the full LR budget')
    p.add_argument('--verify-serial',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world<2 or a.accumulate%world:raise ValueError('requires multiple ranks evenly dividing global batch')
    end=a.stop_after or a.steps
    if not 0<end<=a.steps:raise ValueError('invalid stop boundary')
    task_numbers(0,rank,world,a.accumulate);learning_rates(0,a.steps,a.warmup,a.lr_ramp_steps,a.min_lr_ratio,a.lr_schedule)
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30));started=time.time()
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    dist.barrier();torch.manual_seed(a.seed+rank);np.random.seed(a.seed+rank);random.seed(a.seed+rank)
    try:
        corpus=EpisodeCorpus(a.catalog,a.seed,prompt_policy=a.prompt_policy)
        plans=json.loads(a.validation_plans.read_text());legacy=json.loads(a.legacy_validation_plans.read_text()) if a.legacy_validation_plans else None
        expected='representative_1021_214456' if a.prompt_policy=='representative' else None
        if any(q.get('prompt_policy')!=expected for q in plans):raise ValueError('primary click policy mismatch')
        if legacy is not None:validate_legacy_plans(plans,legacy)
        model,weights=load_model(device);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4);saved=None;start=0
        plan_hash=sha(a.validation_plans.read_bytes());legacy_hash=sha(a.legacy_validation_plans.read_bytes()) if legacy is not None else None
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',weights_only=False);old=saved['run'];old_args=old['args']
            for key in ('arm','steps','warmup','accumulate','eval_every','seed','prompt_policy','lr_schedule','lr_ramp_steps','min_lr_ratio','validation_anchor_count','no_amp'):
                if old_args.get(key)!=getattr(a,key):raise ValueError('resume configuration mismatch: '+key)
            if old['validation_plans_sha256']!=plan_hash or old.get('legacy_validation_plans_sha256')!=legacy_hash:raise ValueError('resume validation changed')
            if old['catalog_sha256']!=sha((a.catalog/'tiles.parquet').read_bytes()) or old['weights']!=weights:raise ValueError('resume data/initialization changed')
            if old.get('training_policy_sha256')!=corpus.training_policy_sha256:raise ValueError('resume training quality policy changed')
            if saved['mapper']!='pixel':raise ValueError('unsupported checkpoint mapper')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);start=int(saved['step'])
            if optimizer_digest(optimizer.state_dict())!=optimizer_digest(saved['optimizer']):raise ValueError('optimizer restore changed state')
            if not start<end:raise ValueError('resume boundary does not precede requested end')
            if 'rank_rng' in saved:
                if len(saved['rank_rng'])!=world:raise ValueError('rank RNG world size mismatch')
                restore_rng(saved['rank_rng'][rank],device)
            elif rank==0:
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'][0],device)
            # On world1->world4 migration ranks1..3 keep independent seeded
            # streams. Exact bitwise continuation of the single CUDA stream
            # is not claimed; subsequent world4 checkpoints retain all RNGs.
        amp=not a.no_amp;workers=collect(dict(rank=rank,worker_pid=os.getpid(),device=local),rank,world)
        digests=collect(state_digest(model.state_dict()),rank,world)
        if rank==0:
            if len(set(digests))!=1:raise ValueError('rank initial states differ')
            run=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},weights=weights,worker_pid=os.getpid(),workers=workers,
                world_size=world,effective_batch=a.accumulate,source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                source_sha256={str(f):sha(f.read_bytes()) for f in sorted(Path(__file__).parent.glob('*.py'))},
                catalog_sha256=sha((a.catalog/'tiles.parquet').read_bytes()),validation_plans_sha256=plan_hash,legacy_validation_plans_sha256=legacy_hash,
                selection_validation='primary fixed single-group only; legacy report-only',query_GT_in_model=False,gt_simulated_clicks=True,puma_used=False,test_used=False,
                gradient_sync='loss/global_batch before backward; NCCL sum; global-unused gradients remain None',
                resume_step=start,parent_checkpoint=None if a.resume is None else str(a.resume),rng_migration='independent rank streams when migrating world1; exact per-rank RNG on subsequent world4 resume')
            run.update(training_policy_sha256=corpus.training_policy_sha256,training_policy=corpus.training_policy)
            dump(a.output/'run.json',run);dump(a.output/'validation_plans.json',plans)
            if legacy is not None:dump(a.output/'legacy_validation_plans.json',legacy)
            if saved is not None:
                state,coverage=resume_state(saved,corpus,a.accumulate)
                dump(a.output/'resume_verified.json',dict(step=start,model_digest=digests[0],optimizer_digest=optimizer_digest(optimizer.state_dict()),optimizer_state_entries=len(optimizer.state),workers=workers,optimizer_restored=True,parent=str(a.resume)))
            else:
                state=dict(history=[],counts={},max_gradient_norms={},best_score=-1e9,best_path=None)
                coverage=Coverage(corpus,datasets=['crc'] if a.arm in ('C0','C1') else None)
        dist.barrier()
        if saved is None:
            baseline=distributed_evaluate(model,corpus,plans,device,a.output/'baseline',rank,world,amp,a.validation_anchor_count)
            legacy_baseline=distributed_evaluate(model,corpus,legacy,device,a.output/'baseline_legacy',rank,world,amp) if legacy is not None else None
            if rank==0:state.update(baseline=baseline,legacy_baseline=legacy_baseline)
        if rank==0:
            counts=Counter(state['counts']);maximum=Counter(state['max_gradient_norms']);history=state['history'];best_score=state['best_score'];best_path=state['best_path']
        pending=[];step_times=[];last_path=None
        task_log=(a.output/f'training_episodes_rank{rank}.jsonl').open('w');progress=(a.output/'progress.jsonl').open('w') if rank==0 else None
        global_log=(a.output/'training_episodes.jsonl').open('w') if rank==0 else None
        for step in range(start,end):
            tick=time.perf_counter();model.set_stage(step>=a.warmup);model.train();optimizer.zero_grad(set_to_none=True)
            rates=learning_rates(step,a.steps,a.warmup,a.lr_ramp_steps,a.min_lr_ratio,a.lr_schedule)
            for group in optimizer.param_groups:group['lr']=rates[group['name']]
            losses=[];local_plans=[];seeds=[]
            numbers=task_numbers(step,rank,world,a.accumulate)
            for number in numbers:
                plan=corpus.plan(number,crc_only=a.arm in ('C0','C1'),quality_aug=a.arm in ('C1','C3'))
                inputs,sup=corpus.materialize(plan,device)
                if a.verify_serial:seeds.append(rng_state(device))
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):output=model(**inputs);loss,_=episode_loss(output,sup,inputs['prompts'],inputs['local'])
                # Scale before backward just like the serial trainer: internal
                # FP16 casts make deferred division numerically different.
                (loss/a.accumulate).backward();losses.append(float(loss.detach()));local_plans.append(plan);task_log.write(json.dumps(plan)+'\n')
            task_log.flush();pending.extend(local_plans);synchronize_gradients(optimizer,average=False)
            mean=torch.tensor(np.mean(losses),device=device);dist.all_reduce(mean);mean/=world
            if a.verify_serial:
                checks=collect(dict(plans=local_plans,seeds=seeds),rank,world)
                if rank==0:
                    check=serial_gradient_check(model,corpus,[q['plans'] for q in checks],[q['seeds'] for q in checks],device,amp,optimizer)
                    check['distributed_mean_loss']=float(mean)
                    check['loss_absolute_difference']=abs(check['serial_mean_loss']-float(mean))
                    dump(a.output/f'serial_check_step_{step+1:05d}.json',check)
                    if not check['passed']:raise ValueError('serial gradient mismatch: '+str(check))
                    if check['loss_absolute_difference']>2e-5:raise ValueError('serial loss mismatch: '+str(check))
                dist.barrier()
            norms=dict(prompt=grad_norm(model.j5.decoder.prompt_model),cell=grad_norm(model.j5.cell),dense=grad_norm(model.dense),
                assignment=grad_norm(model.j5.phase2.assignment),layer3=grad_norm(model.j5.phase2.backbone.layer3),layer4=grad_norm(model.j5.phase2.backbone.layer4))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm) or float(norm)<=0 or not all(np.isfinite(v) for v in norms.values()):raise FloatingPointError('invalid synchronized gradient')
            optimizer.step();torch.cuda.synchronize(device);seconds=time.perf_counter()-tick
            elapsed=torch.tensor(seconds,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX);step_times.append(float(elapsed))
            if rank==0:
                for k,v in norms.items():maximum[k]=max(maximum[k],v)
                row=dict(step=step+1,joint=step>=a.warmup,loss=float(mean),seconds=float(elapsed),gradient_norms=norms,learning_rates=rates,world_size=world,effective_batch=a.accumulate)
                progress.write(json.dumps(row)+'\n');progress.flush()
                if (step+1)%10==0 or step==start:
                    dump(a.output/'status.json',dict(status='training',worker_pid=os.getpid(),workers=workers,step=step+1,total_steps=a.steps,stage='joint' if step>=a.warmup else 'warmup',seconds=time.time()-started))
                    print('TRAIN',json.dumps(row),flush=True)
            if (step+1)%a.eval_every==0 or step+1==end:
                gathered=collect(pending,rank,world);pending=[]
                if rank==0:
                    for plan in sorted([p for q in gathered for p in q],key=lambda p:p['number']):coverage.observe(plan);counts[plan['dataset']+'|'+plan['kind']]+=1;global_log.write(json.dumps(plan)+'\n')
                    global_log.flush()
                report=distributed_evaluate(model,corpus,plans,device,a.output/f'validation_step_{step+1:05d}',rank,world,amp,a.validation_anchor_count)
                old_report=distributed_evaluate(model,corpus,legacy,device,a.output/f'legacy_validation_step_{step+1:05d}',rank,world,amp) if legacy is not None else None
                states=collect(state_digest(model.state_dict()),rank,world);rngs=collect(rng_state(device),rank,world)
                if rank==0:
                    if len(set(states))!=1:raise ValueError('models diverged across ranks')
                    history.append(dict(step=step+1,metrics=report,legacy_metrics=old_report));overall=report['overall'];score=overall['dice']-.25*overall['absent_false_positive_fraction']
                    last_path=a.output/f'checkpoint_step_{step+1:05d}.pth'
                    if score>best_score:best_score=score;best_path=str(last_path)
                    training_state=dict(baseline=state['baseline'],legacy_baseline=state['legacy_baseline'],history=history,counts=dict(counts),max_gradient_norms=dict(maximum),coverage_seen=coverage.seen,best_score=best_score,best_path=best_path)
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,mapper=model.mapper,run=run,metrics=report,legacy_metrics=old_report,
                        rank_rng=rngs,torch_rng=rngs[0]['torch'],cuda_rng=[s['cuda'] for s in rngs],training_state=training_state),last_path)
                    dump(a.output/'best_checkpoint.json',dict(path=best_path,step=int(Path(best_path).stem.rsplit('_',1)[1]),development_score=best_score,selection='primary only'))
                    dump(a.output/'summary.json',dict(status='training',steps=step+1,baseline=state['baseline'],legacy_baseline=state['legacy_baseline'],history=history,
                        episode_counts=dict(counts),max_gradient_norms=dict(maximum),coverage=coverage.summary(),rank_models_exact=True,world_size=world,resume_step=start))
                dist.barrier()
        task_log.close()
        if rank==0:progress.close();global_log.close()
        if rank==0:
            model.eval();inputs,_=corpus.materialize(plans[0],device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):expected=model(**inputs,with_support=False)['query']['pixel_probability'].clone()
            payload=torch.load(last_path,map_location=device,weights_only=False);model.load_state_dict(payload['model'],strict=True)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):actual=model(**inputs,with_support=False)['query']['pixel_probability']
            if not torch.equal(expected,actual):raise ValueError('checkpoint reload changed prediction')
            summary=json.loads((a.output/'summary.json').read_text());summary.update(status='complete' if end==a.steps else 'controlled_stop',steps=end,
                checkpoint_reload_exact=True,seconds=time.time()-started,mean_step_seconds=float(np.mean(step_times)),median_step_seconds=float(np.median(step_times)),puma_used=False,test_used=False)
            dump(a.output/'summary.json',summary);dump(a.output/'status.json',dict(status=summary['status'],step=end,total_steps=a.steps,worker_pid=os.getpid(),workers=workers,seconds=time.time()-started));print('COMPLETE',end,flush=True)
        dist.barrier();dist.destroy_process_group()
    except Exception as exc:
        dump(a.output/f'failure_rank{rank}.json',dict(worker_pid=os.getpid(),error=repr(exc)))
        if rank==0:dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)))
        raise


if __name__=='__main__':main()
