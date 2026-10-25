"""Matched local adaptation with real native gradients and strict resume identity."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .adapted_models import build, METHODS
from .audit import digest,write
from .local_data import LocalData
from .metrics import score,VERSION
from .public_baselines import summarize
from benchmarks.v5.multidomain_training.losses import mask_loss,point_loss
from benchmarks.v5.multidomain_training.distributed_utils import rng_state,restore_rng,optimizer_digest


def forward(model,raw,prompts,plan):
    if model.closed:return model(raw,prompts,dataset=plan['dataset'],target=plan['target'])
    return model(raw,prompts)


def loss_for(model,result,supervision,prompts):
    if model.closed:
        labels=supervision['labels'];offset=result['class_offset']
        mapped=torch.where(labels==255,255,labels-offset)
        allowed=(mapped==255)|((mapped>=0)&(mapped<result['class_logits'].shape[1]))
        if not allowed.all():raise ValueError('closed-reference class map mismatch')
        return F.cross_entropy(result['class_logits'].float(),mapped,ignore_index=255)
    loss,_=mask_loss(result,supervision['truth'],supervision['valid'],supervision['mpp'])
    loss=loss+.1*point_loss(result['pixel_logits'],prompts)
    if 'assignment_regularization' in result:loss=loss+result['assignment_regularization']
    if 'quality' in result:
        # Train IoU calibration; candidate selection itself always uses predicted
        # quality, never an oracle GT argmax. Ground truth is used only in the loss.
        with torch.no_grad():
            predicted=result['candidate_logits']>=0;truth=supervision['truth'][:,None];valid=supervision['valid'][:,None]
            intersection=(predicted&truth&valid).sum((-2,-1));union=((predicted|truth)&valid).sum((-2,-1))
            target=intersection.float()/union.clamp_min(1)
        loss=loss+.1*F.mse_loss(result['quality'].float(),target)
    return loss


def evaluate(model,data,plans,device,output):
    output.mkdir(parents=True,exist_ok=False);model.eval();rows=[]
    with torch.inference_mode():
        for plan in plans:
            raw,prompts,supervision=data.materialize(plan,device,model.contexts,model.cells)
            with torch.autocast('cuda',dtype=torch.bfloat16):result=forward(model,raw,prompts,plan)
            probability=result['pixel_probability'].float()
            if not torch.isfinite(probability).all():raise FloatingPointError('nonfinite evaluation')
            prediction=result['categorical_prediction'] if model.closed else probability>=.5
            record=score(prediction[0].cpu().numpy(),supervision['truth'][0].cpu().numpy(),
                supervision['valid'][0].cpu().numpy(),supervision['mpp'])
            meta=data.rows[plan['query_id']]
            rows.append(plan|record|dict(latency_ms=0.,space_key=meta['space_key'],spatial_group_id=meta['spatial_group_id'],source_id=meta['source_id']))
    write(output/'rows.json',rows);report=summarize(rows);write(output/'summary.json',report);return report


def reference_gate(model,raw,prompts):
    """Compare differentiable bridges to the already-gated public inference path."""
    if not hasattr(model,'sam'):return dict(required=False)
    model.eval()
    with torch.inference_mode():native=model(raw,prompts)['pixel_probability'][0].float().cpu().numpy()
    rgb=raw['rgb'][0].permute(1,2,0).cpu().numpy();scale=np.array(rgb.shape[:2][::-1])
    pos=prompts['positive_xy'][0].cpu().numpy()*scale;neg=prompts['negative_xy'][0].cpu().numpy()*scale
    if hasattr(model,'net_high'):
        from .public_baselines import SharedContextWSISAM
        reference=object.__new__(SharedContextWSISAM)
        reference.sam=model.sam;reference.net_high=model.net_high;reference.net_low=model.net_low
        reference.device=raw['rgb'].device;reference.threshold=.5;reference.selection_rule='single_fused_mask'
        low=raw['rgb_1'][0].permute(1,2,0).cpu().numpy()
        expected=reference.predict(dict(fine=rgb,low=low),pos,neg,'point',None)['probability']
    else:
        from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
        from benchmarks.v4.baseline.adapters.base import EpisodeRequest
        from segment_anything import SamPredictor
        reference=object.__new__(SAMAdapter);reference.model=model.sam;reference.predictor=SamPredictor(model.sam)
        reference.device=raw['rgb'].device;reference.threshold=.5
        expected=reference.timed_predict(EpisodeRequest('native-gate',rgb,pos,neg,'point')).probability
    error=float(np.abs(native-expected).max())
    if error>2e-5 or not np.array_equal(native>=.5,expected>=.5):
        raise ValueError(f'public/native bridge mismatch: {error}')
    return dict(required=True,max_probability_error=error,binary_identical=True)


def grad_norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def fingerprint(model,data,method,steps,warmup,accumulate,seed):
    return dict(method=method,protocol_sha256=digest(data.tasks/'contract.json'),training_plans_sha256=data.realized_sha,
        steps=steps,warmup=warmup,accumulate=accumulate,seed=seed,metric_version=VERSION,initialization=model.prior,
        source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('adapted_models.py'),
            Path(__file__).with_name('local_data.py'),Path('benchmarks/v5/paper_multiscale/model.py'),
            Path('benchmarks/v5/multidomain_training/losses.py'))})


def save(path,model,optimizer,step,contract,history,device):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_sha=optimizer_digest(optimizer.state_dict()),
        step=step,contract=contract,history=history,rng=rng_state(device)),path)


def restore(path,model,optimizer,contract,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['contract']!=contract:raise ValueError('resume initialization/data/code/budget identity mismatch')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    if optimizer_digest(optimizer.state_dict())!=saved['optimizer_sha']:raise ValueError('optimizer resume mismatch')
    restore_rng(saved['rng'],device)
    return saved['step'],saved['history']


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--realized',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--accumulate',type=int,default=4);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261025);p.add_argument('--resume',type=Path);p.add_argument('--gate',action='store_true')
    a=p.parse_args()
    if a.gate:a.steps=4;a.warmup=2;a.accumulate=2;a.eval_every=2
    if min(a.steps,a.accumulate,a.eval_every)<1 or not 0<=a.warmup<a.steps:raise ValueError('invalid budget')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();device=torch.device('cuda:0');torch.set_num_threads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    try:
        data=LocalData(a.tasks,a.realized)
        if len(data.training)!=a.steps*a.accumulate:raise ValueError('training task count must exactly match update budget')
        plans=data.validation
        if a.gate:
            plans=[next(p for p in plans if p['dataset']==dataset) for dataset in sorted({p['dataset'] for p in plans})]
        model=build(a.method,device);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        contract=fingerprint(model,data,a.method,a.steps,a.warmup,a.accumulate,a.seed)
        start=0;history=[]
        if a.resume:start,history=restore(a.resume,model,optimizer,contract,device)
        if not start<a.steps:raise ValueError('resume already finished')
        write(a.output/'run.json',dict(contract=contract,worker_pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},parameters=sum(p.numel() for p in model.parameters()),
            public_bridge_gate_required=a.gate,GT_in_interactive_forward=False,closed_reference_class_ID=model.closed,
            prediction='categorical argmax' if model.closed else 'probability>=.5'))
        raw,prompts,supervision=data.materialize(data.training[start*a.accumulate],device,model.contexts,model.cells)
        if a.gate:write(a.output/'public_bridge_gate.json',reference_gate(model,raw,prompts))
        best=max((v['score'] for v in history),default=-float('inf'));gate_resumed=False;nonzero_joint=False
        with (a.output/'training_consumed.jsonl').open('x') as consumed,(a.output/'progress.jsonl').open('x') as progress:
            for step in range(start,a.steps):
                model.set_stage(step>=a.warmup);model.train();optimizer.zero_grad(set_to_none=True);losses=[]
                decay=.1+.9*.5*(1+math.cos(math.pi*step/max(a.steps-1,1)))
                for group in optimizer.param_groups:
                    group['lr']=(1e-4 if group['name']=='heads' else 1e-5)*decay
                for micro in range(a.accumulate):
                    plan=data.training[step*a.accumulate+micro]
                    raw,prompts,supervision=data.materialize(plan,device,model.contexts,model.cells)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        result=forward(model,raw,prompts,plan);loss=loss_for(model,result,supervision,prompts)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite adaptation loss')
                    (loss/a.accumulate).backward();losses.append(float(loss.detach()))
                    consumed.write(json.dumps(plan,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
                norms={g['name']:grad_norm(g['params']) for g in optimizer.param_groups}
                if not np.isfinite(list(norms.values())).all() or norms['heads']<=0:raise ValueError('invalid head gradients')
                if step>=a.warmup and norms['backbone']>0:nonzero_joint=True
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                record=dict(step=step+1,loss=float(np.mean(losses)),gradient_norms=norms,seconds=time.time()-started)
                progress.write(json.dumps(record)+'\n');progress.flush();consumed.flush()
                if (step+1)%20==0 or a.gate:print(json.dumps(record),flush=True)
                if (step+1)%a.eval_every==0 or step+1==a.steps:
                    report=evaluate(model,data,plans,device,a.output/f'validation_{step+1:05d}_1025')
                    metric=report['3+2']['macro_pair_dice'];history.append(dict(step=step+1,score=metric,metrics=report))
                    checkpoint=a.output/f'checkpoint_step_{step+1:05d}.pth';save(checkpoint,model,optimizer,step+1,contract,history,device)
                    # Reload from disk into the live model, including optimizer/RNG,
                    # and verify a real evaluation prediction. Eval does not consume dropout RNG.
                    model.eval();raw,prompts,_=data.materialize(plans[0],device,model.contexts,model.cells)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        expected=forward(model,raw,prompts,plans[0])['pixel_probability'].float().cpu()
                    restored_step,restored_history=restore(checkpoint,model,optimizer,contract,device)
                    if restored_step!=step+1 or restored_history!=history:raise ValueError('resume counters/history mismatch')
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        actual=forward(model,raw,prompts,plans[0])['pixel_probability'].float().cpu()
                    if not torch.equal(expected,actual):raise ValueError('checkpoint prediction replay mismatch')
                    if metric>=best:
                        best=metric;write(a.output/'best_checkpoint.json',dict(path=str(checkpoint),step=step+1,score=metric,checkpoint_sha256=digest(checkpoint)))
                    write(a.output/'last_checkpoint.json',dict(path=str(checkpoint),step=step+1))
                    if a.gate and step+1==a.warmup:
                        # Destroy the live model/optimizer; a new construction must
                        # restore complete state before taking the joint-training steps.
                        del model,optimizer,result,loss;gc.collect();torch.cuda.empty_cache()
                        model=build(a.method,device);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
                        restored_step,_=restore(checkpoint,model,optimizer,contract,device)
                        if restored_step!=a.warmup:raise ValueError('fresh resume offset')
                        model.eval()
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            replay=forward(model,raw,prompts,plans[0])['pixel_probability'].float().cpu()
                        if not torch.equal(expected,replay):raise ValueError('fresh-process-style model resume mismatch')
                        gate_resumed=True
                write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),step=step+1,total=a.steps,seconds=time.time()-started))
        if not nonzero_joint:raise ValueError('no verified gradient into joint backbone')
        used_sha=digest(a.output/'training_consumed.jsonl')
        if start==0 and used_sha!=data.realized_sha:raise ValueError('consumed coordinate manifest differs from shared file')
        report=dict(status='complete',worker_pid=os.getpid(),method=a.method,gate=a.gate,step=a.steps,
            history=history,training_consumed_sha256=used_sha,full_manifest_parity=start==0,
            checkpoint_reload_exact=True,fresh_gate_resume=gate_resumed,nonzero_joint_backbone_gradient=nonzero_joint,
            seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write(a.output/'summary.json',report);write(a.output/'status.json',report);print(json.dumps(report),flush=True)
    except Exception:
        write(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
