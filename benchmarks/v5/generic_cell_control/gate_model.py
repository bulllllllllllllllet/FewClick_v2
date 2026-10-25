"""Actual genericHE loaders, gradients, fresh checkpoint restore and continuation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import traceback

import pyvips
import numpy as np
import torch

from .models import ARMS, build
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.train_adapted import forward,loss_for
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.multidomain_training.distributed_utils import optimizer_digest


def signature(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if 'backbone' not in name and 'semantic' not in name:
            h.update(name.encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def run(assembly,output):
    require(not torch.cuda.is_available(),'CPU gate must hide all CUDA devices')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cpu')
    completion=json.loads((assembly/'summary.json').read_text());manifest=json.loads((assembly/'manifest.json').read_text())
    require(completion['status']=='complete_gate_assembly' and completion['gate'] and
            digest(assembly/'manifest.json')==completion['manifest_sha256'],'actual gate assembly required')
    records=[]
    for arm in ARMS:
        torch.manual_seed(20261025);np.random.seed(20261025);random.seed(20261025)
        paths=manifest[arm];data=LocalData(Path(paths['tasks']),Path(paths['realized']))
        require(len(data.training)==8 and len(data.validation)==2,'gate task coverage')
        model=build(arm,device);model.set_stage(False);model.train()
        optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        initial=signature(model);registered=sum(p.numel() for p in model.parameters())
        before_backbones=[e.backbone.layer3[0].conv1.weight.detach().clone() for e in model.core.encoders]
        losses=[];gradient_checks=[]
        for step in range(4):
            joint=step>=2;model.set_stage(joint);model.train();optimizer.zero_grad(set_to_none=True)
            domains=[]
            for plan in data.training[step*2:step*2+2]:
                raw,prompts,supervision=data.materialize(plan,device,True,True)
                for key,value in raw.items():
                    if key=='cells' or key.startswith('cells_'):require(value.shape[-1]==74 and torch.isfinite(value).all(),'new cell tensor')
                result=forward(model,raw,prompts,plan);loss=loss_for(model,result,supervision,prompts)
                require(torch.isfinite(loss),'nonfinite genericHE loss');(loss/2).backward();losses.append(float(loss.detach()));domains.append(plan['dataset'])
            require(set(domains)=={'crc','tiger'},'both domains in each update')
            norms={g['name']:sum(float(p.grad.detach().float().norm()) for p in g['params'] if p.grad is not None) for g in optimizer.param_groups}
            require(norms['heads']>0 and np.isfinite(list(norms.values())).all(),'finite nonzero head gradient')
            require(norms['backbone']>0 if joint else norms['backbone']==0,'correct staged backbone gradient')
            gradient_checks.append(dict(step=step+1,joint=joint,norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        updated=signature(model);require(updated!=initial,'head update missing')
        require(all(not torch.equal(before,e.backbone.layer3[0].conv1.weight) for before,e in zip(before_backbones,model.core.encoders)),'each view backbone must update')
        model.eval();predictions={}
        with torch.inference_mode():
            for plan in data.validation:
                raw,prompts,supervision=data.materialize(plan,device,True,True)
                predictions[plan['query_id']]=forward(model,raw,prompts,plan)['pixel_probability'].clone()
        folder=output/arm;folder.mkdir();checkpoint=folder/'checkpoint_four_updates.pth'
        rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        optimizer_sha=optimizer_digest(optimizer.state_dict())
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),RNG=rng,arm=arm,
            tasks_sha256=digest(Path(paths['tasks'])/'contract.json'),pca_sha256=paths['pca_sha256']),checkpoint)
        prior=dict(model.prior);del model,optimizer,result,loss,before_backbones;gc.collect()
        model=build(arm,device);model.set_stage(True)
        optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(saved['arm']==arm and saved['pca_sha256']==paths['pca_sha256'],'checkpoint contract')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        require(optimizer_digest(optimizer.state_dict())==optimizer_sha,'optimizer disk replay')
        torch.set_rng_state(saved['RNG']['torch']);np.random.set_state(saved['RNG']['numpy']);random.setstate(saved['RNG']['python'])
        model.eval();errors=[];metrics=[]
        with torch.inference_mode():
            for plan in data.validation:
                raw,prompts,supervision=data.materialize(plan,device,True,True)
                probability=forward(model,raw,prompts,plan)['pixel_probability']
                require(torch.equal(probability,predictions[plan['query_id']]),'fresh model prediction replay')
                path=folder/f'prediction_{plan["query_id"]}.npy';np.save(path,probability[0].numpy())
                pred=np.load(path,allow_pickle=False)>=.5
                measured=score(pred,supervision['truth'][0].numpy(),supervision['valid'][0].numpy(),supervision['mpp'])
                errors.append(0.);metrics.append(dict(dataset=plan['dataset'],counts=measured))
        model.train();model.set_stage(True);optimizer.zero_grad(set_to_none=True);before_resume=signature(model)
        for plan in data.training[:2]:
            raw,prompts,supervision=data.materialize(plan,device,True,True)
            result=forward(model,raw,prompts,plan);loss=loss_for(model,result,supervision,prompts)
            require(torch.isfinite(loss),'resume loss');(loss/2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();require(signature(model)!=before_resume,'fresh resumed update')
        record=dict(arm=arm,registered_parameters=registered,initial_head_sha256=initial,updated_head_sha256=updated,
            losses=losses,gradient_checks=gradient_checks,all_view_backbones_updated=True,fresh_checkpoint_replay_errors=errors,
            optimizer_replay_exact=True,RNG_restored=True,resumed_update=True,checkpoint_sha256=digest(checkpoint),
            data_protocol_sha256=digest(Path(paths['tasks'])/'contract.json'),coordinate_sha256=data.realized_sha,
            prior=prior,gate_metrics=metrics)
        records.append(record);write(folder/'summary.json',dict(status='passed',gate=True,**record))
        write(output/'status.json',dict(status='running',worker_pid=os.getpid(),arms_passed=len(records)))
        print(json.dumps(dict(status='arm_passed',arm=arm,losses=losses)),flush=True)
        del model,optimizer,saved,result,loss,predictions,raw,prompts,supervision,probability;gc.collect()
    require(len({r['registered_parameters'] for r in records})==1 and len({r['initial_head_sha256'] for r in records})==1,'same architecture/capacity/initialization')
    require(len({r['coordinate_sha256'] for r in records})==1,'same actual gate coordinates')
    write(output/'summary.json',dict(status='passed_cpu_model_gate',gate=True,worker_pid=os.getpid(),records=records,
        assembly_manifest_sha256=digest(assembly/'manifest.json'),source_sha256=digest(__file__),
        scope='Actual genericHE cache/loader/model/loss/staged-update/fresh optimizer+RNG checkpoint gate; formal PCA/GPU/training/accuracy incomplete'))
    write(output/'status.json',dict(status='passed_cpu_model_gate',worker_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assembly',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('output already exists')
    try:run(a.assembly,a.output)
    except Exception:
        if a.output.is_dir():
            failure=dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc())
            write(a.output/'failure.json',failure);write(a.output/'status.json',failure)
        raise
