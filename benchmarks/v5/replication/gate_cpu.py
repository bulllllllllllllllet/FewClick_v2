"""Actual reordered loader, shared model/loss/update and disk checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import pyvips
import numpy as np
import torch

from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.train_adapted import forward,loss_for
from benchmarks.v5.fair_comparison.metrics import score


def head_signature(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        if 'backbone' in key or 'semantic' in key:continue
        h.update(key.encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def gate(protocol,output):
    if torch.cuda.is_available():raise ValueError('CPU model gate must hide CUDA')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    manifest=json.loads((protocol/'manifest.json').read_text());results=[]
    for seed_text,paths in manifest.items():
        seed=int(seed_text);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        folder=output/f'seed_{seed}';folder.mkdir();data=LocalData(Path(paths['tasks']),Path(paths['gate_realized']))
        model=build('fewclick_m3d_image',torch.device('cpu'));model.set_stage(False);model.train()
        optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        initial=head_signature(model);optimizer.zero_grad(set_to_none=True);losses=[]
        for domain in ('crc','tiger'):
            plan=next(p for p in data.training if p['dataset']==domain)
            raw,prompts,supervision=data.materialize(plan,'cpu',True,True)
            result=forward(model,raw,prompts,plan);loss=loss_for(model,result,supervision,prompts)
            if not torch.isfinite(loss):raise ValueError('reordered real loss nonfinite')
            (loss/2).backward();losses.append(float(loss.detach()))
        norms=[p.grad.norm() for p in model.parameters() if p.grad is not None]
        if not norms or not all(torch.isfinite(n) for n in norms) or sum(float(n) for n in norms)<=0:
            raise ValueError('no finite real head gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        updated=head_signature(model)
        if updated==initial:raise ValueError('reordered model did not update')
        model.eval()
        with torch.inference_mode():expected=forward(model,raw,prompts,plan)['pixel_probability'].clone()
        checkpoint=folder/'checkpoint_one_update.pth'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed),checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        with torch.inference_mode():actual=forward(model,raw,prompts,plan)['pixel_probability']
        if not torch.equal(expected,actual):raise ValueError('reordered checkpoint replay mismatch')
        probability=actual[0].numpy();metric=score(probability>=.5,supervision['truth'][0].numpy(),supervision['valid'][0].numpy(),supervision['mpp'])
        results.append(dict(seed=seed,losses=losses,initial_head_sha256=initial,updated_head_sha256=updated,
            checkpoint_sha256=digest(checkpoint),checkpoint_replay_exact=True,actual_update=True,
            tasks_sha256=digest(Path(paths['tasks'])/'contract.json'),gate_coordinate_sha256=data.realized_sha,metrics=metric))
        del model,optimizer,saved,result,loss,expected,actual
    if len({r['initial_head_sha256'] for r in results})!=2:raise ValueError('new-head seeds failed to differ')
    write(output/'summary.json',dict(status='passed',protocol_manifest_sha256=digest(protocol/'manifest.json'),
        results=results,source_sha256=digest(__file__),
        scope='CPU actual reordered two-domain data/model/loss/one update/checkpoint; each method/seed still requires native GPU gate'))
    print(json.dumps(dict(status='passed',seeds=[r['seed'] for r in results])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gate(a.protocol,a.output)
