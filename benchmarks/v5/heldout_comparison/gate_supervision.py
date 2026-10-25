"""Real model/loss checks for ignored supervision; not a heldout accuracy run."""
import argparse
import gc
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import torch

from .audit import FOLDS
from .supervision import FoldData
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.train_adapted import forward,loss_for


def gate(tasks,audit,output):
    require(not torch.cuda.is_available(),'CPU supervision gate must hide CUDA')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261026)
    completed=json.loads((audit/'summary.json').read_text());require(completed['status']=='complete_feasibility_audit','actual coverage audit required')
    model=build('fewclick_m3d_image',torch.device('cpu'));model.set_stage(False);model.eval();results=[]
    for fold in FOLDS:
        data=FoldData(tasks,fold,gate=True);probes=json.loads((audit/f'{fold}_probes.json').read_text())
        for probe in probes:
            plan=probe['plan'];raw,prompts,supervision=data.materialize(plan,'cpu',True,True)
            labels=supervision['labels'];excluded=FOLDS[fold]['exclude'][plan['dataset']]
            require(not any(bool((labels==code).any()) for code in excluded),'heldout class remained supervised')
            result=forward(model,raw,prompts,plan);result['pixel_logits'].retain_grad()
            ordinary=loss_for(model,result,supervision,prompts)
            changed=dict(supervision);changed['truth']=supervision['truth'].clone()
            changed['truth'][~supervision['valid']]=~changed['truth'][~supervision['valid']]
            perturbed=loss_for(model,result,changed,prompts)
            require(torch.isfinite(ordinary) and torch.equal(ordinary,perturbed),'ignored-truth affects actual objective')
            model.zero_grad(set_to_none=True);ordinary.backward()
            gradient=result['pixel_logits'].grad
            require(gradient is not None and torch.isfinite(gradient).all(),'missing finite output gradient')
            ignored=~supervision['valid'];ignored_mass=float(gradient[ignored].abs().sum())
            require(ignored_mass==0,'ignored pixels receive supervised output gradient')
            results.append(dict(fold=fold,dataset=plan['dataset'],split=plan['split'],tile_id=plan['query_id'],
                valid_pixels=int(supervision['valid'].sum()),ignored_pixels=int(ignored.sum()),
                ignored_output_gradient_L1=ignored_mass,loss=float(ordinary.detach()),ignored_truth_invariant=True))
            write(output/'status.json',dict(status='running',worker_pid=os.getpid(),probes_passed=len(results)))
            print(json.dumps(results[-1]),flush=True)
            del result,ordinary,perturbed,gradient;gc.collect()
        data.raw.cache_clear();del data;gc.collect()
    require(len(results)==12,'all fold/domain/split probes')
    write(output/'summary.json',dict(status='passed_supervision_gate',worker_pid=os.getpid(),results=results,
        source_sha256=digest(__file__),mask_adapter_sha256=digest(Path(__file__).with_name('supervision.py')),
        scope='Actual fresh noJ5 image-only FewClick forward/loss; ignored truth invariant and no ignored-pixel gradient; no fold training or new-class accuracy'))
    write(output/'status.json',dict(status='passed_supervision_gate',worker_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();gate(a.tasks,a.audit,a.output)
