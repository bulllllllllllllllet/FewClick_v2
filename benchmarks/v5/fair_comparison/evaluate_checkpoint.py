"""Full local point-group evaluation of a frozen retained or adapted checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from .adapted_models import build,METHODS
from .audit import digest,write
from .local_data import LocalData,fewclick_inputs
from .metrics import score,VERSION
from .protocol import nested
from .public_baselines import summarize
from .train_adapted import forward


class Retained(nn.Module):
    cells=True;contexts=True;closed=False
    def __init__(self,checkpoint,arm,device):
        super().__init__()
        from benchmarks.v5.paper_multiscale.model import load_model
        from benchmarks.v5.paper_multiscale.train import ARMS
        self.core,self.prior=load_model(device,*ARMS[arm]);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['contract']['arm']!=arm:raise ValueError('retained arm mismatch')
        self.core.load_state_dict(saved['model'],strict=True);self.contexts=ARMS[arm][0]
        self.prior.update(retained_target_supervision=True,original_selection=saved['step'])

    def forward(self,raw,prompts):
        images=fewclick_inputs(raw)
        return self.core(images,images,prompts,True,True,with_support=False)['query']


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--method',choices=list(METHODS)+['retained'],required=True);p.add_argument('--arm')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0');started=time.time()
    data=LocalData(a.tasks);contract=data.contract
    if a.method=='retained':
        model=Retained(a.checkpoint,a.arm,device);name='retained_'+a.arm
        saved_catalog=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['contract']['catalog_sha256']
        if saved_catalog!=contract['catalog_sha256']:raise ValueError('retained catalog mismatch')
    else:
        model=build(a.method,device);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if saved['contract']['method']!=a.method or saved['contract']['protocol_sha256']!=digest(a.tasks/'contract.json'):raise ValueError('adapted checkpoint protocol mismatch')
        model.load_state_dict(saved['model'],strict=True);del saved;name=a.method
    model.eval();plans=json.loads((a.tasks/'validation_maximum.json').read_text())
    if a.gate:plans=[next(p for p in plans if p['dataset']==d and p['prompt_group']==0) for d in sorted({p['dataset'] for p in plans})]
    write(a.output/'run.json',dict(method=name,gate=a.gate,protocol_sha256=digest(a.tasks/'contract.json'),
        checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),initialization=model.prior,
        worker_pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),metric_version=VERSION,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(f):digest(f) for f in Path(__file__).parent.glob('*.py')},
        scope='retained-J5 local rescore; not equal target supervision' if a.method=='retained' else 'adapted local development evaluation'))
    rows=[]
    with (a.output/'rows.jsonl').open('x') as handle,torch.inference_mode():
        for maximum in plans:
            for budget in contract['report_budgets']:
                plan=nested(maximum,budget);raw,prompts,supervision=data.materialize(plan,device,model.contexts,model.cells)
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):result=forward(model,raw,prompts,plan)
                probability=result['pixel_probability'].float();torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                if not torch.isfinite(probability).all():raise ValueError('nonfinite checkpoint inference')
                prediction=result['categorical_prediction'] if model.closed else probability>=.5
                record=score(prediction[0].cpu().numpy(),supervision['truth'][0].cpu().numpy(),supervision['valid'][0].cpu().numpy(),supervision['mpp'])
                meta=data.rows[plan['query_id']]
                row=plan|record|dict(latency_ms=elapsed,method=name,space_key=meta['space_key'],source_id=meta['source_id'],spatial_group_id=meta['spatial_group_id'])
                rows.append(row);handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            if len(rows)%80==0:
                record=dict(status='running',worker_pid=os.getpid(),completed=len(rows),total=4*len(plans),seconds=time.time()-started)
                write(a.output/'status.json',record);print(json.dumps(record),flush=True)
    report=dict(status='complete',method=name,gate=a.gate,worker_pid=os.getpid(),metrics=summarize(rows),
        selection_group0=summarize([r for r in rows if r['prompt_group']==0]),seconds=time.time()-started,
        metric_version=VERSION,rows_sha256=digest(a.output/'rows.jsonl'))
    write(a.output/'summary.json',report);write(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows=len(rows)))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
