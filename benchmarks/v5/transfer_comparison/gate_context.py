"""Actual MultiverSeg native gates under distinct, explicit source annotation budgets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .data import TransferData
from .multiverseg import MultiverSegContext
from benchmarks.v5.fair_comparison.audit import write
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
from benchmarks.v4.baseline.adapters.base import EpisodeRequest


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--weight',type=Path,required=True);p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);started=time.time();data=TransferData(a.tasks);device=torch.device('cpu')
    model=MultiverSegContext(a.sources,a.weight,a.dependencies,device)
    config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_zero_shot.yaml').read_text());config['device']='cpu';generator=SAMAdapter(config)
    write(a.output/'run.json',dict(worker_pid=os.getpid(),prior=model.prior,generator=generator.provenance(),
        modes=['model-predicted context from source points','full GT reference with complete annotation only'],
        query_GT_in_model=False,query_human_clicks=0,device='cpu'))
    cases=[];predicted={};excluded=[]
    for dataset in sorted({p['dataset'] for p in data.plans}):
        for presence in ('present','absent'):
            available=[p for p in data.plans if p['dataset']==dataset and p['query_presence']==presence and p['prompt_group']==0]
            for kind in ('model_prediction','full_GT'):
                if kind=='full_GT':
                    candidates=[p for p in available if not (data.labels(p['support_id'])==255).any()]
                    if not candidates:
                        excluded.append(dict(dataset=dataset,presence=presence,kind=kind,reason='no fully annotated support'))
                        continue
                    maximum=candidates[0]
                else:maximum=available[0]
                plan=nested(maximum,'3+2');value=data.materialize_inputs(plan,device,False,False)
                source=value['support']['rgb'];query=value['query']['rgb']
                source_gt=data.labels(plan['support_id']) if kind=='full_GT' else None
                if kind=='model_prediction':
                    key=plan['support_id'],tuple(map(tuple,plan['positive_xy'])),tuple(map(tuple,plan['negative_xy']))
                    if key not in predicted:
                        request=EpisodeRequest(str(plan['pair_id']),source[0].permute(1,2,0).numpy(),
                            np.asarray(plan['positive_xy'],np.float32)*512,np.asarray(plan['negative_xy'],np.float32)*512,'point')
                        prediction=generator.timed_predict(request);predicted[key]=prediction.binary_mask
                    mask=predicted[key];valid=None;annotated_pixels=len(plan['positive_xy'])+len(plan['negative_xy'])
                else:
                    mask=source_gt==plan['target'];valid=source_gt!=255;annotated_pixels=int(valid.sum())
                # The native model runs while query GT is unavailable. Full-GT
                # context is source-only and is deliberately a separate track.
                old=data.rows[plan['query_id']]['labels'];data.rows[plan['query_id']]['labels']='/query_GT_unavailable.png'
                result=model.predict(source,query,mask,mask_kind=kind,source_valid=valid,verify_native=True)
                data.rows[plan['query_id']]['labels']=old
                np.save(a.output/f'{dataset}_{presence}_{kind}.npy',result['pixel_probability'][0].numpy())
                cases.append(dict(dataset=dataset,presence=presence,mask_kind=kind,support=plan['support_id'],query=plan['query_id'],
                    source_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                    support_annotated_pixels=annotated_pixels,query_human_clicks=0,query_GT_unavailable=True,
                    native_forward_identical=result['native_forward_identical']))
                print(json.dumps(cases[-1]),flush=True)
    # Reject the previously dangerous ambiguity: incomplete support GT must not
    # become a complete negative context just because it has binary values.
    rejected=False
    try:model.predict(source,query,np.zeros((512,512),np.uint8),mask_kind='full_GT',source_valid=np.zeros((512,512),bool))
    except ValueError:rejected=True
    if not rejected:raise ValueError('incomplete source GT accepted')
    report=dict(status='passed',worker_pid=os.getpid(),cases=cases,excluded=excluded,
        incomplete_context_rejected=rejected,seconds=time.time()-started,
        scope='actual official v1 native context gates only; no complete accuracy or matched-prior claim')
    write(a.output/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
