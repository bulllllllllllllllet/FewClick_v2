"""Semantic retrieval tests plus actual-label independence of the image loader."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .data import TransferData
from .prototypes import dense_prototype,region_prototype
from benchmarks.v5.fair_comparison.audit import write,digest


def algebra():
    # Source positive left / negative right. Query has the opposite arrangement;
    # a correct transfer follows appearance, not the support point coordinates.
    source=torch.zeros(1,2,8,8);source[:,0,:,:4]=1;source[:,1,:,4:]=1
    query=source.flip(-1)
    prompts=dict(positive_xy=torch.tensor([[[.25,.5]]]),negative_xy=torch.tensor([[[.75,.5]]]),
                 positive_mask=torch.ones(1,1,dtype=torch.bool),negative_mask=torch.ones(1,1,dtype=torch.bool))
    result=dense_prototype(source,query,prompts,(8,8))['pixel_probability']
    assert (result[:,:,:4]<.5).all() and (result[:,:,4:]>.5).all()
    assignment=source.clone()
    support=dict(assignment_low=assignment,tokens=[torch.eye(2)[None]])
    candidate=dict(assignment_low=assignment,tokens=[torch.eye(2).flip(0)[None]])
    regional=region_prototype(support,candidate,prompts,(8,8))['pixel_probability']
    assert (regional[:,:,:4]<.5).all() and (regional[:,:,4:]>.5).all()
    return dict(dense_follows_query_appearance=True,region_follows_query_appearance=True)


def actual_data(tasks,output):
    torch.set_num_threads(4);tests=algebra();data=TransferData(tasks)
    cases=[]
    for dataset in sorted({p['dataset'] for p in data.plans}):
        for presence in ('present','absent'):
            plan=next(p for p in data.plans if p['dataset']==dataset and p['query_presence']==presence)
            # Break BOTH GT paths before loading inputs. Inference must still work.
            original=copy.deepcopy(data.rows)
            for identity in (plan['support_id'],plan['query_id']):
                data.rows[identity]['labels']='/unavailable_for_inference/GT.png'
                data.rows[identity]['counts_json']='UNAVAILABLE';data.rows[identity]['eligible_json']='UNAVAILABLE'
            data.inputs.cache_clear()
            value=data.materialize_inputs(plan,torch.device('cpu'),True,True)
            assert not value['local'] and not value['shared']
            assert value['support']['image'].shape==(1,3,512,512)
            assert value['query']['image_2'].shape==(1,3,512,512)
            forbidden={'truth','valid','labels','target','dataset','query_prompt_maps'}
            assert not forbidden.intersection(value['support']) and not forbidden.intersection(value['query'])
            cases.append(dict(dataset=dataset,presence=presence,source=plan['support_id'],query=plan['query_id'],unavailable_GT_input_load_passed=True))
            data.rows=original
    report=dict(status='passed',prototype_tests=tests,real_cases=cases,protocol_sha256=digest(tasks/'contract.json'),
        limitations='CPU algebra and real image/cache loading; trained-backbone remote prediction still requires a model gate')
    write(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();actual_data(a.tasks,a.output)
