"""Real frozen-backbone remote gates without using a query label loader."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .data import TransferData
from .prototypes import dense_prototype,region_prototype
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--method',choices=['fewclick','sam'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device(a.device);started=time.time()
    data=TransferData(a.tasks);cases=[]
    if a.method=='fewclick':
        from benchmarks.v5.paper_multiscale.model import load_model
        if a.checkpoint is None:raise ValueError('explicit retained checkpoint required')
        model,prior=load_model(device,True,False);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if saved['contract']['arm']!='multi_region':raise ValueError('same-feature graph diagnostic requires M3 without Dense')
        if saved['contract']['catalog_sha256']!=data.contract['catalog_sha256']:raise ValueError('model catalog mismatch')
        model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
        prior.update(checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),track='retained-J5 fixed-feature diagnostic; not clean adaptation')
    else:
        from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
        config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_zero_shot.yaml').read_text());config['device']=str(device)
        adapter=SAMAdapter(config);model=adapter.model;prior=adapter.provenance()
        from segment_anything.utils.transforms import ResizeLongestSide
        transform=ResizeLongestSide(model.image_encoder.img_size)
        def encode(raw):
            rgb=raw['rgb'][0].permute(1,2,0).cpu().numpy();image=transform.apply_image(rgb)
            tensor=torch.as_tensor(image,device=device).permute(2,0,1)
            return model.image_encoder(model.preprocess(tensor)[None])
    write(a.output/'run.json',dict(method=a.method,device=str(device),worker_pid=os.getpid(),prior=prior,
        tasks_sha256=digest(a.tasks/'contract.json'),source_sha256={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}))
    for dataset in sorted({p['dataset'] for p in data.plans}):
        for presence in ('present','absent'):
            maximum=next(p for p in data.plans if p['dataset']==dataset and p['query_presence']==presence and p['prompt_group']==0)
            plan=nested(maximum,'3+2');backups={}
            for identity in (plan['support_id'],plan['query_id']):
                backups[identity]=data.rows[identity]['labels'];data.rows[identity]['labels']='/GT_not_available_in_model_gate.png'
            value=data.materialize_inputs(plan,device,a.method=='fewclick',a.method=='fewclick')
            # Labels remain unavailable throughout BOTH encoders and the decoder.
            with torch.inference_mode():
                if a.method=='fewclick':
                    source=model.encode_image(fewclick_inputs(value['support']));query=model.encode_image(fewclick_inputs(value['query']))
                    task=model.support_task(source,value['prompts']);native,maps=model.decode(query,task,value['prompts'],False)
                    if maps.count_nonzero():raise ValueError('remote query received copied source point maps')
                    retrieved=region_prototype(source,query,value['prompts'])
                else:
                    source=encode(value['support']);query=encode(value['query'])
                    retrieved=dense_prototype(source,query,value['prompts']);native=None
                outputs={'prototype':retrieved}|({'learned_graph':native} if native is not None else {})
                for name,out in outputs.items():
                    probability=out['pixel_probability']
                    if probability.shape!=(1,512,512) or not torch.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ValueError('invalid remote probability')
                    np.save(a.output/f'{dataset}_{presence}_{name}.npy',probability[0].float().cpu().numpy())
            for identity,path in backups.items():data.rows[identity]['labels']=path
            cases.append(dict(dataset=dataset,presence=presence,support=plan['support_id'],query=plan['query_id'],
                both_GT_paths_unavailable=True,outputs=list(outputs),query_human_clicks=0,
                learned_query_maps_zero=True if a.method=='fewclick' else None))
            print(json.dumps(cases[-1]),flush=True)
    report=dict(status='passed',worker_pid=os.getpid(),method=a.method,cases=cases,seconds=time.time()-started,
        scope='real source/query feature and prediction gate; outputs are not complete benchmark accuracy')
    write(a.output/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
