"""Real central-tissue cache replay through disk-backed row streaming."""
import argparse
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import torch

from .stripe import StripeWriter
from .cache_gate import load_cache
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU gate must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    gate=json.loads((a.cache_gate/'summary.json').read_text());best=json.loads((a.training/'best_checkpoint.json').read_text())
    if gate['status']!='passed_real_WSI_cache_stitch_gate' or gate['selected_checkpoint_sha256']!=best['checkpoint_sha256']:
        raise ValueError('actual representative cache gate/weights required')
    if not all(c['fine_white_fraction']<.9 for c in gate['cache_records']):raise ValueError('tissue representative input required')
    model=load('fewclick_m3d_image',Path(best['path']),'cpu',gate['development_protocol_sha256']);data=LocalData(a.development)
    maximum=next(p for p in json.loads((a.development/'validation_maximum.json').read_text()) if p['pair_id']==gate['support_pair_id'] and p['prompt_group']==0)
    support,_,_=data.materialize(nested(maximum,'3+2'),'cpu',True,True);records=[]
    with torch.inference_mode():
        source=model.core.encode_image(fewclick_inputs(support))
        for result in gate['results']:
            _,prompts,_=data.materialize(nested(maximum,result['budget']),'cpu',True,True);task=model.core.support_task(source,prompts)
            first=gate['cache_records'][0]['identity']['geometry']['native_boxes'][0]
            writer=StripeWriter(a.output/result['budget'],result['bounds_level0'],first[2:])
            for record in gate['cache_records']:
                if digest(record['path'])!=record['sha256']:raise ValueError('cache changed')
                encoded=load_cache(Path(record['path']),best['checkpoint_sha256'],record['identity'])
                output,maps=model.core.decode(encoded,task,prompts,False)
                if maps.count_nonzero():raise ValueError('query prompt map nonzero')
                writer.push(output['pixel_probability'][0].numpy(),record['identity']['geometry']['native_boxes'][0])
            accepted=writer.finish();expected=np.load(result['stitched_probability'],allow_pickle=False)
            actual=np.load(a.output/result['budget']/'probability.npy',mmap_mode='r',allow_pickle=False)
            if not np.array_equal(actual,expected):raise ValueError('stream/full-memory probability mismatch')
            mask=np.load(a.output/result['budget']/'mask.npy',mmap_mode='r',allow_pickle=False)
            if not np.array_equal(mask,expected>=.5):raise ValueError('stream threshold mismatch')
            records.append(dict(budget=result['budget'],all_pixels_exact=True,**accepted))
    # Separate edge-overlap geometry stresses a last row/column with a shorter
    # stride; probability averaging must precede the final threshold.
    writer=StripeWriter(a.output/'edge_fixture',[0,0,3,3],[2,2])
    for value,box in zip((.1,.8,.2,.9),([0,0,2,2],[1,0,2,2],[0,1,2,2],[1,1,2,2])):
        writer.push(np.full((2,2),value,np.float32),box)
    fixture=writer.finish();p=np.load(a.output/'edge_fixture/probability.npy')
    if abs(float(p[1,1])-.5)>1e-7 or fixture['maximum_overlap']!=4:raise ValueError('edge overlap averaging')
    rejected=[]
    bad=StripeWriter(a.output/'hole_fixture',[0,0,4,2],[2,2]);bad.push(np.zeros((2,2)),[0,0,2,2])
    try:bad.finish()
    except ValueError:rejected.append('uncovered_domain')
    else:raise ValueError('uncovered domain accepted')
    bad=StripeWriter(a.output/'order_fixture',[0,0,2,3],[2,2]);bad.push(np.zeros((2,2)),[0,0,2,2]);bad.push(np.zeros((2,2)),[0,1,2,2])
    try:bad.push(np.zeros((2,2)),[0,0,2,2])
    except ValueError:rejected.append('out_of_order_row')
    else:raise ValueError('out of order row accepted')
    write(a.output/'summary.json',dict(status='passed_real_streaming_writer_gate',worker_pid=os.getpid(),results=records,rejected=rejected,
        selected_checkpoint_sha256=best['checkpoint_sha256'],source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('stripe.py'))},
        scope='Actual tissue cached-model decode and native-level0 disk row streaming, not full-slide execution or TLS accuracy'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache-gate','development','training','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
