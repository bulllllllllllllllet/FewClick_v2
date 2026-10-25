"""Exercise custom SAM/WSI-SAM feature wrappers with actual G3 weights, no GT."""
import argparse
import gc
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import torch
import torch.nn.functional as F

from .features import sam_features,wsi_features
from .evaluate import InferenceData
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU feature gate must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=InferenceData(a.tasks);plan=nested(next(p for p in data.plans if p['prompt_group']==0),'3+2');results=[]
    for method in ('sam_ft','wsi_sam_ft'):
        training=a.training_root/f'train_{method}_1025_131451'
        done=json.loads((training/'summary.json').read_text());best=json.loads((training/'best_checkpoint.json').read_text())
        if done['status']!='complete' or done['step']!=10000 or digest(best['path'])!=best['checkpoint_sha256']:
            raise ValueError('actual G3 weight evidence missing')
        model=load(method,Path(best['path']),'cpu',digest(a.development/'contract.json'))
        value=data.materialize_inputs(plan,'cpu',model.contexts,model.cells);encoder=sam_features if method=='sam_ft' else wsi_features
        with torch.inference_mode():
            source=encoder(model,value['support']);query=encoder(model,value['query'])
            expected=dense_prototype(source,query,value['prompts'])['pixel_probability']
            folder=a.output/method;folder.mkdir();cache=folder/'features.pth';torch.save(dict(source=source,query=query),cache)
            saved=torch.load(cache,map_location='cpu',weights_only=False)
            actual=dense_prototype(saved['source'],saved['query'],value['prompts'])['pixel_probability']
            if not torch.equal(expected,actual):raise ValueError('feature cache probability replay')
            if method=='wsi_sam_ft':
                # Independent central-grid geometry check using a coordinate ramp,
                # unrelated to learned predictions or query annotations.
                h,w=source.shape[-2:];yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
                grid=torch.stack([xx-.5,yy-.5],-1)[None];ramp=xx[None,None]
                sampled=F.grid_sample(ramp,grid,align_corners=False)[0,0]
                if not torch.allclose(sampled,.25+.5*xx,atol=1e-6,rtol=0):raise ValueError('2x central alignment geometry')
            p=actual[0].numpy();np.save(folder/'probability.npy',p)
            if not np.isfinite(p).all() or p.shape!=(512,512):raise ValueError('invalid feature retrieval output')
            results.append(dict(method=method,checkpoint_sha256=best['checkpoint_sha256'],feature_shape=list(source.shape),
                cache_sha256=digest(cache),probability_sha256=digest(folder/'probability.npy'),cache_max_error=0.,query_GT_access=False,
                scope='Mechanical helper gate with G3 weights trained on all classes; NOT heldout accuracy or heldout checkpoint gate'))
        print(json.dumps(results[-1]),flush=True);del model,source,query,saved,actual,expected;gc.collect()
    write(a.output/'summary.json',dict(status='passed_feature_wrapper_gate',worker_pid=os.getpid(),results=results,
        source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('features.py'))},
        semantic_evaluation=False,GT_forbidden_input_API=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('tasks','development','training-root','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
