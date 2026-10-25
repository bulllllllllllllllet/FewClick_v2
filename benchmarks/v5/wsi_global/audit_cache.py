"""Fresh model/cache replay and full-pixel independent overlap aggregation."""
import argparse
import json
from pathlib import Path

import pyvips
import numpy as np
from PIL import Image
import torch

from .cache_gate import load_cache
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU independent audit must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    gate=json.loads((a.gate/'summary.json').read_text());best=json.loads((a.training/'best_checkpoint.json').read_text())
    if gate['status']!='passed_real_WSI_cache_stitch_gate' or best['checkpoint_sha256']!=gate['selected_checkpoint_sha256']:
        raise ValueError('accepted gate/weight required')
    for path,sha in gate['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('gate source changed')
    model=load('fewclick_m3d_image',Path(best['path']),'cpu',gate['development_protocol_sha256'])
    data=LocalData(a.development)
    maximum=next(p for p in json.loads((a.development/'validation_maximum.json').read_text()) if p['pair_id']==gate['support_pair_id'] and p['prompt_group']==0)
    raw,_,_=data.materialize(nested(maximum,'3+2'),'cpu',True,True)
    caches=[]
    for record in gate['cache_records']:
        if digest(record['path'])!=record['sha256']:raise ValueError('encoded cache file changed')
        caches.append(load_cache(Path(record['path']),gate['selected_checkpoint_sha256'],record['identity']))
    checks=[]
    with torch.inference_mode():
        source=model.core.encode_image(fewclick_inputs(raw))
        for result in gate['results']:
            _,prompts,_=data.materialize(nested(maximum,result['budget']),'cpu',True,True)
            task=model.core.support_task(source,prompts)
            left,top,right,bottom=result['bounds_level0'];layers=[]
            for encoded,record in zip(caches,gate['cache_records']):
                output,maps=model.core.decode(encoded,task,prompts,False)
                if maps.count_nonzero():raise ValueError('query prompt maps nonzero')
                p=output['pixel_probability'][0].numpy();x,y,w,h=record['identity']['geometry']['native_boxes'][0]
                native=np.asarray(Image.fromarray(p.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
                layer=np.full((bottom-top,right-left),np.nan,np.float32)
                layer[y-top:y-top+h,x-left:x-left+w]=native;layers.append(layer)
            # Different representation and aggregation from rectangular running
            # sums in the writer: independent full-frame NaN-layer reduction.
            expected=np.nanmean(np.stack(layers),axis=0)
            path=Path(result['stitched_probability'])
            if digest(path)!=result['sha256']:raise ValueError('stitched output changed')
            actual=np.load(path,allow_pickle=False);error=float(np.abs(actual-expected).max())
            if not np.isfinite(actual).all() or error>1e-7:raise ValueError('all-pixel stitch replay mismatch')
            mask=np.load(a.gate/f'stitched_mask_{result["budget"]}.npy',allow_pickle=False)
            if not np.array_equal(mask,actual>=.5):raise ValueError('stored final threshold mismatch')
            checks.append(dict(budget=result['budget'],pixels=int(actual.size),maximum_probability_error=error,mask_threshold_exact=True))
    bad=dict(gate['cache_records'][0]['identity']);bad['slide_id']='wrong-input'
    rejected=False
    try:load_cache(Path(gate['cache_records'][0]['path']),gate['selected_checkpoint_sha256'],bad)
    except ValueError:rejected=True
    if not rejected:raise ValueError('wrong input cache accepted')
    write(a.output/'summary.json',dict(status='passed_independent_full_pixel_cache_audit',checks=checks,
        checkpoint_sha256=gate['selected_checkpoint_sha256'],wrong_input_cache_rejected=True,source_sha256=digest(__file__),
        scope='Fresh selected model/cache replay; all pixels of two small stitched probes, not full WSI accuracy/efficiency'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('gate','development','training','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
