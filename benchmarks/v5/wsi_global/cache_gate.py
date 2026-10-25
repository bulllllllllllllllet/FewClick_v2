"""Real CPU WSI cache/task-update/stitch probe; no TLS accuracy claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import pyvips
import numpy as np
from PIL import Image
import torch

from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def raw_images(images):
    raw = {}
    for scale,image in enumerate(images):
        suffix = '' if scale == 0 else f'_{scale}'
        raw['image'+suffix] = he_to_tensor(image.copy())[None]
        # Only legal for the explicitly image-only control. No cell detection
        # or appearance features are represented by these ignored placeholders.
        raw['cells'+suffix] = torch.zeros(1,1,74)
        raw['cell_valid'+suffix] = torch.zeros(1,1,dtype=torch.bool)
        raw['total_cell_count' if scale == 0 else f'total_{scale}'] = torch.zeros(1)
    return raw


def load_cache(path, checkpoint_sha, identity):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if payload['checkpoint_sha256'] != checkpoint_sha or payload['identity'] != identity:
        raise ValueError('cache checkpoint/input identity mismatch')
    return payload['encoded']


def stitch(probabilities,boxes):
    left=min(b[0] for b in boxes);top=min(b[1] for b in boxes)
    right=max(b[0]+b[2] for b in boxes);bottom=max(b[1]+b[3] for b in boxes)
    total=np.zeros((bottom-top,right-left),np.float32);weight=np.zeros_like(total)
    native=[]
    for p,(x,y,w,h) in zip(probabilities,boxes):
        value=np.asarray(Image.fromarray(p.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
        total[y-top:y-top+h,x-left:x-left+w]+=value;weight[y-top:y-top+h,x-left:x-left+w]+=1;native.append(value)
    if not (weight>0).all():raise ValueError('uncovered stitch gate canvas')
    result=total/weight
    # Direct coordinate-wise overlap recount, separate from rectangular addition.
    for y,x in [(0,0),(result.shape[0]//2,result.shape[1]//2),(result.shape[0]-1,result.shape[1]-1)]:
        values=[p[y+top-by,x+left-bx] for p,(bx,by,w,h) in zip(native,boxes) if bx<=x+left<bx+w and by<=y+top<by+h]
        if abs(float(result[y,x])-float(np.mean(values)))>1e-7:raise ValueError('overlap mean mismatch')
    return result,weight,[left,top,right,bottom]


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU gate must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    training=json.loads((a.training/'summary.json').read_text());best=json.loads((a.training/'best_checkpoint.json').read_text())
    if training['status']!='complete' or training['step']!=10000 or not training['full_manifest_parity']:
        raise ValueError('formal selected G3 image-only training required')
    expected='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe'
    if training['training_consumed_sha256']!=expected or digest(a.training/'training_consumed.jsonl')!=expected:
        raise ValueError('actual development task identity mismatch')
    if digest(best['path'])!=best['checkpoint_sha256']:raise ValueError('selected weights changed')
    plan_summary=json.loads((a.plan/'summary.json').read_text())
    if plan_summary['status']!='complete_spatial_plan' or digest(a.plan/'slides_private.json')!=plan_summary['slides_sha256']:
        raise ValueError('complete full-cohort spatial plan required')
    record=json.loads((a.plan/'slides_private.json').read_text())[0]
    model=load('fewclick_m3d_image',Path(best['path']),torch.device('cpu'),digest(a.development/'contract.json'))
    if model.prior['cell_mode']!='none':raise ValueError('cell-free probe cannot represent full cell branch')
    original_encode=model.core.encode_image;calls=[]
    def encode(inputs):calls.append(1);return original_encode(inputs)
    model.core.encode_image=encode
    data=LocalData(a.development);maximum=next(p for p in json.loads((a.development/'validation_maximum.json').read_text()) if p['dataset']=='crc' and p['prompt_group']==0)
    support,prompts,_=data.materialize(nested(maximum,'3+2'),'cpu',True,True)
    timings={};tick=time.perf_counter()
    with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(support))
    timings['support_encode_seconds']=time.perf_counter()-tick
    slide=StrictSlide(record['provenance']['wsi_path']);queries=[];cache_records=[]
    try:
        fw,fh=record['geometry']['fine_native_wh'];x0=record['geometry']['x_origins'][0];y0=record['geometry']['y_origins'][0]
        if a.origin is not None:
            selected=json.loads(a.origin.read_text());x0,y0=selected['origin_level0']
            if x0 not in record['geometry']['x_origins'] or y0 not in record['geometry']['y_origins']:
                raise ValueError('gate origin outside planned grid')
        centres=[[x+fw/2,y+fh/2] for y in (y0,y0+fh//2) for x in (x0,x0+fw//2)]
        for number,centre in enumerate(centres):
            tick=time.perf_counter();images,geo=slide.fields(centre);read_seconds=time.perf_counter()-tick
            raw=raw_images(images);input_sha=[hashlib.sha256(im.tobytes()).hexdigest() for im in images]
            tick=time.perf_counter()
            with torch.inference_mode():encoded=model.core.encode_image(raw)
            encode_seconds=time.perf_counter()-tick;path=a.output/f'query_cache_{number}.pth'
            identity=dict(slide_id=record['slide_id'],geometry=geo,RGB_array_sha256=input_sha)
            tick=time.perf_counter();torch.save(dict(encoded=encoded,identity=identity,checkpoint_sha256=best['checkpoint_sha256']),path)
            disk_write=time.perf_counter()-tick;tick=time.perf_counter();loaded=load_cache(path,best['checkpoint_sha256'],identity);disk_read=time.perf_counter()-tick
            queries.append(dict(raw=raw,encoded=loaded,box=geo['native_boxes'][0]))
            cache_records.append(dict(identity=identity,path=str(path),sha256=digest(path),bytes=path.stat().st_size,
                fine_RGB_std=float(images[0].std()),fine_white_fraction=float((images[0].min(2)>240).mean()),
                read_seconds=read_seconds,encode_seconds=encode_seconds,cache_write_seconds=disk_write,cache_read_seconds=disk_read))
            write(a.output/'status.json',dict(status='encoding',worker_pid=os.getpid(),queries=len(queries),total=4))
    finally:slide.close()
    before=len(calls);results=[]
    for budget in ('1+1','3+2'):
        _,prompts,_=data.materialize(nested(maximum,budget),'cpu',True,True);tick=time.perf_counter()
        with torch.inference_mode():
            task=model.core.support_task(source,prompts);probabilities=[]
            for query in queries:
                output,maps=model.core.decode(query['encoded'],task,prompts,False)
                if maps.count_nonzero():raise ValueError('support click coordinates leaked into query')
                p=output['pixel_probability'][0].numpy().copy()
                if not np.isfinite(p).all():raise ValueError('nonfinite WSI prediction')
                probabilities.append(p)
        cached_seconds=time.perf_counter()-tick
        if len(calls)!=before:raise ValueError('online task update re-encoded query')
        probability,weight,bounds=stitch(probabilities,[q['box'] for q in queries]);path=a.output/f'stitched_probability_{budget}.npy';np.save(path,probability)
        np.save(a.output/f'stitched_mask_{budget}.npy',probability>=.5)
        results.append(dict(budget=budget,probabilities=probabilities,cached_task_decode_seconds=cached_seconds,
            stitched_probability=str(path),sha256=digest(path),bounds_level0=bounds,maximum_overlap=int(weight.max())))
    errors=[]
    # Native complete forwards happen only AFTER both cached task updates.
    with torch.inference_mode():
        for item in results:
            _,prompts,_=data.materialize(nested(maximum,item['budget']),'cpu',True,True)
            for query,expected in zip(queries,item.pop('probabilities')):
                direct=model.core(fewclick_inputs(support),query['raw'],prompts,False,False,with_support=False)
                if direct['query_prompt_maps'].count_nonzero():raise ValueError('native query maps nonzero')
                error=float(np.abs(direct['query']['pixel_probability'][0].numpy()-expected).max());errors.append(error)
                if error!=0:raise ValueError('cached/native WSI prediction differs')
    # A concrete probability fixture detects threshold-before-averaging errors.
    fixture,_,_=stitch([np.full((2,2),.1,np.float32),np.full((2,2),.8,np.float32)],[[0,0,2,2],[0,0,2,2]])
    if (fixture>=.5).any():raise ValueError('threshold order wrong')
    rejected=False
    try:load_cache(Path(cache_records[0]['path']),'0'*64,cache_records[0]['identity'])
    except ValueError:rejected=True
    if not rejected:raise ValueError('wrong checkpoint cache accepted')
    write(a.output/'summary.json',dict(status='passed_real_WSI_cache_stitch_gate',worker_pid=os.getpid(),slide_id=record['slide_id'],
        selected_checkpoint_sha256=best['checkpoint_sha256'],development_protocol_sha256=digest(a.development/'contract.json'),
        gate_origin_provenance=None if a.origin is None else dict(path=str(a.origin),sha256=digest(a.origin)),
        plan_sha256=digest(a.plan/'slides_private.json'),support_pair_id=maximum['pair_id'],support_protocol='Fixed CRC development task; not TLS prompts/accuracy',
        cache_records=cache_records,results=results,native_errors=errors,online_query_encode_calls=0,query_prompt_maps_zero=True,
        wrong_weight_cache_rejected=True,threshold_after_probability_stitch=True,timings=timings,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('plan.py'),Path('benchmarks/v5/external_evaluation/wsi_reader.py'))},
        scope='CPU4-query mechanical image-only gate; not full-slide execution, full-cell runtime or TLS semantic accuracy'))
    write(a.output/'status.json',dict(status='passed_real_WSI_cache_stitch_gate',worker_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','development','training','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--origin',type=Path)
    run(p.parse_args())
