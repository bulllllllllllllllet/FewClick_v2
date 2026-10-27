"""Gate the independent binary float32 replayer on actual accepted CPU WSI outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from .probability_replay import replay,require
from .cache_gate import load_cache
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def read(p):return json.loads(Path(p).read_text())


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4);require(not torch.cuda.is_available(),'CPU gate')
    contract=read(a.source/'contract.json');done=read(a.source/'summary.json');require(done['status']=='complete_stream_execution','actual original CPU WSI complete')
    entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()];boxes=[e['identity']['geometry']['native_boxes'][0] for e in entries]
    xs=sorted({b[0] for b in boxes});ys=sorted({b[1] for b in boxes});indices=[i for i,b in enumerate(boxes) if b[0] in xs[-2:] and b[1] in ys[-2:]];require(len(indices)==4,'fixed last2x2 actual overlap')
    data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
    model=load('fewclick_m3d_image',Path(contract['checkpoint']),'cpu',contract['development_protocol_sha256']);raw,_,_=data.materialize(nested(maximum,'1+1'),'cpu',True,True);results=[]
    with torch.inference_mode():
        support=model.core.encode_image(fewclick_inputs(raw))
        for budget in ('1+1','3+2'):
            _,p,_=data.materialize(nested(maximum,budget),'cpu',True,True);task=model.core.support_task(support,p)
            parent=next(d for d in done['passes'] if d['budget']==budget);arrays={n:np.load(parent['outputs'][n]['path'],mmap_mode='r') for n in ('probability','mask','overlap')}
            def decode(i):
                e=entries[i];cached=load_cache(Path(e['path']),contract['checkpoint_sha256'],e['identity']);out,maps=model.core.decode(cached,task,p,False);require(not maps.count_nonzero(),'no query maps');return out['pixel_probability'][0].numpy()
            results.append(dict(budget=budget,**replay(indices,boxes,decode,arrays,contract['bounds_level0'],atol=0.)))
    helper=Path('benchmarks/v5/wsi_global/probability_replay.py')
    result=dict(status='passed_actual_float32_probability_replay_gate',worker_pid=os.getpid(),results=results,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.perf_counter()-started,scope='actual NoCell CPU last4 overlapping tiles; validates generic replay, not full-cell GPU backbone or whole-output acceptance')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','development','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
