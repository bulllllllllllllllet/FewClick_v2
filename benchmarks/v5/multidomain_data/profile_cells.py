"""Synchronized stage timing of the real cell-cache pipeline, on fixed inputs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
import torch

from . import prepare_cells as pipeline
from .prepare import dump,read_rows,sha


def profile(inputs,output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cuda:0')
    started=time.perf_counter();detector,ctp,xcell,audit=pipeline.load_cell_models(device)
    stages=defaultdict(float);counts=defaultdict(int);handles=[]
    def cpu_wrap(name,fn):
        def call(*a,**kw):
            tick=time.perf_counter();result=fn(*a,**kw)
            stages[name]+=time.perf_counter()-tick;counts[name]+=1
            return result
        return call
    detector_eval=detector.eval
    class TimedDetector:
        def eval(self,*a,**kw):
            torch.cuda.synchronize();tick=time.perf_counter();result=detector_eval(*a,**kw);torch.cuda.synchronize()
            stages['segmentation_seconds']+=time.perf_counter()-tick;counts['segmentation_seconds']+=1
            return result
    for name,module in [('ctranspath_seconds',ctp),('xcell_seconds',xcell)]:
        clock=[]
        def before(m,a,clock=clock):
            torch.cuda.synchronize();clock.append(time.perf_counter())
        def after(m,a,o,name=name,clock=clock):
            torch.cuda.synchronize();stages[name]+=time.perf_counter()-clock.pop();counts[name]+=1
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    originals={n:getattr(pipeline,n) for n in ['attended_crop','filter_padding','cells_from_instances']}
    for name,fn in originals.items():
        setattr(pipeline,name,cpu_wrap({'attended_crop':'crop_seconds','filter_padding':'padding_filter_seconds','cells_from_instances':'geometry_seconds'}[name],fn))
    rows=[]
    try:
        for row in read_rows(inputs):
            stages.clear();counts.clear();tick=time.perf_counter()
            if sha(Path(row['image']).read_bytes())!=row['image_sha256']:raise ValueError('RGB identity changed')
            image=np.asarray(Image.open(row['image']).convert('RGB')).copy();io=time.perf_counter()-tick
            torch.cuda.synchronize();tick=time.perf_counter()
            arrays,details=pipeline.prepare_variant(image,json.loads(row['input_mpp_xy_json']),json.loads(row['preparation_json']),TimedDetector(),ctp,xcell,device)
            torch.cuda.synchronize();total=time.perf_counter()-tick
            result=dict(dataset=row['dataset'],quality=row['quality'],tile_id=row['tile_id'],total_seconds=total,input_io_seconds=io,
                other_seconds=total-sum(stages.values()),retained_cells=len(arrays['features']),total_cells=details['total_cell_count'],
                stages=dict(stages),calls=dict(counts))
            rows.append(result);dump(output/f'case_{len(rows):03d}.json',result)
            dump(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(rows)))
            print('PROFILE',json.dumps(result),flush=True)
        names=sorted({k for r in rows for k in r['stages']})
        summary=dict(status='complete',worker_pid=os.getpid(),samples=len(rows),seconds=time.perf_counter()-started,
            mean_total_seconds=float(np.mean([r['total_seconds'] for r in rows])),
            mean_stage_seconds={k:float(np.mean([r['stages'].get(k,0) for r in rows])) for k in names},
            mean_other_seconds=float(np.mean([r['other_seconds'] for r in rows])),rows=rows,
            source_sha256=sha(Path(__file__).read_bytes()),pipeline_sha256=sha(Path(pipeline.__file__).read_bytes()),
            inputs_sha256=sha(inputs.read_bytes()),model_audit=audit,timing='synchronized GPU stage wall time; image IO/serialization separate')
        dump(output/'summary.json',summary);dump(output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(rows)))
    except Exception as exc:
        dump(output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc),completed=len(rows)));raise
    finally:
        for name,fn in originals.items():setattr(pipeline,name,fn)
        for handle in handles:handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();profile(a.inputs,a.output)


if __name__=='__main__':main()
