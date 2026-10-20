"""Fixed-input legacy/cached-crop/InstanSeg end-to-end cache speed comparison."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from . import prepare_cells as pipeline
from .fast_nuclei import CachedCenterCropper,InstanSegDetector
from .prepare import read_rows,dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cuda:0');start=time.perf_counter();rows=[];variants=[]
    original_crop=pipeline.attended_crop
    try:
        detector,ctp,xcell,audit=pipeline.load_cell_models(device)
        instanseg=InstanSegDetector(a.checkpoint,device)
        inputs=read_rows(a.inputs)
        # Explicit warmup excluded from paired measurements, never fit to GT.
        first=inputs[0];rgb=np.asarray(Image.open(first['image']).convert('RGB')).copy()
        high=np.asarray(Image.fromarray(rgb).resize((1024,1024),Image.Resampling.BILINEAR)).copy()
        detector.eval(high,diameter=18,batch_size=1);instanseg.eval(high)
        for number,row in enumerate(inputs):
            if sha(Path(row['image']).read_bytes())!=row['image_sha256']:raise ValueError('image identity changed')
            image=np.asarray(Image.open(row['image']).convert('RGB')).copy();results={}
            for arm,model,cropper in [('cellpose_legacy',detector,original_crop),('cellpose_cached_centers',detector,CachedCenterCropper()),('instanseg_cached_centers',instanseg,CachedCenterCropper())]:
                pipeline.attended_crop=cropper
                torch.cuda.synchronize();tick=time.perf_counter()
                arrays,details=pipeline.prepare_variant(image,json.loads(row['input_mpp_xy_json']),json.loads(row['preparation_json']),model,ctp,xcell,device)
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                path=a.output/f'{number:03d}_{arm}.npz';np.savez_compressed(path,**arrays)
                result=dict(tile_id=row['tile_id'],dataset=row['dataset'],quality=row['quality'],arm=arm,seconds=seconds,
                    total_cells=details['total_cell_count'],retained_cells=len(arrays['features']),cells=str(path))
                rows.append(result);results[arm]=arrays
                if arm=='instanseg_cached_centers':
                    variants.append(dict(row,cells=str(path),cells_sha256=sha(path.read_bytes()),cell_features_status='complete',
                        cell_preparation_json=json.dumps(details|{'detector':'InstanSeg brightfield_nuclei0.1.1','model_mpp':instanseg.pixel_size,'cached_centers':True}),cell_preparation_seconds=seconds))
                print('SPEED',json.dumps(result),flush=True)
            # Exact preservation for the cached-center-only optimization.
            left,right=results['cellpose_legacy'],results['cellpose_cached_centers']
            if not np.array_equal(left['instances'],right['instances']):raise ValueError('repeated Cellpose masks differ; crop-only equivalence unresolved')
            for key in ['meta','reg','features','total']:
                if not np.array_equal(left[key],right[key]):raise ValueError(f'cached-center optimization changed {key}')
            dump(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed_pairs=number+1,total_pairs=len(inputs)))
        stats={arm:dict(mean_seconds=float(np.mean([r['seconds'] for r in rows if r['arm']==arm])),
                        median_seconds=float(np.median([r['seconds'] for r in rows if r['arm']==arm]))) for arm in sorted({r['arm'] for r in rows})}
        dump(a.output/'summary.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.perf_counter()-start,rows=rows,arms=stats,
            cached_center_full_features_exact=True,tissue_GT_used=False,source_sha256=sha(Path(__file__).read_bytes()),
            instanseg_sha256=instanseg.weight_sha256,input_manifest_sha256=sha(a.inputs.read_bytes()),
            scope='10 fixed five-source/quality inputs; throughput, exact crop optimization, no target performance selection'))
        pd.DataFrame(variants).to_parquet(a.output/'instanseg_inputs_with_cells.parquet',index=False)
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed_pairs=len(inputs)))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise
    finally:pipeline.attended_crop=original_crop


if __name__=='__main__':main()
