"""Apply the exact current cell recipe to external fine/true-context fields."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .prepare import digest, write
from .cell_data import assemble
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector,CachedCenterCropper,DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models,WEIGHTS


def prepare(tasks,output,gate=False,reuse=None):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    started=time.time();completed=[]
    try:
        contract=json.loads((tasks/'contract.json').read_text())
        if digest(tasks/'fields.parquet')!=contract['fields_sha256']:
            raise ValueError('external field identity changed')
        rows=pd.read_parquet(tasks/'fields.parquet').to_dict('records')
        rows=sorted(rows,key=lambda r:r['tile_id'])
        if gate:rows=rows[:1]
        detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device)
        ctp,xcell,audit=load_feature_models(device);cropper=CachedCenterCropper()
        recipe=dict(detector='InstanSeg brightfield nuclei0.1.1',
            feature_weights={k:dict(path=str(v),sha256=digest(v)) for k,v in WEIGHTS.items() if k!='cpsam'},
            detector_sha256=digest(DEFAULT_INSTANSEG_CHECKPOINT),fine='prepared512 fine RGB with actual output MPP',
            context='true native RGB and native MPP; identical training prepare_variant',
            tissue_GT_used=False,nuclear_class='unknown',feature_dim=74,max_cells=255,
            XCell_training='internal dataset per user; current CRC/TIGER excluded',
            source_sha256={str(Path(inspect.getfile(function))):digest(inspect.getfile(function))
                           for function in (prepare_variant,InstanSegDetector,load_feature_models,native_context,assemble)})
        recipe_sha=__import__('hashlib').sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if reuse is not None:
            previous=json.loads((reuse/'summary.json').read_text())
            run=json.loads((reuse/'run.json').read_text())
            if previous['status']!='complete' or not previous['gate'] or run['recipe_sha256']!=recipe_sha or run['protocol_sha256']!=digest(tasks/'contract.json'):
                raise ValueError('native cell gate mismatch')
            completed=pd.read_parquet(reuse/'fields_with_cells.parquet').to_dict('records')
            for row in completed:
                assemble(row,'cpu')
            ids={r['tile_id'] for r in completed}
            rows=[r for r in rows if r['tile_id'] not in ids]
        elif not gate:
            raise ValueError('full native extraction requires real GPU gate')
        write(output/'run.json',dict(worker_pid=os.getpid(),gate=gate,protocol_sha256=digest(tasks/'contract.json'),
            recipe=recipe,recipe_sha256=recipe_sha,audit=audit,device=str(device),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            source_sha256=digest(__file__),reuse=str(reuse) if reuse else None))
        with (output/'progress.jsonl').open('x') as log:
            for row in rows:
                geometry=json.loads(row['multiscale_json']);images=json.loads(row['images_json'])
                if digest(geometry['source_image'])!=row['native_image_sha256']:
                    raise ValueError('native external RGB changed')
                arrays={};details=[]
                for scale in (0,1,2):
                    if scale==0:
                        if digest(images[0]['path'])!=images[0]['sha256']:raise ValueError('fine RGB changed')
                        rgb=np.asarray(Image.open(images[0]['path']).convert('RGB')).copy()
                        mpp=json.loads(row['input_mpp_xy_json'])
                    else:
                        rgb=native_context(row,scale);mpp=geometry['native_mpp']
                    h,w=rgb.shape[:2]
                    meta=dict(input_to_image_affine=[w/512,0,0,0,h/512,0],source_image_size=[w,h])
                    cell,info=prepare_variant(rgb,mpp,meta,detector,ctp,xcell,device,cropper=cropper)
                    arrays[f'cells_{scale}']=cell['features'];arrays[f'total_{scale}']=cell['total']
                    arrays[f'meta_{scale}']=cell['meta'];arrays[f'reg_{scale}']=cell['reg']
                    details.append(info)
                path=output/f'{row["tile_id"]}.npz'
                np.savez_compressed(path,**arrays)
                with np.load(path,allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[k],v) for k,v in arrays.items()):raise ValueError('cell disk reload mismatch')
                prepared=row|dict(cell_cache=str(path),cell_cache_sha256=digest(path),cell_cache_status='complete')
                assemble(prepared|dict(labels='/unavailable/tissue_GT.png'),'cpu')
                completed.append(prepared)
                log.write(json.dumps(dict(tile_id=row['tile_id'],details=details))+'\n');log.flush()
                write(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(completed),seconds=time.time()-started))
                print(json.dumps(dict(completed=len(completed),tile_id=row['tile_id'],cell_totals=[int(arrays[f'total_{s}']) for s in (0,1,2)])),flush=True)
        pd.DataFrame(completed).to_parquet(output/'fields_with_cells.parquet',index=False)
        if len(completed)!=(1 if gate else contract['selected_fields']):raise ValueError('incomplete external cell fields')
        write(output/'summary.json',dict(status='complete',worker_pid=os.getpid(),gate=gate,fields=len(completed),
            fields_sha256=digest(output/'fields_with_cells.parquet'),recipe_sha256=recipe_sha,
            tissue_GT_used=False,all_disk_roundtrips=True,seconds=time.time()-started))
        write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),fields=len(completed)))
    except Exception:
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),completed=len(completed),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--reuse',type=Path)
    a=p.parse_args();prepare(a.tasks,a.output,a.gate,a.reuse)
