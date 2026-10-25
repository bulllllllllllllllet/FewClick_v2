"""Real cached-data/model replay before scheduling new native GPU extraction."""
import argparse
import json
from pathlib import Path

import pyvips
import numpy as np
from PIL import Image
import torch

from .prepare import digest,write
from .cell_data import assemble
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.train_adapted import forward,loss_for
from benchmarks.v5.fair_comparison.metrics import score


def gate(tasks,checkpoint,output):
    if torch.cuda.is_available():raise ValueError('CPU gate must hide CUDA')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=LocalData(tasks);device=torch.device('cpu')
    model=build('fewclick_m3d_xcell',device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['contract']['method']!='fewclick_m3d_xcell':raise ValueError('wrong real model checkpoint')
    model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
    results=[]
    for dataset in ('crc','tiger'):
        plan=next(p for p in data.validation if p['dataset']==dataset)
        row=data.rows[plan['query_id']];folder=output/dataset;folder.mkdir()
        original,gt=data.raw(row['tile_id'],True,True)
        variant=json.loads(row['variants_json'])['original'];arrays={};images=[]
        with np.load(variant['cells'],allow_pickle=False) as fine:
            arrays['cells_0']=fine['features'].copy();arrays['total_0']=fine['total'].copy()
        with np.load(row['multiscale_cache'],allow_pickle=False) as context:
            for scale in (1,2):
                arrays[f'cells_{scale}']=context[f'cells_{scale}'].copy();arrays[f'total_{scale}']=context[f'total_{scale}'].copy()
            for scale in (0,1,2):
                image=np.asarray(Image.open(variant['image']).convert('RGB')) if scale==0 else context[f'image_{scale}']
                path=folder/f'image_{scale}.png';Image.fromarray(image).save(path)
                images.append(dict(path=str(path),sha256=digest(path)))
        cache=folder/'real_cells.npz';np.savez(cache,**arrays)
        external=dict(images_json=json.dumps(images),cell_cache=str(cache),cell_cache_sha256=digest(cache),
                      cell_cache_status='complete',labels='/deliberately_missing/query_GT.png')
        assembled=assemble(external,device)
        if set(assembled)!=set(original) or any(not torch.equal(assembled[k],original[k]) for k in original):
            raise ValueError('new cell assembly differs from real training inputs')
        raw,prompts,supervision=data.materialize(plan,device,True,True)
        with torch.inference_mode():
            actual=forward(model,assembled,prompts,plan)
            legacy=forward(model,raw,prompts,plan)
            if not torch.equal(actual['pixel_probability'],legacy['pixel_probability']):raise ValueError('model replay differs')
            loss=loss_for(model,actual,supervision,prompts)
        if not torch.isfinite(loss):raise ValueError('nonfinite real loss')
        probability=actual['pixel_probability'][0].numpy()
        prediction=probability>=.5
        metric=score(prediction,gt==plan['target'],gt!=255,json.loads(row['input_mpp_xy_json']))
        results.append(dict(dataset=dataset,input_keys=len(original),tensor_replay_exact=True,model_replay_exact=True,
            loss=float(loss),metrics=metric,real_cell_counts=[len(arrays[f'cells_{s}']) for s in (0,1,2)]))
    write(output/'summary.json',dict(status='passed',GT_unavailable_assembly=True,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),tasks=str(tasks),
        source_sha256={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('cell_data.py'),Path(__file__).with_name('cells.py')]},
        domains=results,scope='CPU assembly/model/loss replay on real existing CRC/TIGER cells, not external extraction or accuracy'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();gate(a.tasks,a.checkpoint,a.output)
