"""Public-weight local baselines on a shared immutable point/task contract."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from .audit import digest, write
from .metrics import score, VERSION
from .protocol import nested
from benchmarks.v4.baseline.adapters.base import EpisodeRequest
from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
from benchmarks.v4.baseline.integrations.sam_external import WSISAMBackend


class SharedContextWSISAM(WSISAMBackend):
    @staticmethod
    def _low_context(inputs):
        # The same prepared 512x512 physical 2x field as FewClick. The official
        # integration resizes it to 1024; no fine-image enlargement pretends to be context.
        return inputs['low']


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row['click_budget']].append(row)
    result={}
    for budget,group in groups.items():
        pairs=defaultdict(list)
        for r in group: pairs[r['pair_id']].append(r)
        pair_dice=[np.mean([r['dice'] for r in values]) for values in pairs.values()]
        bf=[r['bf1_um2'] for r in group if r['bf1_um2'] is not None]
        result[budget]=dict(tasks=len(group),pairs=len(pairs),macro_pair_dice=float(np.mean(pair_dice)),
            bf1_um2=float(np.mean(bf)) if bf else None,bf1_evaluable=len(bf),
            latency_ms_mean=float(np.mean([r['latency_ms'] for r in group])),
            datasets={d:float(np.mean([r['dice'] for r in group if r['dataset']==d])) for d in sorted({r['dataset'] for r in group})})
    return result


def run(tasks,method,output,gate=False):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time()
    try:
        contract=json.loads((tasks/'contract.json').read_text());catalog=Path(contract['catalog'])
        if digest(catalog/'tiles.parquet')!=contract['catalog_sha256'] or digest(tasks/'validation_maximum.json')!=contract['validation_sha256']:
            raise ValueError('protocol identity mismatch')
        metadata=pd.read_parquet(catalog/'tiles.parquet').to_dict('records');by_id={r['tile_id']:r for r in metadata}
        plans=json.loads((tasks/'validation_maximum.json').read_text())
        if gate:
            selected=[]
            for dataset in sorted({p['dataset'] for p in plans}):
                selected.extend([p for p in plans if p['dataset']==dataset and p['prompt_group']==0][:2])
            plans=selected
        config_path=Path('benchmarks/v4/baseline/configs')/('sam_zero_shot.yaml' if method=='sam' else 'wsi_sam_zero_shot.yaml')
        config=yaml.safe_load(config_path.read_text())
        if method=='sam':
            model=SAMAdapter(config);prior=model.provenance()
        else:
            for checkpoint in config['checkpoints'].values():
                if digest(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('checkpoint identity mismatch')
            model=SharedContextWSISAM(config);prior=model.load_report
        run_record=dict(method=method,stage='public',adaptation_steps=0,worker_pid=os.getpid(),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gate=gate,
            protocol_sha256=digest(tasks/'contract.json'),metric_version=VERSION,weights=prior,
            configuration=config,configuration_sha256=digest(config_path),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            sources={str(f):digest(f) for f in Path(__file__).parent.glob('*.py')},
            views=['fine'] if method=='sam' else ['fine','2x'],input_render_size=512,
            encoder_render_size=1024,GT_in_forward=False,scope='public prior local development baseline; not matched adaptation')
        write(output/'run.json',run_record);results=[];cached_id=None
        with (output/'rows.jsonl').open('x') as handle:
            for p in plans:
                if p['support_id']!=p['query_id'] or p['kind']!='local':raise ValueError('nonlocal task in local baseline')
                row=by_id[p['query_id']]
                if row['tile_id']!=cached_id:
                    variant=json.loads(row['variants_json'])['original']
                    if digest(variant['image'])!=variant['image_sha256']:raise ValueError('RGB identity changed')
                    fine=np.asarray(Image.open(variant['image']).convert('RGB')).copy()
                    gt=np.asarray(Image.open(row['labels']));import hashlib
                    if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('GT identity changed')
                    if method!='sam':
                        if digest(row['multiscale_cache'])!=row['multiscale_sha256']:raise ValueError('context identity changed')
                        with np.load(row['multiscale_cache'],allow_pickle=False) as cache:low=cache['image_1'].copy()
                    cached_id=row['tile_id']
                shape=fine.shape[:2];scale=np.array(shape[::-1],np.float32)
                for budget in contract['report_budgets']:
                    plan=nested(p,budget);pos=np.array(plan['positive_xy'],np.float32)*scale
                    neg=np.array(plan['negative_xy'],np.float32)*scale
                    xy=np.floor(np.concatenate((pos,neg))).astype(int)
                    if ((xy<0).any() or (xy[:,0]>=shape[1]).any() or (xy[:,1]>=shape[0]).any()):raise ValueError('prompt outside image')
                    if not (gt[xy[:len(pos),1],xy[:len(pos),0]]==p['target']).all():raise ValueError('positive sign mismatch')
                    if not ((gt[xy[len(pos):,1],xy[len(pos):,0]]!=p['target'])&(gt[xy[len(pos):,1],xy[len(pos):,0]]!=255)).all():raise ValueError('negative sign mismatch')
                    torch.cuda.synchronize();tick=time.perf_counter()
                    if method=='sam':
                        request=EpisodeRequest(str(p['number']),fine,pos,neg,'point');request.validate()
                        prediction=model.timed_predict(request);probability=prediction.probability;candidates=prediction.candidates
                    else:
                        prediction=model.predict(dict(fine=fine,low=low),pos,neg,'point',None)
                        probability=prediction['probability'];candidates=prediction['candidates']
                    torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                    if probability.shape!=gt.shape or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ValueError('invalid probability')
                    result=score(probability>=.5,gt==p['target'],gt!=255,json.loads(row['input_mpp_xy_json']))
                    result.update(plan,method=method,latency_ms=elapsed,candidates=candidates,space_key=row['space_key'],
                                  source_id=row['source_id'],spatial_group_id=row['spatial_group_id'])
                    results.append(result);handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                if len(results)%80==0:
                    write(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(results),total=len(plans)*4,seconds=time.time()-started))
                    print(json.dumps(dict(method=method,completed=len(results),total=len(plans)*4)),flush=True)
        report=dict(status='complete',worker_pid=os.getpid(),method=method,gate=gate,metrics=summarize(results),
            selection_group0=summarize([r for r in results if r['prompt_group']==0]),seconds=time.time()-started,
            metric_version=VERSION,rows_sha256=digest(output/'rows.jsonl'),
            scope='matched local tasks and points; different public priors/views; development only')
        write(output/'summary.json',report);write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows=len(results)))
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception:
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--method',choices=['sam','wsi_sam'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    run(a.tasks,a.method,a.output,a.gate)
