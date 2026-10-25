"""External scoring of development-selected adapted weights; no external fitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import pyvips
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .prepare import digest,write
from .evaluate_public import inputs,supervision
from .cell_data import assemble
from benchmarks.v5.fair_comparison.adapted_models import SAMFineTune,build
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.metrics import score,VERSION
from benchmarks.v5.fair_comparison.public_baselines import summarize
from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor

METHODS=('sam_ft','wsi_sam_ft','fewclick_m3d_image','fewclick_m3d_xcell')


class DeviceSAM(SAMFineTune):
    """Same differentiable bridge with an explicit construction device."""
    def __init__(self,device):
        nn.Module.__init__(self)
        from segment_anything.utils.transforms import ResizeLongestSide
        config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_zero_shot.yaml').read_text())
        config['device']=str(device)
        adapter=SAMAdapter(config);self.sam=adapter.model;self.prior=adapter.provenance()
        self.transform=ResizeLongestSide(self.sam.image_encoder.img_size)
        self.joint=False;self.set_stage(False)


def raw_inputs(row,model,device):
    if model.cells:return assemble(row,device)
    data=inputs(row,model.contexts);raw={}
    for scale,key in enumerate(('fine','low') if model.contexts else ('fine',)):
        suffix='' if scale==0 else f'_{scale}'
        raw['rgb'+suffix]=torch.from_numpy(data[key]).permute(2,0,1)[None].to(device)
        raw['image'+suffix]=he_to_tensor(data[key])[None].to(device)
    return raw


def prompts(plan,device):
    result={}
    for sign in ('positive','negative'):
        xy=np.asarray(plan[sign+'_xy'],np.float32)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or (xy<=0).any() or (xy>=1).any():
            raise ValueError('invalid external points')
        result[sign+'_xy']=torch.from_numpy(xy)[None].to(device)
        result[sign+'_mask']=torch.ones(1,len(xy),dtype=torch.bool,device=device)
    result['prompt_size_id']=torch.ones(1,dtype=torch.long,device=device)
    return result


def load(method,checkpoint,device,protocol_sha):
    model=DeviceSAM(device) if method=='sam_ft' else build(method,device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['contract']['method']!=method or saved['contract']['protocol_sha256']!=protocol_sha:
        raise ValueError('adapted checkpoint method/development contract mismatch')
    model.load_state_dict(saved['model'],strict=True)
    # Verify actual on-device tensors after strict load, including public backbone.
    for key,value in model.state_dict().items():
        if not torch.equal(value.cpu(),saved['model'][key].cpu()):raise ValueError('loaded tensor differs '+key)
    del saved;model.eval()
    return model


def run(tasks,development_tasks,checkpoint,method,output,device,gate=False,verified_gate=None,training=None,prepared=None):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time()
    source={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('cell_data.py'),Path(__file__).with_name('evaluate_public.py')]}
    try:
        contract=json.loads((tasks/'contract.json').read_text())
        if digest(tasks/'fields.parquet')!=contract['fields_sha256'] or digest(tasks/'local_maximum.json')!=contract['clicks_sha256']:
            raise ValueError('external fields/points changed')
        checkpoint_sha=digest(checkpoint)
        if training is not None:
            result=json.loads((training/'summary.json').read_text());selected=json.loads((training/'best_checkpoint.json').read_text())
            if result['status']!='complete' or result['step']!=10000 or not result['full_manifest_parity']:
                raise ValueError('formal development training incomplete')
            if selected['checkpoint_sha256']!=checkpoint_sha or Path(selected['path']).resolve()!=checkpoint.resolve():
                raise ValueError('checkpoint was not selected by development protocol')
            expected='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe'
            if result['training_consumed_sha256']!=expected or digest(training/'training_consumed.jsonl')!=expected:
                raise ValueError('formal adapted supervision differs')
        elif not gate:raise ValueError('full score requires completed development training evidence')
        if not gate:
            if verified_gate is None:raise ValueError('same-checkpoint real gate required')
            prior=json.loads((verified_gate/'summary.json').read_text());meta=json.loads((verified_gate/'run.json').read_text())
            if prior['status']!='complete' or not prior['gate'] or meta['checkpoint_sha256']!=checkpoint_sha or meta['method']!=method:
                raise ValueError('formal adapted checkpoint gate missing')
            if meta['source_sha256']!=source or meta['protocol_sha256']!=digest(tasks/'contract.json') or meta['device']!=str(device):
                raise ValueError('gate code/protocol/device changed')
        rows=pd.read_parquet(tasks/'fields.parquet').to_dict('records');by_id={r['tile_id']:r for r in rows}
        if prepared is not None:
            completion=json.loads((prepared/'summary.json').read_text())
            if completion['status']!='complete' or completion['gate'] or completion['fields']!=len(rows):raise ValueError('external cells incomplete')
            if digest(prepared/'fields_with_cells.parquet')!=completion['fields_sha256']:raise ValueError('prepared cells table changed')
            prepared_rows=pd.read_parquet(prepared/'fields_with_cells.parquet').to_dict('records')
            if set(by_id)!={r['tile_id'] for r in prepared_rows}:raise ValueError('external cells field IDs differ')
            for r in prepared_rows:
                original=by_id[r['tile_id']]
                for key in ('images_json','labels_array_sha256','source_id','multiscale_json'):
                    if r[key]!=original[key]:raise ValueError('prepared external geometry/GT identity changed')
            by_id={r['tile_id']:r for r in prepared_rows}
        model=load(method,checkpoint,device,digest(development_tasks/'contract.json'))
        if model.cells and prepared is None:raise ValueError('FewClick requires real native cell preparation')
        plans=json.loads((tasks/'local_maximum.json').read_text())
        if gate:
            selected=[];seen=set()
            for p in plans:
                if p['query_id'] not in seen:selected.append(p);seen.add(p['query_id'])
                if len(selected)==2:break
            plans=selected
        prior=dict(model.prior)
        if method=='fewclick_m3d_xcell':prior.update(XCell_overlap='user-confirmed internal training; current CRC/TIGER not used',provenance_overlay='2026-10-25 user clarification')
        write(output/'run.json',dict(method=method,worker_pid=os.getpid(),gate=gate,device=str(device),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            development_training=str(training) if training else None,protocol_sha256=digest(tasks/'contract.json'),
            development_protocol_sha256=digest(development_tasks/'contract.json'),source_sha256=source,initialization=prior,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),metric_version=VERSION,
            precision='bfloat16 autocast' if device.type=='cuda' else 'float32 CPU',external_fitting=False,GT_in_forward=False))
        records=[]
        with (output/'rows.jsonl').open('x') as stream,torch.inference_mode():
            for maximum in plans:
                if maximum['support_id']!=maximum['query_id'] or maximum['kind']!='local':raise ValueError('nonlocal external task')
                row=by_id[maximum['query_id']]
                input_row=row|dict(labels='/deliberately_unavailable/query_GT.png') if gate else row
                raw=raw_inputs(input_row,model,device)
                for budget in contract['report_budgets']:
                    plan=nested(maximum,budget);clicks=prompts(plan,device)
                    if device.type=='cuda':torch.cuda.synchronize()
                    tick=time.perf_counter()
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        # Deliberately no dataset, class ID or supervision argument.
                        result=model(raw,clicks)
                    if device.type=='cuda':torch.cuda.synchronize()
                    elapsed=1000*(time.perf_counter()-tick)
                    probability=result['pixel_probability'][0].float().cpu().numpy()
                    if probability.shape!=(512,512) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                        raise ValueError('invalid external adapted prediction')
                    truth,valid=supervision(row,plan['target'])
                    metric=score(probability>=.5,truth,valid,json.loads(row['input_mpp_xy_json']))
                    if gate:
                        path=output/f'prediction_{plan["pair_id"]}_{budget}.npy';np.save(path,probability)
                        pred=np.load(path,allow_pickle=False)>=.5
                        actual=dict(tp=int((pred&truth&valid).sum()),fp=int((pred&~truth&valid).sum()),fn=int((~pred&truth&valid).sum()))
                        if any(metric[k]!=v for k,v in actual.items()):raise ValueError('independent adapted count replay failed')
                    metric.update(plan,method=method,latency_ms=elapsed,source_id=row['source_id'],space_key=row['source_id'],spatial_group_id=row['source_id'])
                    records.append(metric);stream.write(json.dumps(metric,allow_nan=False)+'\n');stream.flush()
                write(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(records),total=len(plans)*4))
                print(json.dumps(dict(method=method,completed=len(records),total=len(plans)*4)),flush=True)
        write(output/'summary.json',dict(status='complete',gate=gate,method=method,worker_pid=os.getpid(),rows=len(records),
            rows_sha256=digest(output/'rows.jsonl'),metrics=summarize(records),GT_unavailable_forward_passed=gate,
            saved_direct_count_replay=gate,seconds=time.time()-started,
            scope='External native-centre ROI subset; development-selected adapted weights; no external fitting'))
        write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows=len(records)))
    except Exception:
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--development-tasks',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gate',action='store_true')
    p.add_argument('--verified-gate',type=Path);p.add_argument('--training',type=Path);p.add_argument('--prepared',type=Path)
    a=p.parse_args();run(a.tasks,a.development_tasks,a.checkpoint,a.method,a.output,torch.device(a.device),a.gate,a.verified_gate,a.training,a.prepared)
