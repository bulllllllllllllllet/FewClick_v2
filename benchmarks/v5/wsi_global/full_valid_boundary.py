"""Physical BF1 for separately accepted support-inclusive full-valid WSIs.

The original strict-remote driver and metric kernel remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .completion import atomic_json
from .streaming_boundary import physical_boundary_counts
from benchmarks.v5.fair_comparison.metrics import boundary,boundary_score


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:raise ValueError(why)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):atomic_json(a.output/n,d)
    try:
        source=read(a.source/'summary.json');accepted=read(a.acceptance/'summary.json')
        fixture=source['status'] in ('passed_closed12_raster_production_fixture','passed_full_valid_prediction_driver_gate')
        require(fixture==a.gate,'explicit representative fixture versus formal full run')
        wanted=('accepted_closed12_full_valid_raster_fixture' if fixture else 'accepted_closed12_full_valid_native_GT_and_remote_invariance') if a.kind=='closed12' else ('accepted_full_valid_prediction_gate' if fixture else 'accepted_full_valid_saved_predictions_GT_and_remote_invariance_qualified')
        require(accepted['status']==wanted and accepted['source_summary_sha256']==sha(a.source/'summary.json'),'complete independent native prediction/GT source receipt')
        require(sha(a.source/'rows.json')==source['rows_sha256'],'all immutable source rows')
        wanted_scope='complete_valid4x_including_support' if a.kind=='closed12' else 'complete_valid4x_including_support_separate_diagnostic'
        require(source['signature']['evaluation_scope']==wanted_scope,'explicit accepted full-valid geometry, never a renamed remote source')
        if not fixture:require(source['rows']==(360 if a.kind=='closed12' else 11040 if source['signature']['method']=='fewclick_m3d_image' else 5520),'all original full-valid class/task rows')

        data=a.root/'main30_full_1027_171918'
        signature=dict(evaluation_scope='complete_valid4x_including_support_separate_diagnostic',source_sha256=sha(__file__),kernel_sha256=sha('benchmarks/v5/wsi_global/streaming_boundary.py'),frozen_metric_sha256=sha('benchmarks/v5/fair_comparison/metrics.py'),kind=a.kind,tolerance_um=2.,block_rows=256,GT_data_summary_sha256=sha(data/'summary.json'))
        core=read(a.root/'streaming_boundary_gate_1028_032500/summary.json');require(core['status']=='passed_actual_WSI_streaming_physical_BF1_gate' and core['source_sha256']==signature['kernel_sha256'] and core['frozen_metric_source_sha256']==signature['frozen_metric_sha256'],'actual64case EDT/KDTree physical kernel gate')
        if not fixture:
            require(a.verified_gate is not None,'actual same-kind BF1 production driver fixture required');g=read(a.verified_gate/'summary.json')
            require(g['status']=='passed_full_valid_boundary_driver_gate' and g['signature']==signature,'same driver/metric/data-domain fixture')
        require(0<=a.rank<a.shards,'valid disjoint row shard')
        records={r['wsi_id']:r for r in read(data/'slides_private.json')};all_rows=read(a.source/'rows.json')
        chosen=[(i,r) for i,r in enumerate(all_rows) if i%a.shards==a.rank]
        require(chosen and len(all_rows)==source['rows'],'complete original row identities')
        items={r['item']:r for r in read(a.source/'items.json')} if a.kind=='closed12' else {}
        rows=[];checked=set()
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=fixture,source_summary_sha256=sha(a.source/'summary.json'),acceptance_sha256=sha(a.acceptance/'summary.json'),rank=a.rank,shards=a.shards,all_rows=len(all_rows),assigned_rows=[i for i,r in chosen],scope='BF1 on original complete GT-cell grid of standard full-valid domain including source; separate diagnostic; ignore255; boundary at invalid/frame ignored; no prompt-component censor or per-distance BF1 in this stage'))
        for number,(index,row) in enumerate(chosen):
            r=records[row['wsi_id']];gtpath=data/r['wsi_id']/'labels.npy'
            if r['wsi_id'] not in checked:
                require(sha(gtpath)==r['files'][str(gtpath)]['sha256'],'actual accepted original GT bytes');checked.add(r['wsi_id'])
            labels=np.load(gtpath,mmap_mode='r');gh,gw=labels.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh;mpp=(sx*r['native_mpp'][0],sy*r['native_mpp'][1]);target=row['target']
            if a.kind=='closed12':
                item=items[row['item']];d=item['prediction_files']['class_prediction.npy'];require(sha(d['path'])==d['sha256'],'accepted native categorical bytes')
                pred=np.load(d['path'],mmap_mode='r');x0,y0,x1,y1=item['GT_domain']['GT_cell_bounds'];left,top,right,bottom=item['coverage']['bounds_native']
                xx=np.floor((np.arange(x0,x1)+.5)*sx).astype(np.int64)-left
                def read_rows(lo,hi):
                    truth=np.asarray(labels[y0+lo:y0+hi,x0:x1]);yy=np.floor((np.arange(y0+lo,y0+hi)+.5)*sy).astype(np.int64)-top
                    prediction=np.asarray(pred[np.ix_(yy,xx)])==target
                    return prediction,truth==target,truth!=255
            else:
                require(sha(row['prediction'])==row['prediction_sha256'],'accepted packed prediction metadata');d=read(row['prediction']);require(sha(d['mask_path'])==d['mask_sha256'],'accepted whole packed mask bytes')
                pred=np.load(d['mask_path'],mmap_mode='r');x0,y0,x1,y1=d['GT_cell_bounds'];require(d['GT_shape']==[y1-y0,x1-x0] and d['bitorder']=='big','original GT packed prediction layout')
                def read_rows(lo,hi):
                    truth=np.asarray(labels[y0+lo:y0+hi,x0:x1]);prediction=np.unpackbits(np.asarray(pred[lo:hi]),axis=1,count=x1-x0,bitorder='big').astype(bool)
                    return prediction,truth==target,truth!=255
            result=physical_boundary_counts(read_rows,(y1-y0,x1-x0),mpp,block_rows=256)
            if fixture:
                prediction,truth,valid=read_rows(0,y1-y0);ref=boundary_score(boundary(prediction,valid),boundary(truth,valid),mpp)
                require(all(result[k]==ref[k] for k in ('predicted_boundary_pixels','true_boundary_pixels','boundary_evaluable','bf1_um2')),'actual full driver versus frozen whole-domain EDT')
            blocks=result.pop('blocks');folder=a.output/f'row_{index:06d}';folder.mkdir();atomic_json(folder/'blocks.json',blocks)
            result.update(block_counts_path=str(folder/'blocks.json'),block_counts_sha256=sha(folder/'blocks.json'))
            identity={k:row[k] for k in ('method','variant','wsi_id','item','target','prompt_group','budget') if k in row}
            rows.append(dict(source_row_index=index,**identity,physical_BF1=result));dump('rows.json',rows)
            dump('status.json',dict(status='WSI_original_GT_physical_BF1',worker_pid=pid,completed=number+1,total=len(chosen),source_row_index=index,rank=a.rank,seconds=time.perf_counter()-started))
            del labels,pred
        result=dict(status='passed_full_valid_boundary_driver_gate' if fixture else 'complete_full_valid_boundary_shard_unaccepted',worker_pid=pid,signature=signature,rank=a.rank,shards=a.shards,source_rows=len(all_rows),rows=len(rows),boundary_evaluable=sum(r['physical_BF1']['boundary_evaluable'] for r in rows),GT_boundary_NA=sum(not r['physical_BF1']['boundary_evaluable'] for r in rows),rows_sha256=sha(a.output/'rows.json'),seconds=time.perf_counter()-started,
                    remaining='Independent full-valid BF1 recount/shard aggregation and separate paired report; closed12 privileged and source-inclusive domain not independent prompt tasks')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--kind',choices=('closed12','interactive'),required=True);p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    run(p.parse_args())
