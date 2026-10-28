"""Independent full-valid physical-disk boundary matches; no producer imports.

The dilation footprint is an anisotropic Euclidean disk, not the historical
square tolerance. Every saved BF1 row is recounted from original GT and the
accepted prediction, using 197-row bands and independent boundary extraction.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.ndimage import binary_dilation

from .completion import atomic_json


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:raise ValueError(why)


def edges(mask,valid):
    edge=np.zeros(mask.shape,bool);h,w=mask.shape
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        y0,y1=max(0,-dr),min(h,h-dr);x0,x1=max(0,-dc),min(w,w-dc)
        edge[y0:y1,x0:x1] |= valid[y0:y1,x0:x1]&valid[y0+dr:y1+dr,x0+dc:x1+dc]&(mask[y0:y1,x0:x1]!=mask[y0+dr:y1+dr,x0+dc:x1+dc])
    return edge


def disk_counts(read_rows,shape,mpp):
    h,w=shape;mx,my=mpp;radius=2.+1e-9
    require(min(h,w,mx,my)>0 and np.isfinite([mx,my]).all(),'positive physical boundary geometry')
    rx,ry=math.ceil(radius/mx),math.ceil(radius/my)
    dy,dx=np.mgrid[-ry:ry+1,-rx:rx+1]
    disk=np.hypot(dx*mx,dy*my)<=radius
    halo=ry+2;npred=ntrue=mp=mt=0
    for y in range(0,h,197):
        end=min(y+197,h);lo=max(0,y-halo);hi=min(h,end+halo)
        p,t,v=read_rows(lo,hi);p=np.asarray(p,bool);t=np.asarray(t,bool);v=np.asarray(v,bool)
        require(p.shape==t.shape==v.shape==(hi-lo,w),'independent callback physical footprint')
        pe=edges(p,v);te=edges(t,v);core=slice(y-lo,end-lo)
        pc=int(pe[core].sum());tc=int(te[core].sum());npred+=pc;ntrue+=tc
        if pc and te.any():mp+=int(np.count_nonzero(pe[core]&binary_dilation(te,structure=disk,border_value=0)[core]))
        if tc and pe.any():mt+=int(np.count_nonzero(te[core]&binary_dilation(pe,structure=disk,border_value=0)[core]))
    precision=mp/npred if npred else 0.;recall=mt/ntrue if ntrue else None
    bf1=None if not ntrue else 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(predicted_boundary_pixels=npred,true_boundary_pixels=ntrue,matched_predicted_boundary_pixels=mp,matched_true_boundary_pixels=mt,boundary_evaluable=ntrue>0,boundary_precision=precision,boundary_recall=recall,bf1_um2=bf1)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):atomic_json(a.output/n,d)
    try:
        source=read(a.source/'summary.json');contract=read(a.source/'contract.json');fixture=source['status']=='passed_full_valid_boundary_driver_gate'
        require(fixture or source['status']=='complete_full_valid_boundary_shard_unaccepted','completed BF1 driver/shard')
        signature=source['signature'];kind=signature['kind']
        require(signature['evaluation_scope']=='complete_valid4x_including_support_separate_diagnostic','separate full-valid BF1 scope')
        for key,path in [('source_sha256','benchmarks/v5/wsi_global/full_valid_boundary.py'),('kernel_sha256','benchmarks/v5/wsi_global/streaming_boundary.py'),('frozen_metric_sha256','benchmarks/v5/fair_comparison/metrics.py')]:require(signature[key]==sha(path),'unchanged producer/metric source')
        recipe=dict(source_sha256=sha(__file__),producer_signature=signature)
        if not fixture:
            require(a.verified_gate is not None,'actual independent physical disk BF1 fixture required');g=read(a.verified_gate/'summary.json')
            require(g['status']=='accepted_full_valid_boundary_gate' and g['recipe']==recipe,'same independent BF1 recipe')
        require(sha(a.predictions/'summary.json')==contract['source_summary_sha256'] and sha(a.prediction_acceptance/'summary.json')==contract['acceptance_sha256'],'accepted underlying native/packed prediction receipt')
        require(read(a.prediction_acceptance/'summary.json')['source_summary_sha256']==sha(a.predictions/'summary.json'),'independent prediction source binding')
        require(sha(a.source/'rows.json')==source['rows_sha256'],'all BF1 result bytes')
        prediction_summary=read(a.predictions/'summary.json');require(sha(a.predictions/'rows.json')==prediction_summary['rows_sha256'],'all original source task rows')
        expected_receipt=('accepted_closed12_full_valid_raster_fixture' if fixture else 'accepted_closed12_full_valid_native_GT_and_remote_invariance') if kind=='closed12' else ('accepted_full_valid_prediction_gate' if fixture else 'accepted_full_valid_saved_predictions_GT_and_remote_invariance_qualified')
        require(read(a.prediction_acceptance/'summary.json')['status']==expected_receipt,'independently accepted actual full-valid predictions')
        require(prediction_summary['signature']['evaluation_scope']==('complete_valid4x_including_support' if kind=='closed12' else signature['evaluation_scope']),'not a renamed strict-remote prediction')

        all_rows=read(a.predictions/'rows.json');rows=read(a.source/'rows.json');indices=[r['source_row_index'] for r in rows]
        require(indices==contract['assigned_rows']==[i for i in range(len(all_rows)) if i%source['shards']==source['rank']],'every disjoint assigned source row, no omissions/duplicates')
        records={r['wsi_id']:r for r in read(a.root/'main30_full_1027_171918/slides_private.json')}
        items={r['item']:r for r in read(a.predictions/'items.json')} if kind=='closed12' else {}
        checks=[];verified_gt=set()
        for number,row in enumerate(rows):
            original=all_rows[row['source_row_index']]
            require(all(row[k]==original[k] for k in ('method','variant','wsi_id','item','target','prompt_group','budget') if k in original),'all model/task identity columns')
            r=records[row['wsi_id']];gtpath=a.root/'main30_full_1027_171918'/r['wsi_id']/'labels.npy'
            if r['wsi_id'] not in verified_gt:require(sha(gtpath)==r['files'][str(gtpath)]['sha256'],'actual originalGT bytes');verified_gt.add(r['wsi_id'])
            gt=np.load(gtpath,mmap_mode='r');gh,gw=gt.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh;mpp=(sx*r['native_mpp'][0],sy*r['native_mpp'][1]);target=row['target']
            if kind=='closed12':
                item=items[row['item']];d=item['prediction_files']['class_prediction.npy'];require(sha(d['path'])==d['sha256'],'underlying native categorical prediction bytes')
                pred=np.load(d['path'],mmap_mode='r');x0,y0,x1,y1=item['GT_domain']['GT_cell_bounds'];left,top,right,bottom=item['coverage']['bounds_native']
                require([x0,y0,x1,y1]==[math.ceil(left/sx),math.ceil(top/sy),math.floor(right/sx),math.floor(bottom/sy)],'complete original GT physical cells')
                xx=np.array([math.floor((x+.5)*sx)-left for x in range(x0,x1)])
                def read_rows(lo,hi):
                    labels=np.asarray(gt[y0+lo:y0+hi,x0:x1]);yy=np.array([math.floor((y+.5)*sy)-top for y in range(y0+lo,y0+hi)])
                    return np.asarray(pred[np.ix_(yy,xx)])==target,labels==target,labels!=255
            else:
                require(sha(original['prediction'])==original['prediction_sha256'],'underlying packed geometry bytes');d=read(original['prediction']);require(sha(d['mask_path'])==d['mask_sha256'],'actual packed model mask bytes')
                pred=np.load(d['mask_path'],mmap_mode='r');x0,y0,x1,y1=d['GT_cell_bounds'];require(d['bitorder']=='big','packed mask bit order')
                def read_rows(lo,hi):
                    labels=np.asarray(gt[y0+lo:y0+hi,x0:x1]);guess=np.unpackbits(np.asarray(pred[lo:hi]),axis=1,count=x1-x0,bitorder='big').astype(bool)
                    return guess,labels==target,labels!=255
            result=disk_counts(read_rows,(y1-y0,x1-x0),mpp);actual=row['physical_BF1']
            require(actual['GT_mpp_xy']==list(mpp) and actual['tolerance_um']==2.,'actual originalGT physical spacing/tolerance')
            require(all(result[k]==actual[k] for k in result),'every independent boundary count/matched count/eligibility/precision/recall/BF1 scalar')
            require(sha(actual['block_counts_path'])==actual['block_counts_sha256'],'all producer block count bytes')
            blocks=read(actual['block_counts_path']);position=0
            for b in blocks:require(b['start_row']==position and b['rows']>0,'all producer BF1 blocks contiguous');position+=b['rows']
            require(position==y1-y0 and sum(b['predicted_boundary_pixels'] for b in blocks)==result['predicted_boundary_pixels'] and sum(b['true_boundary_pixels'] for b in blocks)==result['true_boundary_pixels'] and sum(b['matched_predicted'] for b in blocks)==result['matched_predicted_boundary_pixels'] and sum(b['matched_true'] for b in blocks)==result['matched_true_boundary_pixels'],'all raw block counters conserve entire domain')
            checks.append(dict(source_row_index=row['source_row_index'],**result));dump('status.json',dict(status='independent_WSI_physical_disk_BF1',worker_pid=pid,completed=number+1,total=len(rows)))
            del gt,pred
        dump('checks.json',checks)
        result=dict(status='accepted_full_valid_boundary_gate' if fixture else 'accepted_full_valid_boundary_shard',worker_pid=pid,recipe=recipe,rank=source['rank'],shards=source['shards'],source_rows=len(all_rows),rows=len(rows),source_summary_sha256=sha(a.source/'summary.json'),prediction_summary_sha256=sha(a.predictions/'summary.json'),checks_sha256=sha(a.output/'checks.json'),seconds=time.perf_counter()-started,
                    scope='All saved originalGT physical BF1 counts independently recomputed using4direction boundaries and anisotropic Euclidean-disk matching, no EDT/kernel/driver imported',remaining='Full-valid shard aggregation and separate paired reporting; fullcell XCell inputs/predictions still separate')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','predictions','prediction-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
