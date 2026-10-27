"""Independent saved instance count, physical geometry and full74 feature audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def geometric_features(inst,mpp,reg):
    require(inst.ndim==2 and inst.dtype.kind in 'iu' and int(inst.min())>=0,'valid saved native instances')
    maximum=int(inst.max());areas=np.zeros(maximum+1,np.int64);rows=np.zeros(maximum+1,np.float64);cols=np.zeros(maximum+1,np.float64);h,w=inst.shape
    for y in range(0,h,128):
        end=min(y+128,h);ids=inst[y:end].reshape(-1).astype(np.int64,copy=False)
        areas+=np.bincount(ids,minlength=maximum+1)
        rows+=np.bincount(ids,weights=np.repeat(np.arange(y,end,dtype=np.float64),w),minlength=maximum+1)
        cols+=np.bincount(ids,weights=np.tile(np.arange(w,dtype=np.float64),end-y),minlength=maximum+1)
    ids=np.flatnonzero(areas);ids=ids[ids!=0];kept=ids[:255]
    yx=np.column_stack((rows[kept]/areas[kept],cols[kept]/areas[kept]));order=np.lexsort((yx[:,1],yx[:,0])) if len(kept) else np.empty(0,dtype=int);yx=yx[order];kept=kept[order]
    expected=np.zeros((len(kept),74),np.float32)
    if len(kept):
        expected[:,0]=yx[:,1]/w;expected[:,1]=yx[:,0]/h;expected[:,2]=np.log1p(areas[kept]*mpp[0]*mpp[1]/(.25*.25));expected[:,3]=1
    require(reg.shape==(len(kept),64) and np.isfinite(reg).all(),'64D actual reg shape/finite');expected[:,10:]=reg
    return expected,len(ids)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();started=time.perf_counter()
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        producer=read(a.source/'contract.json');signature=dict(source_sha256=sha(__file__),producer_contract_sha256=sha(a.source/'contract.json'))
        for p,d in producer['signature']['source_sha256'].items():require(sha(p)==d,'producer source identity')
        entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').read_text().splitlines(keepends=True) if line.endswith('\n')]
        if a.gate:
            chosen={0,1525}
            for scale in (0,1,2):
                zero=next((e['tile'] for e in entries if any(c['scale']==scale and c['details']['total_cell_count']==0 for c in e['identity']['cell_records'])),None)
                if zero is not None:chosen.add(zero)
            entries=[e for e in entries if e['tile'] in chosen];require({e['tile'] for e in entries}==chosen,'fixed spatial/actual-zero completed entries available')
        else:
            done=read(a.source/'summary.json');complete=read(a.source/'cache_complete.json')
            require(done['status']=='complete_fullcell_GPU_WSI_execution' and done['tiles']==3050 and Path(str(a.source)+'.log').read_text().rstrip().endswith('EXIT_CODE=0'),'full producer finished; do not interfere timing')
            require(len(entries)==3050 and sha(a.source/'cache_manifest.jsonl')==complete['manifest_sha256'],'complete native source manifest')
            require(a.verified_gate is not None,'actual fullcell input geometry gate required');gate=read(a.verified_gate/'summary.json');require(gate['status']=='passed_actual_fullcell_input_gate' and gate['signature']==signature,'same-source independent geometry gate')
            require(0<=a.rank<a.shards,'valid audit shard');entries=[e for e in entries if e['tile']%a.shards==a.rank]
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,rank=a.rank,shards=a.shards,tiles=[e['tile'] for e in entries],scope='saved actual native cells, not detection rerun or unique-WSI nuclei count'))
        results=[];zero=[];retained=total=0
        for number,e in enumerate(entries):
            require(len(e['identity']['cell_records'])==3,'all3actual scales')
            checks=[]
            for c,b in zip(e['identity']['cell_records'],e['identity']['geometry']['native_boxes']):
                require(sha(c['path'])==c['sha256'],'every actual cell NPZ byte')
                with np.load(c['path'],allow_pickle=False) as z:
                    inst=z['instances'];reg=z['reg'];features=z['features'];actual=int(z['total']);d=c['details'];mx,my=d['detection_mpp_xy']
                    require(d['detection_size']==[inst.shape[1],inst.shape[0]],'actual detection shape')
                    native_mpp=e['identity']['geometry']['native_mpp'];require(math.isclose(mx,b[2]*native_mpp[0]/inst.shape[1],rel_tol=1e-12) and math.isclose(my,b[3]*native_mpp[1]/inst.shape[0],rel_tol=1e-12),'independent physical detector pitch')
                    expected,count=geometric_features(inst,[mx,my],reg)
                    require(count==actual==d['total_cell_count'] and len(expected)==d['retained_cell_count']==min(count,255),'true versus retained nuclear counts')
                    require(features.dtype==np.float32 and np.isfinite(features).all() and np.array_equal(features,expected),'all actual74 dimensions, centroid/area/unknown/reg exact')
                    if count==0:zero.append(dict(tile=e['tile'],scale=c['scale']))
                    total+=count;retained+=len(expected);checks.append(dict(scale=c['scale'],total=count,retained=len(expected),features_exact=True,shape=list(inst.shape)))
            results.append(dict(tile=e['tile'],views=checks))
            if number%10==0 or a.gate:dump('status.json',dict(status='independent_instance_geometry',worker_pid=pid,completed=number+1,total=len(entries),rank=a.rank))
        dump('entries.json',results);dump('input_snapshot.json',entries)
        result=dict(status='passed_actual_fullcell_input_gate' if a.gate else 'accepted_fullcell_input_shard',worker_pid=pid,signature=signature,rank=a.rank,shards=a.shards,tiles=len(entries),views=3*len(entries),sum_true_instances_across_views=total,sum_retained_across_views=retained,zero_cell_cases=zero,entries_sha256=sha(a.output/'entries.json'),input_snapshot_sha256=sha(a.output/'input_snapshot.json'),seconds=time.perf_counter()-started,
            remaining='Fresh encoder/zero-count density/two-budget decode and whole-output probability acceptance separate; no TLS accuracy or detection-history invariance claim')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],tiles=len(entries),views=result['views'],zero_cases=zero,seconds=result['seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);run(p.parse_args())
