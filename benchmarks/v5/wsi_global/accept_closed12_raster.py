"""Fresh-process whole-native12 probability replay and scalar-coordinate GT audit."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4)
    source=read(a.source/'summary.json');fixture=source['status']=='passed_closed12_raster_production_fixture'
    require(fixture or source['status']=='complete_closed12_WSI_quality_execution_unaccepted','complete raster execution')
    for p,d in source['signature']['source_sha256'].items():require(sha(p)==d,'frozen producer code')
    require(sha(a.source/'items.json')==source['items_sha256'] and sha(a.source/'rows.json')==source['rows_sha256'],'all result/metric SHA')
    if not fixture:
        require(a.verified_fixture is not None,'actual independent audit fixture required');verified=read(a.verified_fixture/'summary.json');require(verified['status']=='accepted_closed12_raster_fixture' and verified['source_sha256']==sha(__file__),'same-source independent gate')
    items=read(a.source/'items.json');rows=read(a.source/'rows.json');data={r['wsi_id']:r for r in read(a.data/'slides_private.json')}
    require(len(items)==(2 if fixture else 30),'all requested raster items')
    if fixture:entries=read(a.semantic_gate/'input_cache_snapshot.json')
    else:
        cache=read(a.cache/'summary.json');require(sha(a.cache/'cache_manifest.jsonl')==cache['cache_manifest_sha256'],'complete encoded manifest SHA');entries=[json.loads(line) for line in (a.cache/'cache_manifest.jsonl').open()]
    for e in entries:require(sha(e['path'])==e['sha256'],'all original encoded bytes')
    @lru_cache(maxsize=128)
    def native_probability(path,w,h):
        z=torch.load(path,map_location='cpu',weights_only=False)['logits'];p=z.softmax(1)[0].numpy()
        return np.stack([np.asarray(Image.fromarray(v.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR)) for v in p],axis=2)
    evidence=[];total_pixels=0
    for number,item in enumerate(items):
        coverage=item['coverage'];left,top,right,bottom=coverage['bounds_native'];width=right-left;height=bottom-top
        for p in item['prediction_files'].values():require(sha(p['path'])==p['sha256'],'all actual native output bytes')
        folder=a.source/item['item'];pred=np.load(folder/'class_prediction.npy',mmap_mode='r');overlap=np.load(folder/'overlap.npy',mmap_mode='r');prob0=np.load(folder/'probability_class0.npy',mmap_mode='r');prob10=np.load(folder/'probability_class10.npy',mmap_mode='r')
        require(pred.shape==(height,width) and pred.dtype==np.uint8 and overlap.dtype==np.uint16,'actual output shape/dtype')
        group=[e for e in entries if e['identity']['wsi_id']==item['wsi_id'] and left<=e['identity']['native_box'][0] and e['identity']['native_box'][0]+e['identity']['native_box'][2]<=right and top<=e['identity']['native_box'][1] and e['identity']['native_box'][1]+e['identity']['native_box'][3]<=bottom]
        require(len(group)==coverage['tiles'],'independent tile inclusion')
        covered=weighted=0;maximum=0;next_y=top
        for block in coverage['blocks']:
            y=block['start_native_y'];n=block['rows'];require(y==next_y and n>0 and y+n<=bottom,'complete nonoverlapping probability blocks');next_y=y+n
            total=np.zeros((n,width,12),np.float64);count=np.zeros((n,width),np.uint16)
            for e in group:
                x,by,w,h=e['identity']['native_box'];lo=max(y,by);hi=min(y+n,by+h)
                if lo>=hi:continue
                values=native_probability(e['path'],w,h);total[lo-y:hi-y,x-left:x-left+w]+=values[lo-by:hi-by];count[lo-y:hi-y,x-left:x-left+w]+=1
            require((count>0).all(),'all native pixels covered');expected=total/count[:,:,None]
            require(hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()==block['sha256_float64_HWC'],'all12 exact native probability block')
            lo=y-top;hi=lo+n
            require(np.array_equal(pred[lo:hi],expected.argmax(2)) and np.array_equal(overlap[lo:hi],count),'all categorical/count pixels')
            require(np.array_equal(prob0[lo:hi],expected[:,:,0].astype(np.float32)) and np.array_equal(prob10[lo:hi],expected[:,:,10].astype(np.float32)),'all saved continuous probability pixels')
            covered+=n*width;weighted+=int(count.sum(dtype=np.uint64));maximum=max(maximum,int(count.max()))
            for array in (pred,overlap,prob0,prob10):array._mmap.madvise(__import__('mmap').MADV_DONTNEED)
            (a.output/'status.json').write_text(json.dumps(dict(status='independent_full_native_probability',worker_pid=pid,item=item['item'],block_y=y,items_complete=number,total_items=len(items)))+'\n')
        require(next_y==bottom and covered==coverage['pixels'] and weighted==coverage['weighted_pixels'] and maximum==coverage['maximum_overlap'],'independent complete coverage conservation')
        r=data[item['wsi_id']];w,h=r['native_wh'];gw,gh=r['gt_wh'];sx,sy=w/gw,h/gh
        gx=[x for x in range(gw) if x*sx>=left and (x+1)*sx<=right];gy=[y for y in range(gh) if y*sy>=top and (y+1)*sy<=bottom]
        require([gx[0],gy[0],gx[-1]+1,gy[-1]+1]==item['GT_domain']['GT_cell_bounds'],'independent whole-cell physical GT domain')
        labels=np.load(a.data/item['wsi_id']/'labels.npy',mmap_mode='r');xx=np.array([math.floor((x+.5)*sx)-left for x in gx],int);matrix=np.zeros((12,12),np.int64);unknown=0
        for y in gy:
            truth=np.asarray(labels[y,gx[0]:gx[-1]+1]);guess=np.asarray(pred[math.floor((y+.5)*sy)-top,xx]);valid=truth!=255
            matrix+=np.bincount(truth[valid].astype(int)*12+guess[valid].astype(int),minlength=144).reshape(12,12);unknown+=int((~valid).sum())
        require(matrix.tolist()==item['confusion'] and unknown==item['GT_domain']['unknown'],'all independently scored GT confusion pixels')
        require(np.array_equal(np.load(folder/'confusion.npy'),matrix),'saved confusion tensor')
        for row in [r for r in rows if r['item']==item['item']]:
            c=row['target'];tp=int(matrix[c,c]);fp=int(matrix[:,c].sum())-tp;fn=int(matrix[c].sum())-tp;truth=tp+fn;valid=int(matrix.sum())
            require((row['tp'],row['fp'],row['fn'],row['positive'],row['valid'])==(tp,fp,fn,truth,valid),'all class denominators')
            expected=2*tp/(2*tp+fp+fn) if truth else None;require(row['dice']==expected,'every Dice scalar')
            require(row['iou']==(tp/(tp+fp+fn) if truth else None) and row['recall']==(tp/truth if truth else None) and row['precision']==(tp/(tp+fp) if tp+fp else 0.) and row['false_positive_fraction']==fp/valid,'all remaining class metrics')
        evidence.append(dict(item=item['item'],pixels=covered,GT_pixels=int(matrix.sum()),maximum_overlap=maximum));total_pixels+=covered;native_probability.cache_clear();del labels,pred,overlap,prob0,prob10
        print(json.dumps(evidence[-1]),flush=True)
    result=dict(status='accepted_closed12_raster_fixture' if fixture else 'accepted_closed12_full30_native_raster_and_GT_scores',worker_pid=pid,items=len(items),pixels=total_pixels,evidence=evidence,source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-start,
        scope='Every complete12channel native block digest, category/count/selected probabilities, all GT physical-cell indices/confusion/class metrics; no producer stripe/scorer imported',remaining='Statistical report/matched276 versus all330+background, full-domain extension and interactive/whole-cell methods/components/distance/BF1')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],pixels=total_pixels,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','cache','protocol','data','semantic-gate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-fixture',type=Path);run(p.parse_args())
