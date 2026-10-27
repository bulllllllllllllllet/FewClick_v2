"""Real central and last-overlap12class WSI probability/scoring gate."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .semantic_stripe import SemanticStripe


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(test,message):
    if not test:raise ValueError(message)


def confusion(pred,labels,bounds,scale):
    left,top,right,bottom=bounds;sx,sy=scale
    x0,y0,x1,y1=math.ceil(left/sx),math.ceil(top/sy),math.floor(right/sx),math.floor(bottom/sy)
    xs=np.floor((np.arange(x0,x1)+.5)*sx).astype(int)-left
    matrix=np.zeros((12,12),np.int64);unknown=0
    for y in range(y0,y1,193):
        end=min(y+193,y1);ys=np.floor((np.arange(y,end)+.5)*sy).astype(int)-top
        truth=np.asarray(labels[y:end,x0:x1]);guess=np.asarray(pred[np.ix_(ys,xs)]);valid=truth!=255
        check((guess<12).all(),'valid categorical output')
        matrix+=np.bincount(truth[valid].astype(int)*12+guess[valid].astype(int),minlength=144).reshape(12,12);unknown+=int((~valid).sum())
    check(int(matrix.sum())+unknown==(x1-x0)*(y1-y0),'GT pixel conservation')
    return matrix,dict(GT_cell_bounds=[x0,y0,x1,y1],unknown=unknown)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4);pid=os.getpid()
    protocol=read(a.protocol/'query_inputs.json')[0];data=next(r for r in read(a.data/'slides_private.json') if r['wsi_id']==protocol['wsi_id']);contract=read(a.cache/'contract.json')
    check(contract['method']=='closed_reference' and contract['signature']['protocol_sha256']==sha(a.protocol/'summary.json'),'accepted source cache method/protocol')
    tiles=[json.loads(line) for line in Path(protocol['grid']).open()];xs=sorted({t['native_boxes'][0][0] for t in tiles});ys=sorted({t['native_boxes'][0][1] for t in tiles})
    # Read only newline-committed, immutable cache entries. Full job acceptance is not claimed.
    entries=[json.loads(line) for line in (a.cache/'cache_manifest.jsonl').read_text().splitlines(keepends=True) if line.endswith('\n')]
    bykey={(e['identity']['wsi_id'],e['identity']['tile']):e for e in entries};results=[];snapshots=[]
    for name,(ix,iy) in [('central',(len(xs)//2,len(ys)//2)),('last_overlap',(len(xs)-2,len(ys)-2))]:
        selected=[t for t in tiles if t['native_boxes'][0][0] in xs[ix:ix+2] and t['native_boxes'][0][1] in ys[iy:iy+2]];check(len(selected)==4,'fixed2x2 gate')
        boxes=[t['native_boxes'][0] for t in selected];bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)];fw,fh=boxes[0][2:]
        writer=SemanticStripe(a.output/name,bounds,[fw,fh]);probabilities=[]
        for t in selected:
            e=bykey[protocol['wsi_id'],t['number']];check(sha(e['path'])==e['sha256'],'actual committed cache SHA');payload=torch.load(e['path'],map_location='cpu',weights_only=False)
            check(payload['identity']==e['identity'] and e['identity']['native_box']==t['native_boxes'][0] and payload['checkpoint_sha256']==contract['signature']['weight_sha256'],'actual12class cache identity')
            p=payload['logits'].softmax(1)[0].numpy();writer.push(p,t['native_boxes'][0]);probabilities.append(p.copy());snapshots.append(e)
        coverage=writer.finish();(a.output/name/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
        # Independent full-canvas accumulation: no stripe buffer, no incremental flush implementation.
        left,top,right,bottom=bounds;total=np.zeros((bottom-top,right-left,12),np.float64);count=np.zeros((bottom-top,right-left),np.uint16)
        for p,(x,y,w,h) in zip(probabilities,boxes):
            images=np.stack([np.asarray(Image.fromarray(v.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR)) for v in p],axis=2)
            total[y-top:y-top+h,x-left:x-left+w]+=images;count[y-top:y-top+h,x-left:x-left+w]+=1
        check((count>0).all(),'independent native coverage');expected=total/count[:,:,None];pred=np.load(a.output/name/'class_prediction.npy',mmap_mode='r')
        check(np.array_equal(pred,expected.argmax(2)) and np.array_equal(np.load(a.output/name/'overlap.npy'),count),'all categorical/count pixels')
        for c in (0,10):check(np.array_equal(np.load(a.output/name/f'probability_class{c}.npy'),expected[:,:,c].astype(np.float32)),'all selected probability pixels')
        for block in coverage['blocks']:
            y=block['start_native_y']-top;values=np.ascontiguousarray(expected[y:y+block['rows']]);check(hashlib.sha256(values.tobytes()).hexdigest()==block['sha256_float64_HWC'],'every complete12channel probability block')
        # The first query-label access happens after all prediction files are finalized.
        gt=np.load(a.data/protocol['wsi_id']/'labels.npy',mmap_mode='r');gw,gh=data['gt_wh'];w,h=data['native_wh'];sx,sy=w/gw,h/gh
        matrix,domain=confusion(pred,gt,bounds,(sx,sy));x0,y0,x1,y1=domain['GT_cell_bounds'];scalar=Counter()
        for gy in range(y0,y1):
            py=math.floor((gy+.5)*sy)-top
            for gx in range(x0,x1):
                truth=int(gt[gy,gx])
                if truth!=255:scalar[truth,int(pred[py,math.floor((gx+.5)*sx)-left])]+=1
        independent=np.array([[scalar[t,p] for p in range(12)] for t in range(12)],np.int64);check(np.array_equal(matrix,independent),'independent scalar GT/categorical confusion')
        result=dict(window=name,tiles=[t['number'] for t in selected],pixels=coverage['pixels'],maximum_overlap=coverage['maximum_overlap'],GT_domain=domain,confusion=matrix.tolist(),all12_probability_block_SHA_exact=True)
        results.append(result);print(json.dumps(dict(window=name,pixels=result['pixels'],overlap=result['maximum_overlap'],scored_GT_pixels=int(matrix.sum()))),flush=True)
    (a.output/'input_cache_snapshot.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    result=dict(status='passed_native12_probability_stitch_and_GT_scoring_gate',worker_pid=pid,results=results,source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/semantic_stripe.py')]},input_snapshot_sha256=sha(a.output/'input_cache_snapshot.json'),seconds=time.perf_counter()-started,
        scope='8 actual committed tiles, fixed central/last-overlap windows; exact all12 native probability/hash/class/count and scalar originalGT confusion; not full30 accuracy',remaining='Complete cache acceptance then full native raster/quality/independent full replay, GT components/distance/BF1')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','protocol','data','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
