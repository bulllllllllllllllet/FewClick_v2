"""Fresh independent PIL/native-point replay of sampled interactive probabilities."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:
        raise ValueError(message)


def replay(case,records,native_wh,gt_wh):
    left,top,right,bottom=case['native_coverage']['bounds_level0']
    w,h=native_wh;gw,gh=gt_wh;sx,sy=w/gw,h/gh
    gx0,gy0,gx1,gy1=math.ceil(left/sx),math.ceil(top/sy),math.floor(right/sx),math.floor(bottom/sy)
    require([gx0,gy0,gx1,gy1]==case['sampled_coverage']['GT_cell_bounds'],'independent complete-cell indices')
    x=np.array([math.floor((i+.5)*sx) for i in range(gx0,gx1)],np.int64)
    y=np.array([math.floor((i+.5)*sy) for i in range(gy0,gy1)],np.int64)
    maps={}
    for name,files in case['files'].items():
        for d in files.values():
            require(sha(d['path'])==d['sha256'],'every saved native/sampled output byte')
        maps[name]={k:np.load(d['path'],mmap_mode='r') for k,d in files.items()}
    tiles=[]
    for d in records:
        require(sha(d['path'])==d['sha256'],'all fixed512 decoder probabilities')
        p=np.load(d['path'],allow_pickle=False)
        require(p.dtype==np.float32 and p.ndim==2 and np.isfinite(p).all() and p.min()>=0 and p.max()<=1,'actual per-tile probability format')
        bx,by,bw,bh=d['native_box']
        tiles.append((d['native_box'],np.asarray(Image.fromarray(p).resize((bw,bh),Image.Resampling.BILINEAR))))
    maximum=weighted=0
    for offset in range(0,len(y),173):
        yy=y[offset:offset+173];total=np.zeros((len(yy),len(x)),np.float32);count=np.zeros_like(total,dtype=np.uint16)
        for (bx,by,bw,bh),native in tiles:
            rows=np.flatnonzero((yy>=by)&(yy<by+bh));cols=np.flatnonzero((x>=bx)&(x<bx+bw))
            if len(rows) and len(cols):
                idx=np.ix_(rows,cols);total[idx]+=native[np.ix_(yy[rows]-by,x[cols]-bx)];count[idx]+=1
        require(np.all(count>0),'independent complete sample coverage')
        expected=total/count;binary=expected>=.5
        for name,values in maps.items():
            if name=='sampled':
                actual={k:np.asarray(v[offset:offset+len(yy)]) for k,v in values.items()}
            else:
                actual={k:np.asarray(v[np.ix_(yy-top,x-left)]) for k,v in values.items()}
            require(np.array_equal(actual['probability.npy'],expected),'all probabilities equal independent native PIL/FP32 overlap mean')
            require(np.array_equal(actual['mask.npy'],binary) and np.array_equal(actual['overlap.npy'],count),'all threshold/overlap pixels exact')
        weighted+=int(count.sum(dtype=np.uint64));maximum=max(maximum,int(count.max()))
    require(weighted==case['sampled_coverage']['weighted_GT_cells'] and maximum==case['sampled_coverage']['maximum_overlap'],'all overlap counters conserved')
    return dict(GT_cells=len(x)*len(y),maximum_overlap=maximum,probability_error=0.,mask_disagreements=0)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);started=time.perf_counter();pid=os.getpid()
    source=read(a.source/'summary.json')
    require(source['status']=='passed_actual_interactive_native_GT_projection_gate','actual complete three-method native projection gate')
    require(sha(a.source/'results.json')==source['results_sha256'] and sha(a.source/'outputs.json')==source['outputs_sha256'],'all native gate result/input records bound')
    for p,h in source['sources'].items():
        require(sha(p)==h,'native/sampled producer sources frozen')
    results=read(a.source/'results.json');outputs=read(a.source/'outputs.json')
    require(len(results)==64 and len(outputs)==256,'all64 tasks and256 probabilities')
    def key(d):
        return tuple(d[k] for k in ('method','window','target','budget','variant'))
    require(len({key(r) for r in results})==64,'all method/window/target/budget/variant identities unique')
    identities=read(a.root/'main30_full_1027_171918/slides_private.json');r=identities[0];evidence=[]
    for number,result in enumerate(results):
        records=[d for d in outputs if key(d)==key(result)]
        require(len(records)==4 and result['online_encoder_calls']==0,'all four native contributions and no online encoder')
        check=replay(result,records,r['native_wh'],r['gt_wh']);check.update(identity=key(result));evidence.append(check)
        (a.output/'status.json').write_text(json.dumps(dict(status='independent_PIL_native_point_replay',worker_pid=pid,completed=number+1,total=64))+'\n')
    fixture=a.source/'anisotropic_fixture';synthetic=read(a.source/'synthetic.json');records=[]
    for i,(y,x) in enumerate((y,x) for y in (5,13,19) for x in (5,13,21,24)):
        p=fixture/f'tile{i:02d}.npy';records.append(dict(path=str(p),sha256=sha(p),native_box=[x,y,8,8]))
    synthetic_evidence=replay(synthetic,records,[37,29],[19,14])
    result=dict(status='accepted_actual_interactive_native_GT_projection',worker_pid=pid,cases=64,GT_cells=sum(r['GT_cells'] for r in evidence),evidence=evidence,synthetic=synthetic_evidence,source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-started,
                scope='Independent PIL reconstruction at every complete original GT-cell centre, actual three-model cached decoder outputs and anisotropic noninteger fixture; exact native FP32 mean/mask/count. No GT labels or new encoder parity claims',remaining='Full30 interactive decoder/scoring pipeline, GPU execution, BF1 and full-cell quality')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','root','output'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
