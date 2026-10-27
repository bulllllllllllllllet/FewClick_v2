"""Independent label-count, sampled palette and grid acceptance for main CRC30."""
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import time
import traceback

import numpy as np
import yaml


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def check(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    def dump(name,d): (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    try:
        source=read(a.source/'summary.json');contract=read(a.source/'contract.json');rows=read(a.source/'slides_private.json')
        check(source['status'] in ('passed_main30_source_gate','complete_main30_full_GT_and_geometry_audit_qualified'),'finished source')
        check(source['slides']==len(rows)==(2 if source['gate'] else 30),'complete source cohort')
        check(sha(a.source/'slides_private.json')==source['slides_sha256'],'source results SHA')
        for p,d in source['input_sha256'].items():check(sha(p)==d,'frozen original inputs')
        classes=yaml.safe_load(Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml').read_text())['data']['class_map']
        color_to_label={tuple(c['rgb']):c['id'] for c in classes};totals=np.zeros(12,dtype=np.int64);evidence=[]
        for i,r in enumerate(rows):
            for p,d in r['files'].items():check(Path(p).stat().st_size==d['bytes'] and sha(p)==d['sha256'],'every source/output file SHA')
            w,h=r['native_wh'];gw,gh=r['gt_wh'];sx,sy=w/gw,h/gh;mx,my=r['native_mpp'];fw,fh=round(256/mx),round(256/my)
            marginx,marginy=math.ceil(3*fw/2),math.ceil(3*fh/2)
            bounds=[marginx,marginy,w-marginx,h-marginy]
            check(bounds==r['geometry']['valid_fine_bounds_xyxy'],'independent native-domain formula')
            tiles=[json.loads(line) for line in Path(r['grid']).open()];pairs=set();xs=set();ys=set()
            for t in tiles:
                x,y,bw,bh=t['native_boxes'][0];check((bw,bh)==(fw,fh),'fine shape');check((x,y) not in pairs,'unique tile');pairs.add((x,y));xs.add(x);ys.add(y)
                for k,b in zip((1,2,4),t['native_boxes']):
                    bx,by,bw,bh=b;check((bw,bh)==(fw*k,fh*k) and min(bx,by)>=0 and bx+bw<=w and by+bh<=h,'all true-context boxes')
                    check(max(abs(bx+bw/2-t['centre_level0'][0]),abs(by+bh/2-t['centre_level0'][1]))<=.5,'all centres')
            check(len(pairs)==len(xs)*len(ys)==r['geometry']['tiles'],'Cartesian grid')
            check(min(xs)==bounds[0] and max(xs)+fw==bounds[2] and min(ys)==bounds[1] and max(ys)+fh==bounds[3],'native coverage extent')
            check(all(b-a<=fw for a,b in zip(sorted(xs),sorted(xs)[1:])) and all(b-a<=fh for a,b in zip(sorted(ys),sorted(ys)[1:])),'no internal coverage holes')
            gx=np.flatnonzero((np.arange(gw)*sx>=bounds[0]) & ((np.arange(gw)+1)*sx<=bounds[2]))
            gy=np.flatnonzero((np.arange(gh)*sy>=bounds[1]) & ((np.arange(gh)+1)*sy<=bounds[3]))
            check([int(gx[0]),int(gy[0]),int(gx[-1])+1,int(gy[-1])+1]==r['GT_full_cell_domain_xyxy'],'independent cell area domain')
            labels=np.load(a.source/r['wsi_id']/'labels.npy',mmap_mode='r');raw=np.memmap(a.source/r['wsi_id']/'original_gt_rgb.raw',mode='r',dtype=np.uint8,shape=(gh,gw,3))
            check(labels.shape==(gh,gw) and labels.dtype==np.uint8,'label dimensions')
            counts=np.zeros(256,dtype=np.int64);domain=np.zeros(256,dtype=np.int64)
            for y in range(0,gh,193):
                end=min(y+193,gh);part=labels[y:end];counts+=np.bincount(part.reshape(-1),minlength=256)
                lo=max(y,int(gy[0]));hi=min(end,int(gy[-1])+1)
                if lo<hi:domain+=np.bincount(labels[lo:hi,int(gx[0]):int(gx[-1])+1].reshape(-1),minlength=256)
                if hasattr(labels._mmap,'madvise'):labels._mmap.madvise(mmap.MADV_DONTNEED)
            check(all(int(counts[c])==r['class_counts'][str(c)] and int(domain[c])==r['domain_class_counts'][str(c)] for c in range(12)),'independent full label/domain histogram')
            check(int(counts[255])==r['unknown_pixels'] and int(domain[255])==r['domain_unknown_pixels'],'unknown counts')
            probes=0
            for y in np.linspace(0,gh-1,33,dtype=int):
                for x in np.linspace(0,gw-1,33,dtype=int):
                    expected=color_to_label.get(tuple(map(int,raw[y,x])),255)
                    check(int(labels[y,x])==expected,'independent tuple palette sample');probes+=1
            totals+=domain[:12];del raw,labels
            evidence.append(dict(wsi_id=r['wsi_id'],label_pixels=gw*gh,palette_probes=probes,grid_tiles=len(tiles)))
            dump('status.json',dict(status='independent_histogram_and_geometry',worker_pid=pid,slides=i+1,total=len(rows)));print(json.dumps(evidence[-1]),flush=True)
        check(all(int(totals[c])==source['class_domain_pixels'][str(c)] for c in range(12)),'cohort domain totals')
        result=dict(status='accepted_main30_gate' if source['gate'] else 'accepted_main30_data_and_geometry_qualified',worker_pid=pid,slides=len(rows),evidence=evidence,
            source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),total_seconds=time.perf_counter()-started,
            scope='Every source/output byte hash; full labels/domain counts and every grid footprint;1089 independently tuple-decoded palette points per slide',
            limits=['No model quality or independent clinical patient claim','Extent affine inherited, anatomical alignment not independently proven','No full rescan/perceptual/upstream-pretraining exposure proof'])
        dump('summary.json',result);print(json.dumps(dict(status=result['status'],slides=len(rows),seconds=result['total_seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
