"""Independent real source-band, click, RGB and query-footprint acceptance."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid();summary=read(a.protocol/'summary.json');contract=read(a.protocol/'contract.json')
    check(summary['status']=='prepared_support_query_protocol','complete protocol')
    for name,d in summary['output_sha256'].items():check(sha(a.protocol/name)==d,'protocol file SHA')
    for p,d in summary['source_sha256'].items():check(sha(p)==d,'source code unchanged')
    check(contract['data_summary_sha256']==sha(a.data/'summary.json'),'accepted data identity')
    original={r['wsi_id']:r for r in read(a.data/'slides_private.json')}
    inputs=read(a.protocol/'query_inputs.json');tasks=read(a.protocol/'support_tasks_maximum.json');coverage=read(a.protocol/'coverage.json');supports=read(a.protocol/'support_inputs_private.json')
    check(len(inputs)==len(original)==summary['slides'],'all expected slides')
    groups={};points=0;RGBviews=0;total_query=0
    for r in inputs:
        sid=r['wsi_id'];old=original[sid];w,h=old['native_wh'];gw,gh=old['gt_wh'];sx,sy=w/gw,h/gh;cut=w//4;gap=math.ceil(1/old['native_mpp'][0]);folder=a.protocol/sid
        check(r['source_cut_native']==cut and r['gap_native']==gap,'predefined geometry-only band')
        band=np.load(folder/'support_band_labels.npy',mmap_mode='r');source=np.load(a.data/sid/'labels.npy',mmap_mode='r')
        check(band.shape==(gh,math.floor(cut/sx)) and np.array_equal(band,source[:,:band.shape[1]]),'full original source-band export')
        del source
        oldgrid=[json.loads(line) for line in Path(old['grid']).open()]
        expected=[t for t in oldgrid if t['native_boxes'][2][0]>=cut+gap]
        grid=[json.loads(line) for line in Path(r['grid']).open()]
        check(grid==expected and sha(r['grid'])==r['grid_sha256'],'complete geometry-only remote grid, no class filter')
        check(len(grid)==r['query_tiles'],'query count');total_query+=len(grid)
        candidates=read(folder/'source_candidate_geometry.json');bytarget={c:[t['tile'] for t in candidates if c in t['eligible_targets']] for c in range(12) if c!=2}
        bynumber={t['number']:t for t in oldgrid};local=[p for p in supports if p['wsi_id']==sid];selected={p['support_id']:p for p in local}
        native=pyvips.Image.new_from_file(old['source_wsi'],access='random')
        for p in local:
            t=bynumber[p['tile']];box=t['native_boxes'][0]
            check(t['native_boxes'][2][0]+t['native_boxes'][2][2]<=cut,'full source4x inside support band')
            lab=np.load(p['labels'],allow_pickle=False);check(sha(p['labels'])==p['labels_sha256'],'support label cache SHA')
            # Reconstruct pixel centres with Python scalar arithmetic, then index only the exported band.
            xx=[math.floor((box[0]+(j+.5)*box[2]/512)/sx) for j in range(512)]
            yy=[math.floor((box[1]+(j+.5)*box[3]/512)/sy) for j in range(512)]
            check(np.array_equal(lab,band[np.ix_(yy,xx)]),'all actual source label sample pixels')
            for item in p['images']:
                b=item['native_box'];check(b in t['native_boxes'],'source view in original full-context grid')
                raw=native.crop(*b).extract_band(0,n=3);rgb=np.frombuffer(raw.write_to_memory(),np.uint8).reshape(b[3],b[2],3)
                image=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR))
                check(sha(item['path'])==item['sha256'] and np.array_equal(np.asarray(Image.open(item['path'])),image),'actual native RGB with accepted PIL resampling');RGBviews+=1
        for c in range(12):
            if c==2:continue
            cov=next(q for q in coverage if q['wsi_id']==sid and q['target']==c)
            check(cov['legal_source_tiles']==len(bytarget[c]),'all candidate availability retained')
            plans=[t for t in tasks if t['wsi_id']==sid and t['target']==c]
            check(len(plans)==(5 if bytarget[c] else 0),'unavailable targets explicit, no invented supports')
            if not plans:continue
            chosen=min(bytarget[c],key=lambda n:hashlib.sha256(f'WSI-source-band-v1|{sid}|{c}|{n}'.encode()).hexdigest())
            check({t['prompt_group'] for t in plans}==set(range(5)) and all(t['source_tile']==chosen for t in plans),'fixed SHA selection and groups')
            for plan in plans:
                p=selected[plan['support_id']];lab=np.load(p['labels'],allow_pickle=False);box=bynumber[chosen]['native_boxes'][0]
                for sign,count in [('positive',5),('negative',2)]:
                    check(len(plan[sign+'_xy'])==count,'maximum point budget')
                    for (x,y),native_xy in zip(plan[sign+'_xy'],plan[sign+'_native_xy']):
                        check(0<x<1 and 0<y<1,'normalised point bound')
                        native_x,native_y=box[0]+x*box[2],box[1]+y*box[3]
                        check([native_x,native_y]==native_xy and native_x<cut,'source coordinate backprojection')
                        v=int(lab[math.floor(y*512),math.floor(x*512)])
                        check(v==c if sign=='positive' else v not in (c,255),'support sign and unknown policy');points+=1
                groups[sid,c]=5
    def no_GT(value):
        if isinstance(value,dict):
            check(not (set(value)&{'labels','target','class_counts','domain_class_counts','GT_to_native_affine','labels_sha256','labels_array_sha256'}),'inference metadata separated from labels/target IDs')
            for v in value.values():no_GT(v)
        elif isinstance(value,list):
            for v in value:no_GT(v)
    no_GT(inputs)
    result=dict(status='accepted_source_band_query_protocol',worker_pid=pid,slides=len(inputs),tasks=len(tasks),available_class_slide_pairs=len(groups),query_tiles=total_query,
        checked_source_clicks=points,actual_native_RGB_views=RGBviews,protocol_summary_sha256=sha(a.protocol/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-started,
        scope='Source-band pixels, selected source-label pixels/RGB, all click signs/native backprojections, source4x versus complete query geometry, inference metadata permissions',remaining='No model predictions or quality scoring; unprompted component/distance evaluator not accepted')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','data','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
