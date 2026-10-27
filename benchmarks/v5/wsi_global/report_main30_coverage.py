"""Report every CRC30 class and strict remote area after protocol freeze."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();data=read(a.data/'slides_private.json');protocol=read(a.protocol/'query_inputs.json');available=read(a.protocol/'coverage.json');accepted=read(a.acceptance/'summary.json')
    assert accepted['status']=='accepted_source_band_query_protocol' and accepted['slides']==30 and accepted['protocol_summary_sha256']==sha(a.protocol/'summary.json')
    names=['tumor_epithelium','tumor_stroma','background','necrosis','normal_gland','normal_stroma','submucosa_serosa','muscle','lymphocyte_aggregate','mucus','fat','blood'];rows=[]
    for r in protocol:
        old=next(x for x in data if x['wsi_id']==r['wsi_id']);gw,gh=old['gt_wh'];w,h=old['native_wh'];sx,sy=w/gw,h/gh
        tiles=[json.loads(line) for line in Path(r['grid']).open()];boxes=[t['native_boxes'][0] for t in tiles]
        bx0=min(b[0] for b in boxes);by0=min(b[1] for b in boxes);bx1=max(b[0]+b[2] for b in boxes);by1=max(b[1]+b[3] for b in boxes)
        assert len(tiles)==len(set(b[0] for b in boxes))*len(set(b[1] for b in boxes))
        x0,y0,x1,y1=math.ceil(bx0/sx),math.ceil(by0/sy),math.floor(bx1/sx),math.floor(by1/sy)
        labels=np.load(a.data/r['wsi_id']/'labels.npy',mmap_mode='r');counts=np.zeros(256,dtype=np.int64)
        for y in range(y0,y1,193):counts+=np.bincount(np.asarray(labels[y:min(y+193,y1),x0:x1]).reshape(-1),minlength=256)
        assert int(counts.sum())==(x1-x0)*(y1-y0)
        for c in range(12):
            cov=next((x for x in available if x['wsi_id']==r['wsi_id'] and x['target']==c),None)
            rows.append(dict(wsi_id=r['wsi_id'],target=c,name=names[c],full_GT_pixels=old['class_counts'][str(c)],remote_GT_pixels=int(counts[c]),GT_domain_pixels=int(counts.sum()),unknown_pixels=int(counts[255]),
                source_support_available=bool(cov and cov['status']=='available'),support_status='background_not_positive_target' if c==2 else cov['status'],
                native_remote_bounds=[bx0,by0,bx1,by1],GT_full_pixel_bounds=[x0,y0,x1,y1],native_area_fraction=(bx1-bx0)*(by1-by0)/(w*h)))
        print(json.dumps(dict(wsi_id=r['wsi_id'],remote_native_area_fraction=rows[-1]['native_area_fraction'])),flush=True)
    by_class=[]
    for c in range(12):
        selected=[r for r in rows if r['target']==c]
        by_class.append(dict(target=c,name=names[c],slides=30,full_GT_present=sum(r['full_GT_pixels']>0 for r in selected),remote_GT_present=sum(r['remote_GT_pixels']>0 for r in selected),
            support_available=sum(r['source_support_available'] for r in selected),support_and_remote_present=sum(r['source_support_available'] and r['remote_GT_pixels']>0 for r in selected),
            support_available_remote_absent=sum(r['source_support_available'] and r['remote_GT_pixels']==0 for r in selected),remote_present_but_support_unavailable=sum(r['remote_GT_pixels']>0 and not r['source_support_available'] for r in selected)))
    result=dict(status='complete_frozen_main30_task_coverage_report',worker_pid=os.getpid(),source_sha256=sha(__file__),protocol_summary_sha256=sha(a.protocol/'summary.json'),acceptance_summary_sha256=sha(a.acceptance/'summary.json'),rows=rows,by_class=by_class,seconds=time.perf_counter()-start,
        scope='GT used only for post-freeze reporting/scoring denominators; does not change source/query selection; no model quality score',limitations=['Strict remote subdomain is not full tissue coverage','Unsupported targets are recorded unavailable, not silently omitted or scored as prediction failures','Background remains negative/scoring domain and closed-reference class'])
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=['# 主30组织与固定远端任务覆盖','', '| ID | 组织 | 全GT存在 | 远端存在 | 有合法支持 | 支持且远端存在 | 支持有/远端无 | 远端有/支持无 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in by_class:text.append('|'+str(r['target'])+'|'+r['name']+'|'+'|'.join(str(r[k]) for k in ['full_GT_present','remote_GT_present','support_available','support_and_remote_present','support_available_remote_absent','remote_present_but_support_unavailable'])+'|')
    text+=['','背景不设正目标，支持列0不代表训练没见背景。各片完整矩形/GT像素域、背景/unknown与排除范围见summary.json。所有30张和11组织可用性都保留；无模型预测/精度结果。']
    (a.output/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','protocol','acceptance','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
