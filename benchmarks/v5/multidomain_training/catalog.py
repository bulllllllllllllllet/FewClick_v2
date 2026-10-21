"""Audit immutable cell inputs and support-click eligibility before episodes."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import minimum_filter,maximum_filter

from benchmarks.v5.multidomain_data.prepare import read_rows,dump,sha

INPUTS=Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/instanseg_adoption_1020_185343/full_cells_1020_205325/inputs_with_cells.parquet')
SOURCES=Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/prepare_1020_174937/catalog_1020_180034/sources.parquet')
TARGETS={'crc':list(range(2))+list(range(3,12)),'ocelot':[2],'ignite':list(range(2,17)),'tiger':list(range(1,7))}


def interior_labels(labels,mpp,margin_um=2.):
    """Conservative rectangular interior: every neighbour has the same label."""
    radius=[int(np.ceil(margin_um/float(v))) for v in mpp]
    size=(2*radius[1]+1,2*radius[0]+1)
    lo=minimum_filter(labels,size=size,mode='constant',cval=255)
    hi=maximum_filter(labels,size=size,mode='constant',cval=255)
    return np.where((lo==hi)&(labels!=255),labels,255).astype(np.uint8)


def footprint(row,source):
    e=json.loads(source['extra_json']);x,y,w,h=json.loads(row['image_box_json'])
    mx,my=source['available_mpp_x'],source['available_mpp_y'];ox=oy=0.
    if row['dataset']=='crc':ox,oy=e['image_window'][:2];ox*=mx;oy*=my
    elif row['dataset']=='ocelot':
        meta=e['metadata'];ox=meta['tissue']['x_start']*meta['mpp_x'];oy=meta['tissue']['y_start']*meta['mpp_y']
    elif row['dataset']=='tiger':ox,oy=e['slide_origin_xy'];ox*=mx;oy*=my
    verified=bool(source['slide_id_verified'])
    key=row['dataset']+'|'+str(source['slide_id'] if verified else source['source_id'])
    return key,[ox+x*mx,oy+y*my,ox+(x+w)*mx,oy+(y+h)*my],verified


def disjoint(box,boxes,gap_um=1.):
    b=np.asarray(boxes)
    return (box[2]+gap_um<=b[:,0])|(b[:,2]+gap_um<=box[0])|(box[3]+gap_um<=b[:,1])|(b[:,3]+gap_um<=box[1])


def split_overlap_audit(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['space_key']].append(r)
    checked=0
    for key,group in groups.items():
        active=[]
        for row in sorted(group,key=lambda r:r['x0_um']):
            active=[r for r in active if r['x1_um']>row['x0_um']+1e-5]
            for other in active:
                if row['split']==other['split']:continue
                checked+=1
                if min(row['y1_um'],other['y1_um'])>max(row['y0_um'],other['y0_um'])+1e-5:
                    raise ValueError(f'cross-split physical overlap: {key} {row["tile_id"]} {other["tile_id"]}')
            active.append(row)
    return dict(status='passed',cross_split_x_candidates_checked=checked,scope='known same-slide/ROI fine-image footprints; unknown cross-ROI correspondence remains unverified')


def build_tile(item):
    rows,source=item;first=rows[0];qualities={r['quality']:r for r in rows}
    if set(qualities)!={'original','0.8','1.0'}:raise ValueError('incomplete quality set')
    for r in rows:
        if r['labels_array_sha256']!=first['labels_array_sha256'] or r['split']!=first['split']:raise ValueError('quality identity/split differs')
    labels=np.asarray(Image.open(first['labels'])).copy()
    if labels.shape!=(512,512) or sha(labels.tobytes())!=first['labels_array_sha256']:raise ValueError('GT identity/shape')
    mpp=json.loads(first['input_mpp_xy_json']);inside=interior_labels(labels,mpp)
    counts={int(k):int(v) for k,v in json.loads(first['class_counts_json']).items()}
    inner={int(k):int(v) for k,v in zip(*np.unique(inside,return_counts=True)) if k!=255}
    valid=sum(v for k,v in counts.items() if k!=255);total=sum(inner.values())
    eligible=[c for c in TARGETS[first['dataset']] if counts.get(c,0)>=64 and valid-counts.get(c,0)>=64 and inner.get(c,0)>=5 and total-inner.get(c,0)>=5]
    key,box,verified=footprint(first,source)
    variants={q:{k:r[k] for k in ['image','image_sha256','cells','cells_sha256']} for q,r in qualities.items()}
    return {k:first[k] for k in ['tile_id','source_id','dataset','organ','split','spatial_group_id','training_allowed','labels','labels_array_sha256','input_mpp_xy_json','cell_recipe_id']}|dict(
        space_key=key,slide_verified=verified,patient_id=source['patient_id'],x0_um=box[0],y0_um=box[1],x1_um=box[2],y1_um=box[3],
        counts_json=json.dumps(counts),eligible_json=json.dumps(eligible),interior_counts_json=json.dumps(inner),variants_json=json.dumps(variants))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=INPUTS);p.add_argument('--sources',type=Path,default=SOURCES);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        source={r['source_id']:r for r in read_rows(a.sources)};grouped=defaultdict(list)
        for r in read_rows(a.inputs):
            if r['dataset']=='puma':continue
            if r['split'] not in ('train','val'):raise ValueError('test input is not allowed')
            if r['split']=='train' and not r['training_allowed']:raise ValueError('training policy mismatch')
            grouped[r['tile_id']].append(r)
        dump(a.output/'run.json',dict(worker_pid=os.getpid(),inputs=str(a.inputs),input_sha256=sha(a.inputs.read_bytes()),sources_sha256=sha(a.sources.read_bytes()),source_sha256=sha(Path(__file__).read_bytes()),margin_um=2.,minimum_interior_points_per_sign=5,puma_GT_read=False))
        rows=[]
        with ThreadPoolExecutor(max_workers=a.threads) as pool,(a.output/'progress.jsonl').open('w') as log:
            work=((value,source[value[0]['source_id']]) for _,value in sorted(grouped.items()))
            for row in pool.map(build_tile,work):
                rows.append(row);log.write(json.dumps(dict(tile_id=row['tile_id'],eligible=json.loads(row['eligible_json'])))+'\n')
                if len(rows)%200==0:
                    log.flush();dump(a.output/'status.json',dict(status='building',worker_pid=os.getpid(),completed=len(rows),total=len(grouped),seconds=time.time()-started));print('CATALOG',len(rows),'/',len(grouped),flush=True)
        audit=split_overlap_audit(rows)
        if len({r['cell_recipe_id'] for r in rows})!=1:raise ValueError('mixed detector/feature recipes')
        pd.DataFrame(rows).to_parquet(a.output/'tiles.parquet',index=False)
        dump(a.output/'summary.json',dict(status='complete',tiles=len(rows),split_overlap=audit,counts=dict(Counter(r['dataset']+'|'+r['split'] for r in rows)),eligible_counts=dict(Counter(r['dataset']+'|'+r['split'] for r in rows if json.loads(r['eligible_json']))),seconds=time.time()-started,puma_used=False,test_used=False))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(rows),seconds=time.time()-started))
    except Exception as exc:dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
