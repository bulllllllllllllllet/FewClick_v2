"""Independent OpenCV8CCL bijection, full area/centroid and prompt-domain audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();started=time.perf_counter();cv2.setNumThreads(4)
    source=read(a.source/'summary.json');require(source['status'] in ('prepared_GT_component_gate','complete_main30_GT_component_preparation'),'wholeGT source preparation complete')
    require(source['recipe']['source_sha256']==sha('benchmarks/v5/wsi_global/gt_components.py'),'component recipe frozen')
    if not source['gate']:
        require(a.verified_gate is not None,'real independent complete2slide component gate required');g=read(a.verified_gate/'summary.json');require(g['status']=='accepted_GT_component_gate' and g['source_sha256']==sha(__file__) and g['opencv_version']==cv2.__version__,'same independent CCL audit recipe')
    require(sha(a.source/'items.json')==source['items_sha256'] and sha(a.source/'task_domains.json')==source['tasks_sha256'],'all producer data outputs bound')
    items=read(a.source/'items.json');plans=read(a.protocol/'support_tasks_maximum.json');task_domains=read(a.source/'task_domains.json');data={r['wsi_id']:r for r in read(a.data/'slides_private.json')};queries=read(a.protocol/'query_inputs.json')
    require(len(items)==(2 if source['gate'] else 30) and len(task_domains)==4*len(plans),'all slides and budget metadata present')
    evidence=[];component_total=0;checked_tasks=0
    for number,item in enumerate(items):
        sid=item['wsi_id'];r=data[sid];folder=a.source/sid
        for d in item['files'].values():require(sha(d['path'])==d['sha256'],'every component raster/table/task byte')
        gt=np.load(a.data/sid/'labels.npy',mmap_mode='r');components=np.load(folder/'components.npy',mmap_mode='r');require(components.shape==gt.shape and components.dtype==np.uint32,'wholeGT actual component raster')
        with np.load(folder/'component_table.npz',allow_pickle=False) as z:table={k:z[k] for k in z.files}
        classes=table['classes'];areas=table['areas'];remote=table['remote_areas'];gh,gw=gt.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh
        require(len(classes)==item['components']+1 and areas.shape==remote.shape==classes.shape,'all component IDs/table shapes')
        row=next(q for q in queries if q['wsi_id']==sid);tiles=[json.loads(line) for line in Path(row['grid']).open()];boxes=[t['native_boxes'][0] for t in tiles]
        bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)];require(bounds==item['remote_native_bounds'],'frozen query image geometry')
        gx=np.flatnonzero((np.arange(gw)*sx>=bounds[0])&((np.arange(gw)+1)*sx<=bounds[2]));gy=np.flatnonzero((np.arange(gh)*sy>=bounds[1])&((np.arange(gh)+1)*sy<=bounds[3]));x0,y0,x1,y1=int(gx[0]),int(gy[0]),int(gx[-1])+1,int(gy[-1])+1
        require([x0,y0,x1,y1]==item['remote_GT_bounds'],'independent complete-cell GT domain')
        remote_by_id={};class_by_id={};hist_by_class={};centroid_error=0.;count=0
        for c in range(12):
            mask=(np.asarray(gt)==c).astype(np.uint8)
            n,local,stats,centres=cv2.connectedComponentsWithStatsWithAlgorithm(mask,8,cv2.CV_32S,cv2.CCL_GRANA)
            mapping=np.zeros(n,np.uint32)
            for label_id in range(1,n):
                left,top,width,height,area=map(int,stats[label_id]);xs=np.flatnonzero(local[top,left:left+width]==label_id);require(len(xs)>0,'independent topmost original component pixel')
                gid=int(components[top,left+int(xs[0])]);require(gid>0 and int(classes[gid])==c,'component correspondence and original GT class');mapping[label_id]=gid
                require(int(areas[gid])==area,'all fullGT component pixel areas')
                error=max(abs(float(table['centroid_x_GT_index'][gid])-float(centres[label_id,0])),abs(float(table['centroid_y_GT_index'][gid])-float(centres[label_id,1])));require(error<=1e-8,'independent centroid sums/division');centroid_error=max(centroid_error,error)
                require(abs(float(table['centroid_x_native'][gid])-(float(centres[label_id,0])+.5)*sx)<=3e-8 and abs(float(table['centroid_y_native'][gid])-(float(centres[label_id,1])+.5)*sy)<=3e-8,'physical pixel-centre centroid mapping')
                class_by_id[gid]=c
            require(len(set(map(int,mapping[1:])))==n-1 and set(map(int,mapping[1:]))==set(map(int,np.flatnonzero(classes==c))),'bijection, no merge or fragmentation')
            for y in range(0,gh,193):
                end=min(y+193,gh);expected=np.where(np.asarray(gt[y:end])==c,np.asarray(components[y:end]),0)
                require(np.array_equal(mapping[local[y:end]],expected),'every originalGT pixel independently connected8')
            local_remote=np.bincount(local[y0:y1,x0:x1].reshape(-1),minlength=n)
            for label_id in range(1,n):
                gid=int(mapping[label_id]);require(int(remote[gid])==int(local_remote[label_id]),'all remote component areas');remote_by_id[gid]=int(local_remote[label_id])
            count+=n-1;hist_by_class[c]=int(local_remote[1:].sum())
            (a.output/'status.json').write_text(json.dumps(dict(status='independent_original8CCL',worker_pid=pid,slides_complete=number,total_slides=len(items),wsi_id=sid,classes_complete=c+1))+'\n')
            del mask,local,stats,centres
        require(count==item['components'] and np.array_equal(components==0,gt==255),'all same-class components plus void0')
        valid=sum(remote_by_id.values());require(valid+item['remote_unknown_pixels']==len(gx)*len(gy),'remote valid/nonvoid coverage')
        for plan in [p for p in plans if p['wsi_id']==sid]:
            point_ids={}
            for sign in ('positive','negative'):
                ids=[]
                for nx,ny in plan[sign+'_native_xy']:
                    x,y=math.floor(nx/sx),math.floor(ny/sy);gid=int(components[y,x]);require(gid>0 and (class_by_id[gid]==plan['target'] if sign=='positive' else class_by_id[gid]!=plan['target']),'actual source point component class');ids.append(gid)
                point_ids[sign]=ids
            for budget in ('3+2','1+1','3+1','5+1'):
                npos,nneg=map(int,budget.split('+'));t=next(t for t in task_domains if t['wsi_id']==sid and t['target']==plan['target'] and t['prompt_group']==plan['prompt_group'] and t['budget']==budget)
                positive=point_ids['positive'][:npos];negative=point_ids['negative'][:nneg];prompted=set(positive);removed=sum(remote_by_id[g] for g in prompted);positive_area=hist_by_class[plan['target']]
                unprompted=sum(class_by_id[g]==plan['target'] and n>0 and g not in prompted for g,n in remote_by_id.items())
                require(t['positive_component_ids']==positive and t['negative_component_ids']==negative and t['prompted_components']==sorted(prompted),'all budget-prefix point/component bindings')
                require((t['remote_valid_pixels'],t['remote_target_pixels'],t['prompted_target_pixels_in_remote'],t['censored_valid_pixels'],t['unprompted_target_pixels'],t['negative_GT_pixels'],t['unprompted_components'])==(valid,positive_area,removed,valid-removed,positive_area-removed,valid-positive_area,unprompted),'every independently recomputed prompt-censored denominator')
                checked_tasks+=1
        component_total+=count;evidence.append(dict(wsi_id=sid,components=count,every_original_pixel_bijection=True,centroid_maximum_error=centroid_error));print(json.dumps(evidence[-1]),flush=True);del gt,components
    require(checked_tasks==len(task_domains),'every original task-budget accounted')
    result=dict(status='accepted_GT_component_gate' if source['gate'] else 'accepted_all30_GT_components_and_prompt_domains',worker_pid=pid,slides=len(items),components=component_total,tasks=checked_tasks,evidence=evidence,source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),opencv_version=cv2.__version__,seconds=time.perf_counter()-started,
        scope='WholeGT8 connectivity pixelwise bijection via independent GRANA, all area/remote/centroid, actual source-point/component and censored-domain counts; no prediction quality',remaining='Actual predicted component area recall/detection and distance-stratified segmentation scores')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],components=component_total,tasks=checked_tasks,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','data','protocol','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
