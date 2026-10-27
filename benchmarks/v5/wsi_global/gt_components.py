"""Whole-original-GT same-class8 components and source-prompt-censored domains."""
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import skimage
from skimage.measure import label


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();pid=os.getpid()
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        receipt=read(a.data_acceptance/'summary.json');source=read(a.data/'summary.json')
        require(receipt['status'] in ('accepted_main30_gate','accepted_main30_data_and_geometry_qualified') and receipt['source_summary_sha256']==sha(a.data/'summary.json'),'accepted original whole GT')
        accepted=read(a.protocol_acceptance/'summary.json');require(accepted['status']=='accepted_source_band_query_protocol' and accepted['protocol_summary_sha256']==sha(a.protocol/'summary.json'),'accepted frozen source prompts/query')
        gate=source['gate'];records=read(a.data/'slides_private.json');queries=read(a.protocol/'query_inputs.json');plans=read(a.protocol/'support_tasks_maximum.json')
        require(len(records)==len(queries)==(2 if gate else 30),'all original requested slides')
        recipe=dict(source_sha256=sha(__file__),numpy=np.__version__,skimage=skimage.__version__,connectivity=8,void=255,full_GT_before_query_crop=True,filters=False)
        if not gate:
            require(a.verified_gate is not None,'actual full two-slide component gate required');g=read(a.verified_gate/'summary.json');require(g['status']=='prepared_GT_component_gate' and g['recipe']==recipe,'same scientific component recipe')
        fixture=np.array([[0,255,1],[255,0,1],[2,2,255]],np.uint8);test,n=label(fixture,background=255,connectivity=2,return_num=True)
        require(n==3 and test[0,0]==test[1,1]!=0 and test[2,0]==test[2,1]!=0 and test[0,1]==0 and test[0,2]!=test[0,0],'class0/background2/void and8connected diagonal semantics')
        dump('contract.json',dict(worker_pid=pid,recipe=recipe,data_summary_sha256=sha(a.data/'summary.json'),protocol_summary_sha256=sha(a.protocol/'summary.json'),gate=gate,GT_role='scoring-only after frozen source/query protocol; never input to model or prompt selection'))
        results=[];all_tasks=[]
        for index,r in enumerate(records):
            available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            require(available>40*1024**3,'actual RAM headroom for full-image original8CCL')
            folder=a.output/r['wsi_id'];folder.mkdir();gtpath=a.data/r['wsi_id']/'labels.npy';require(sha(gtpath)==r['files'][str(gtpath)]['sha256'],'original GT identity')
            gt=np.load(gtpath,mmap_mode='r');gh,gw=gt.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh;mx,my=r['native_mpp'];q=next(q for q in queries if q['wsi_id']==r['wsi_id']);tiles=[json.loads(line) for line in Path(q['grid']).open()];boxes=[t['native_boxes'][0] for t in tiles]
            bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
            gx0,gy0,gx1,gy1=math.ceil(bounds[0]/sx),math.ceil(bounds[1]/sy),math.floor(bounds[2]/sx),math.floor(bounds[3]/sy)
            dump('status.json',dict(status='full_original_GT_multiclass8_label',worker_pid=pid,completed_slides=index,total=len(records),wsi_id=r['wsi_id']))
            components,count=label(np.asarray(gt),background=255,connectivity=2,return_num=True);require(count<2**32,'uint32 semantic component capacity')
            output=np.lib.format.open_memmap(folder/'components.npy',mode='w+',dtype=np.uint32,shape=gt.shape)
            classes=np.full(count+1,255,np.uint8);areas=np.zeros(count+1,np.int64);remote=np.zeros(count+1,np.int64);row_sum=np.zeros(count+1,np.float64);col_sum=np.zeros(count+1,np.float64)
            for y in range(0,gh,256):
                end=min(y+256,gh);ids=components[y:end].reshape(-1);codes=np.asarray(gt[y:end]).reshape(-1);old=classes[ids];known=old!=255
                require(np.array_equal(ids==0,codes==255) and np.array_equal(old[known],codes[known]),'void and same-class component consistency across bands')
                classes[ids]=codes;require(np.array_equal(classes[ids],codes),'no component merges different tissue classes')
                areas+=np.bincount(ids,minlength=count+1);row_sum+=np.bincount(ids,weights=np.repeat(np.arange(y,end,dtype=np.float64),gw),minlength=count+1);col_sum+=np.bincount(ids,weights=np.tile(np.arange(gw,dtype=np.float64),end-y),minlength=count+1)
                lo,hi=max(y,gy0),min(end,gy1)
                if lo<hi:remote+=np.bincount(components[lo:hi,gx0:gx1].reshape(-1),minlength=count+1)
                output[y:end]=components[y:end].astype(np.uint32)
            output.flush();del components
            require(int(areas.sum())==gw*gh and int(remote.sum())==(gx1-gx0)*(gy1-gy0),'all original/remote component area conserved')
            class_areas=np.zeros(256,np.int64);np.add.at(class_areas,classes,areas)
            require(all(int(class_areas[c])==r['class_counts'][str(c)] for c in range(12)) and int(class_areas[255])==r['unknown_pixels'],'component class totals match accepted originalGT census')
            cx=np.divide(col_sum,areas,out=np.zeros_like(col_sum),where=areas>0);cy=np.divide(row_sum,areas,out=np.zeros_like(row_sum),where=areas>0)
            np.savez_compressed(folder/'component_table.npz',classes=classes,areas=areas,remote_areas=remote,centroid_x_GT_index=cx,centroid_y_GT_index=cy,centroid_x_native=(cx+.5)*sx,centroid_y_native=(cy+.5)*sy)
            this=[];valid=int(remote[1:].sum())
            for p in [p for p in plans if p['wsi_id']==r['wsi_id']]:
                ids_by_sign={}
                for sign in ('positive','negative'):
                    point_ids=[]
                    for nx,ny in p[sign+'_native_xy']:
                        gx,gy=math.floor(nx/sx),math.floor(ny/sy);require(0<=gx<gw and 0<=gy<gh,'source point fullGT coordinate')
                        component=int(output[gy,gx]);code=int(classes[component]);require(component>0 and (code==p['target'] if sign=='positive' else code not in (p['target'],255)),'all source-point component identities/signs')
                        point_ids.append(component)
                    ids_by_sign[sign]=point_ids
                target_ids=np.flatnonzero(classes==p['target']);target_remote=int(remote[target_ids].sum())
                for budget in ('3+2','1+1','3+1','5+1'):
                    npos,nneg=map(int,budget.split('+'));prompted=sorted(set(ids_by_sign['positive'][:npos]));censored=int(remote[prompted].sum());unprompted=target_ids[(remote[target_ids]>0)&~np.isin(target_ids,prompted)]
                    item=dict(wsi_id=r['wsi_id'],target=p['target'],prompt_group=p['prompt_group'],budget=budget,positive_component_ids=ids_by_sign['positive'][:npos],negative_component_ids=ids_by_sign['negative'][:nneg],prompted_components=prompted,
                        remote_valid_pixels=valid,remote_target_pixels=target_remote,prompted_target_pixels_in_remote=censored,censored_valid_pixels=valid-censored,unprompted_target_pixels=target_remote-censored,unprompted_components=int(len(unprompted)),negative_GT_pixels=valid-target_remote,
                        component_detection_area_recall_threshold=.5,sensitivity_threshold=.1)
                    require(item['censored_valid_pixels']==item['negative_GT_pixels']+item['unprompted_target_pixels'],'non-target query pixels retained after prompt-component removal');this.append(item)
            dump(r['wsi_id']+'/task_component_domains.json',this);all_tasks+=this
            files={p.name:dict(path=str(p),sha256=sha(p)) for p in folder.iterdir() if p.is_file()};result=dict(wsi_id=r['wsi_id'],GT_shape=[gh,gw],components=count,GT_to_native=[sx,sy],GT_cell_area_um2=sx*mx*sy*my,remote_native_bounds=bounds,remote_GT_bounds=[gx0,gy0,gx1,gy1],unknown_pixels=int(areas[0]),remote_unknown_pixels=int(remote[0]),class_component_counts={str(c):int(np.count_nonzero(classes==c)) for c in range(12)},files=files,tasks=len(this))
            results.append(result);dump('items.json',results);print(json.dumps(dict(wsi_id=r['wsi_id'],components=count,tasks=len(this))),flush=True)
            del gt,output
        dump('task_domains.json',all_tasks);summary=dict(status='prepared_GT_component_gate' if gate else 'complete_main30_GT_component_preparation',worker_pid=pid,gate=gate,recipe=recipe,slides=len(results),tasks=len(all_tasks),items_sha256=sha(a.output/'items.json'),tasks_sha256=sha(a.output/'task_domains.json'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.perf_counter()-start,
            scope='Full originalGT8connected tissue components and frozen source-point-censored scoring domains only; not predictions/quality/biological nuclei',remaining='Independent CCL/GT-domain acceptance, actual per-component prediction counts and distance/BF1 scoring')
        dump('summary.json',summary);dump('status.json',dict(status=summary['status'],worker_pid=pid));print(json.dumps(dict(status=summary['status'],slides=len(results),seconds=summary['seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('data','data-acceptance','protocol','protocol-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
