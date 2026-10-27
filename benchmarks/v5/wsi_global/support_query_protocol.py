"""Freeze same-WSI source-only prompts and a geometrically disjoint remote grid."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
from PIL import Image

from benchmarks.v5.multidomain_training.catalog import interior_labels
from benchmarks.v5.multidomain_training.click_policy import representative_clicks, POLICY

pyvips.cache_set_max(0)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(test,message):
    if not test:raise ValueError(message)


def dump(path,d):Path(path).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def labels_for_box(support,box,scale_xy):
    x,y,w,h=box;sx,sy=scale_xy
    xs=np.floor((x+(np.arange(512)+.5)*w/512)/sx).astype(int)
    ys=np.floor((y+(np.arange(512)+.5)*h/512)/sy).astype(int)
    require(min(xs)>=0 and max(xs)<support.shape[1] and min(ys)>=0 and max(ys)<support.shape[0], 'support-only annotation access')
    return np.asarray(support[np.ix_(ys,xs)]).copy()


def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    try:
        audited=read(a.data/'summary.json');accepted=read(a.acceptance/'summary.json')
        require(audited['status'] in ('passed_main30_source_gate','complete_main30_full_GT_and_geometry_audit_qualified'),'data source complete')
        require(accepted['status'] in ('accepted_main30_gate','accepted_main30_data_and_geometry_qualified') and accepted['source_summary_sha256']==sha(a.data/'summary.json'),'independent source acceptance')
        records=read(a.data/'slides_private.json');require(sha(a.data/'slides_private.json')==audited['slides_sha256'],'data records binding')
        sources={str(p):sha(p) for p in (Path(__file__),Path('benchmarks/v5/multidomain_training/catalog.py'),Path('benchmarks/v5/multidomain_training/click_policy.py'))}
        if not audited['gate']:
            require(a.verified_gate is not None, 'actual same-source support/query gate required')
            old=read(a.verified_gate/'summary.json');require(old['status']=='prepared_support_query_protocol' and old['source_gate'] and old['source_sha256']==sources,'support/query gate recipe')
        dump(out/'contract.json',dict(worker_pid=pid,source_sha256=sources,data_summary_sha256=sha(a.data/'summary.json'),acceptance_sha256=sha(a.acceptance/'summary.json'),
            source_band='native x in [0,floor(width/4))',query_domain='every original tile with coarse x>=cut+ceil(1um/mpp_x)',
            support_selection='first geometric-click-eligible source tile by fixed SHA order, independently for each class',groups=5,maximum='5+2',budgets=['3+2','1+1','3+1','5+1'],threshold=.5,
            GT_use='only predefined left support band for clicks; no query truth in candidate/query selection',no_performance_claim=True))
        all_tasks=[];coverage=[];inference=[];scoring=[];native_probes=[]
        for index,r in enumerate(records):
            folder=out/r['wsi_id'];folder.mkdir();w,h=r['native_wh'];gw,gh=r['gt_wh'];sx,sy=w/gw,h/gh;mx,my=r['native_mpp'];cut=w//4;gap=math.ceil(1/mx)
            original_labels=np.load(a.data/r['wsi_id']/'labels.npy',mmap_mode='r')
            # Copy only the preregistered source band; downstream point generation never sees query labels.
            columns=math.floor(cut/sx);support=np.asarray(original_labels[:,:columns]).copy();del original_labels
            np.save(folder/'support_band_labels.npy',support,allow_pickle=False)
            tiles=[json.loads(line) for line in Path(r['grid']).open()]
            source_tiles=[t for t in tiles if t['native_boxes'][2][0]+t['native_boxes'][2][2]<=cut]
            query_tiles=[t for t in tiles if t['native_boxes'][2][0]>=cut+gap]
            require(source_tiles and query_tiles,'nonempty disjoint physical domains')
            require(max(t['native_boxes'][2][0]+t['native_boxes'][2][2] for t in source_tiles)+gap<=min(t['native_boxes'][2][0] for t in query_tiles),'support/query full4x interval separation')
            querypath=folder/'remote_grid.jsonl'
            querypath.write_text(''.join(json.dumps(t)+'\n' for t in query_tiles))
            candidates=[];available={target:[] for target in range(12) if target!=2};fw,fh=r['geometry']['fine_native_wh'];mpp=[fw*mx/512,fh*my/512]
            for t in source_tiles:
                labels=labels_for_box(support,t['native_boxes'][0],(sx,sy));inside=interior_labels(labels,mpp);counts=Counter(map(int,inside.ravel()));valid=sum(n for code,n in counts.items() if code!=255)
                legal=[c for c in available if counts[c]>=5 and valid-counts[c]>=2]
                candidates.append(dict(tile=t['number'],eligible_targets=legal,interior_counts={str(c):n for c,n in counts.items()}))
                for c in legal:available[c].append(t)
            dump(folder/'source_candidate_geometry.json',candidates)
            native=pyvips.Image.new_from_file(r['source_wsi'],access='random');saved={}
            for target in available:
                options=available[target]
                coverage.append(dict(wsi_id=r['wsi_id'],target=target,source_candidate_tiles=len(source_tiles),legal_source_tiles=len(options),status='available' if options else 'no_5positive_2negative_source_interior',support_GT_columns=columns,source_cut_native=cut,remote_query_tiles=len(query_tiles)))
                if not options:continue
                chosen=min(options,key=lambda t:hashlib.sha256(f'WSI-source-band-v1|{r["wsi_id"]}|{target}|{t["number"]}'.encode()).hexdigest())
                number=chosen['number'];box=chosen['native_boxes'][0];labels=labels_for_box(support,box,(sx,sy));tileid=f'{r["wsi_id"]}_support_{number:06d}'
                if number not in saved:
                    sourcefolder=folder/tileid;sourcefolder.mkdir();np.save(sourcefolder/'labels.npy',labels,allow_pickle=False);images=[]
                    for scale,b in enumerate(chosen['native_boxes']):
                        crop=native.crop(*b).extract_band(0,n=3)
                        rgb=np.frombuffer(crop.write_to_memory(),dtype=np.uint8).reshape(b[3],b[2],3)
                        image=Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)
                        p=sourcefolder/f'image_scale{scale}.png';image.save(p)
                        require(np.array_equal(np.asarray(Image.open(p)),np.asarray(image)),'actual support RGB disk roundtrip')
                        images.append(dict(path=str(p),sha256=sha(p),native_box=b))
                    saved[number]=dict(support_id=tileid,images=images,labels=str(sourcefolder/'labels.npy'),labels_sha256=sha(sourcefolder/'labels.npy'),labels_array_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),mpp=mpp)
                    native_probes.append(dict(wsi_id=r['wsi_id'],tile=number,**saved[number]))
                for group in range(5):
                    seed=20261025+index*100000+target*1000+group
                    clicks,audit=representative_clicks(labels,target,mpp,5,2,np.random.default_rng(seed),dataset='crc')
                    for sign in ('positive','negative'):
                        xy=np.asarray(clicks[sign+'_xy']);px=np.floor(xy*512).astype(int);values=labels[px[:,1],px[:,0]]
                        require(bool(np.all(values==target)) if sign=='positive' else bool(np.all((values!=target)&(values!=255))), 'actual support click signs')
                    native_points={sign+'_native_xy':[[box[0]+p[0]*box[2],box[1]+p[1]*box[3]] for p in clicks[sign+'_xy']] for sign in ('positive','negative')}
                    all_tasks.append(dict(wsi_id=r['wsi_id'],support_id=tileid,source_tile=number,target=target,prompt_group=group,click_seed=seed,gt_simulated_source_clicks=True,positive_xy=clicks['positive_xy'],negative_xy=clicks['negative_xy'],click_audit=audit,**native_points))
            inference.append(dict(wsi_id=r['wsi_id'],source_wsi=r['source_wsi'],native_wh=r['native_wh'],native_mpp=r['native_mpp'],grid=str(querypath),grid_sha256=sha(querypath),source_cut_native=cut,gap_native=gap,query_tiles=len(query_tiles),support_inputs=[{k:v for k,v in s.items() if k not in ('labels','labels_sha256','labels_array_sha256')} for s in saved.values()]))
            scoring.append(dict(wsi_id=r['wsi_id'],labels=str(a.data/r['wsi_id']/'labels.npy'),GT_to_native_affine=r['GT_to_native_affine'],original_valid4x_bounds=r['geometry']['valid_fine_bounds_xyxy'],source_cut_native=cut,gap_native=gap,query_grid=str(querypath)))
            dump(out/'status.json',dict(status='preparing_source_only_prompts',worker_pid=pid,slides=index+1,total=len(records),tasks=len(all_tasks)))
            print(json.dumps(dict(slide=r['wsi_id'],source_tiles=len(source_tiles),query_tiles=len(query_tiles),available_targets=sum(bool(v) for v in available.values()),tasks=len(all_tasks))),flush=True)
        dump(out/'support_tasks_maximum.json',all_tasks);dump(out/'support_inputs_private.json',native_probes);dump(out/'coverage.json',coverage);dump(out/'query_inputs.json',inference);dump(out/'scoring_private.json',scoring)
        result=dict(status='prepared_support_query_protocol',worker_pid=pid,source_gate=audited['gate'],slides=len(records),tasks=len(all_tasks),source_sha256=sources,
            class_slide_pairs_available=sum(c['status']=='available' for c in coverage),class_slide_pairs_unavailable=sum(c['status']!='available' for c in coverage),
            query_tiles=sum(r['query_tiles'] for r in inference),output_sha256={p.name:sha(p) for p in out.glob('*.json') if p.name!='status.json'},total_seconds=time.perf_counter()-started,
            remaining=['Independent support label/coordinate/grid permission acceptance','Actual cell/encoder/model cache gates','Full WSI prediction and remote scoring','GT connected component/prompt censorship/distance scoring not yet implemented'],
            scope='Same-WSI fixed geometric source band and strict remote context-disjoint domain; not whole tissue coverage or model quality')
        dump(out/'summary.json',result);dump(out/'status.json',dict(status=result['status'],worker_pid=pid,slides=len(records)));print(json.dumps(dict(status=result['status'],tasks=len(all_tasks),seconds=result['total_seconds'])),flush=True)
    except Exception:
        dump(out/'status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('data','acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
