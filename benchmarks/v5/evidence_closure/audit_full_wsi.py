"""Independent band-wise float64 replay of a complete immutable WSI stream."""
import argparse
from collections import defaultdict
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import time
import traceback

import pyvips
import numpy as np
from PIL import Image
import torch

from .audit_heldout_local import read,sha,require
from benchmarks.v5.wsi_global.cache_gate import load_cache,raw_images
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def tree_error(a,b):
    if torch.is_tensor(a):
        require(torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,'encoded tensor schema')
        if not a.is_floating_point():require(torch.equal(a,b),'encoded integer identity');return 0.
        require(torch.isfinite(a).all() and torch.isfinite(b).all(),'finite encoded cache')
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        require(set(a)==set(b),'encoded dictionary schema');return max([tree_error(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),'encoded sequence schema');return max([tree_error(x,y) for x,y in zip(a,b)]+[0.])
    require(a==b,'encoded scalar identity');return 0.


def axis_counts(starts,length,left,right):
    change=np.zeros(right-left+1,np.int64)
    for x in starts:change[x-left]+=1;change[x-left+length]-=1
    return np.cumsum(change)[:-1]


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.time();torch.set_num_threads(4)
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid))
    try:
        require(not torch.cuda.is_available(),'CPU audit must hide CUDA')
        contract=read(a.run/'contract.json');done=read(a.run/'summary.json');complete=read(a.run/'cache_complete.json')
        require(done['status']=='complete_stream_execution' and not done['gate'] and done['tiles']==3050,'full completed stream')
        require(a.run.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'),'original stream exit')
        for p,value in contract['source_sha256'].items():require(sha(p)==value,'original execution source identity')
        plan=read(a.plan/'summary.json');require(sha(str(a.plan/'slides_private.json'))==plan['slides_sha256']==contract['plan_sha256'],'plan identity')
        record=read(a.plan/'slides_private.json')[contract['slide_index']]
        require(record['slide_id']==contract['slide_id'] and sha(record['grid'])==record['grid_sha256']==contract['grid_sha256'],'slide/grid binding')
        tiles=[json.loads(line) for line in Path(record['grid']).open()]
        entries=[json.loads(line) for line in (a.run/'cache_manifest.jsonl').open()]
        require(sha(str(a.run/'cache_manifest.jsonl'))==complete['manifest_sha256'] and len(entries)==len(tiles)==3050,'all3050 cache identities')
        boxes=[t['native_boxes'][0] for t in tiles];fw,fh=record['geometry']['fine_native_wh']
        xs=sorted({b[0] for b in boxes});ys=sorted({b[1] for b in boxes})
        require(len(xs)*len(ys)==3050 and [(b[0],b[1]) for b in boxes]==[(x,y) for y in ys for x in xs],'complete row-major Cartesian grid')
        for i,(entry,tile) in enumerate(zip(entries,tiles)):
            identity=entry['identity'];geometry=identity['geometry']
            require(entry['number']==i and identity['slide_id']==record['slide_id'],'cache row identity')
            require(geometry['native_boxes']==tile['native_boxes'] and geometry['native_mpp']==contract['native_mpp'] and geometry['padding'] is False,'physical input footprint')
            for box in geometry['native_boxes']:
                x,y,w,h=box;require(min(x,y)>=0 and min(w,h)>0 and x+w<=record['width'] and y+h<=record['height'],'unfabricated complete context')
        require(sum(e['bytes'] for e in entries)==done['cache_bytes']==complete['cache_bytes'],'cache byte conservation')
        require(sha(contract['checkpoint'])==contract['checkpoint_sha256'],'selected model bytes')
        require(sha(str(a.development/'contract.json'))==contract['development_protocol_sha256'],'support development identity')
        signature=dict(source_sha256=sha(__file__),original_contract_sha256=sha(str(a.run/'contract.json')),
            original_summary_sha256=sha(str(a.run/'summary.json')),grid_sha256=contract['grid_sha256'],checkpoint_sha256=contract['checkpoint_sha256'])
        if not a.gate:
            require(a.verified_gate is not None,'actual representative gate required')
            gate=read(a.verified_gate/'summary.json');require(gate['status']=='passed_full_WSI_audit_gate' and gate['signature']==signature,'same full-audit gate identity')
        central=[y*len(xs)+x for y in (len(ys)//2,len(ys)//2+1) for x in (len(xs)//2,len(xs)//2+1)]
        corner=[y*len(xs)+x for y in (len(ys)-2,len(ys)-1) for x in (len(xs)-2,len(xs)-1)]
        windows=[('central',central),('last_overlap',corner)] if a.gate else [('full',list(range(3050)))]
        probe_ids=sorted({len(entries)//2,len(entries)-1} if a.gate else
            {0,len(entries)-1,*[y*len(xs)+x for x,y in [(len(xs)//4,len(ys)//4),(len(xs)//2,len(ys)//2),(3*len(xs)//4,3*len(ys)//4),(len(xs)//4,3*len(ys)//4),(3*len(xs)//4,len(ys)//4)]]})
        needed=sorted({i for _,indices in windows for i in indices}|set(probe_ids))
        dump('run.json',dict(worker_pid=pid,gate=a.gate,signature=signature,windows=[dict(name=n,tiles=len(ix)) for n,ix in windows],
            native_probe_ids=probe_ids,device='cpu',threads=4,query_GT_used=False,cell_detection_performed=False,
            scope='Image-only CPU fixed CRC support; independent probability/stitch validation, not TLS accuracy or GPU efficiency'))
        for index,i in enumerate(needed):
            entry=entries[i];require(Path(entry['path']).stat().st_size==entry['bytes'] and sha(entry['path'])==entry['sha256'],'actual cache file SHA')
            if index%100==0:dump('status.json',dict(status='hashing_cache',worker_pid=pid,completed=index+1,total=len(needed)))
        model=load('fewclick_m3d_image',Path(contract['checkpoint']),'cpu',contract['development_protocol_sha256'])
        require(model.prior['cell_mode']=='none','placeholders only for image-only model')
        encode_count=[0];original=model.core.encode_image
        def counted(raw):encode_count[0]+=1;return original(raw)
        model.core.encode_image=counted
        data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
        raw,_,_=data.materialize(nested(maximum,'1+1'),'cpu',True,True)
        with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(raw))
        slide=StrictSlide(record['provenance']['wsi_path']);native_probes=[]
        try:
            require([slide.width,slide.height]==[record['width'],record['height']] and [slide.mx,slide.my]==contract['native_mpp'],'current actual WSI header')
            with torch.inference_mode():
                for i in probe_ids:
                    images,geometry=slide.fields(tiles[i]['centre_level0'])
                    require(geometry==entries[i]['identity']['geometry'],'native geometry replay')
                    require([hashlib.sha256(im.tobytes()).hexdigest() for im in images]==entries[i]['identity']['RGB_array_sha256'],'native RGB hash replay')
                    fresh=model.core.encode_image(raw_images(images));cached=load_cache(Path(entries[i]['path']),contract['checkpoint_sha256'],entries[i]['identity'])
                    error=tree_error(fresh,cached);require(error<=2e-6,'native/cache encoded tensor replay')
                    for scale,im in enumerate(images):Image.fromarray(im).save(a.output/f'native_{i:06d}_scale{scale}.png')
                    native_probes.append(dict(tile=i,encoded_max_error=error,fine_white_fraction=float((images[0].min(2)>240).mean()),fine_std=float(images[0].std())))
        finally:slide.close()
        require(any(p['fine_std']>1 and p['fine_white_fraction']<.95 for p in native_probes),'representative native probes include nonconstant tissue input')
        before=encode_count[0];passes=[];probe_replays=[]
        full_left,full_top,full_right,full_bottom=contract['bounds_level0'];full_shape=(full_bottom-full_top,full_right-full_left)
        for budget in ('1+1','3+2'):
            parent=next(p for p in done['passes'] if p['budget']==budget)
            require(read(a.run/f'pass_complete_{budget}.json')==parent,'pass summary identity')
            arrays={}
            for name in ('probability','mask','overlap'):
                info=parent['outputs'][name]
                if not a.gate:require(sha(info['path'])==info['sha256'],'full output file SHA')
                arrays[name]=np.load(info['path'],mmap_mode='r',allow_pickle=False)
                require(arrays[name].shape==full_shape,'full output native canvas')
            require(arrays['probability'].dtype==np.float32 and arrays['mask'].dtype==np.bool_ and arrays['overlap'].dtype==np.uint16,'output data types')
            _,prompts,_=data.materialize(nested(maximum,budget),'cpu',True,True)
            with torch.inference_mode():task=model.core.support_task(source,prompts)
            for window,indices in windows:
                by_y=defaultdict(list)
                for i in indices:by_y[boxes[i][1]].append(i)
                wx=sorted({boxes[i][0] for i in indices});wy=sorted(by_y)
                left,top,right,bottom=min(wx),min(wy),max(wx)+fw,max(wy)+fh
                xc=axis_counts(wx,fw,left,right);yc=axis_counts(wy,fh,top,bottom)
                edges=sorted(set(wy)|{y+fh for y in wy});active={};pixels=weighted=decoded=roundoff_masks=0;max_error=0.;max_overlap=0
                with torch.inference_mode():
                    for band,(y0,y1) in enumerate(zip(edges,edges[1:])):
                        needed_y=[y for y in wy if y<y1 and y+fh>y0]
                        for y in list(active):
                            if y not in needed_y:del active[y]
                        for y in needed_y:
                            if y in active:continue
                            row=[]
                            for i in by_y[y]:
                                entry=entries[i];encoded=load_cache(Path(entry['path']),contract['checkpoint_sha256'],entry['identity'])
                                result,maps=model.core.decode(encoded,task,prompts,False);require(not maps.count_nonzero(),'no query heatmaps')
                                p=result['pixel_probability'][0].numpy();require(np.isfinite(p).all() and p.min()>=0 and p.max()<=1,'decoded probability range')
                                original_probe=a.run/f'probe_{budget}_{i:06d}.npy'
                                if original_probe.exists():
                                    error=float(np.abs(np.load(original_probe,allow_pickle=False)-p).max());require(error<=1e-7,'original saved512 probability replay')
                                    probe_replays.append(dict(tile=i,budget=budget,maximum_error=error))
                                if i in probe_ids:np.save(a.output/f'probability_{budget}_{i:06d}.npy',p,allow_pickle=False)
                                native=np.asarray(Image.fromarray(p.astype(np.float32)).resize((fw,fh),Image.Resampling.BILINEAR))
                                row.append((boxes[i][0],native));decoded+=1
                            active[y]=row
                        total=np.zeros((y1-y0,right-left),np.float64)
                        for y,row in active.items():
                            for x,p in row:total[:,x-left:x-left+fw]+=p[y0-y:y1-y]
                        count=(yc[y0-top:y1-top,None]*xc[None,:]).astype(np.uint16)
                        require((count>0).all(),'independent complete Cartesian coverage')
                        expected=(total/count).astype(np.float32)
                        region=(slice(y0-full_top,y1-full_top),slice(left-full_left,right-full_left))
                        actual=arrays['probability'][region];mask=arrays['mask'][region];overlap=arrays['overlap'][region]
                        require(np.isfinite(actual).all() and actual.min()>=0 and actual.max()<=1,'saved output finite/range')
                        error=float(np.abs(actual-expected).max());require(error<=2e-7,'independent full probability stitch')
                        require(np.array_equal(overlap,count),'every pixel overlap count')
                        require(np.array_equal(mask,actual>=.5),'every saved threshold mask pixel')
                        mismatch=mask!=(expected>=.5)
                        require(not mismatch.any() or np.max(np.abs(expected[mismatch]-.5))<=2e-7,'independent mask mismatch outside float rounding band')
                        roundoff_masks+=int(mismatch.sum());max_error=max(max_error,error);pixels+=actual.size;weighted+=int(count.sum(dtype=np.uint64));max_overlap=max(max_overlap,int(count.max()))
                        for array in arrays.values():array._mmap.madvise(mmap.MADV_DONTNEED)
                        dump('status.json',dict(status='independent_full_pixel_replay',worker_pid=pid,budget=budget,window=window,band=band+1,total_bands=len(edges)-1,decoded_tiles=decoded,pixels=pixels))
                        if band%10==0:print(json.dumps(dict(budget=budget,window=window,band=band+1,pixels=pixels)),flush=True)
                require(decoded==len(indices) and pixels==(right-left)*(bottom-top) and weighted==len(indices)*fw*fh,'independent area/weight conservation')
                if not a.gate:
                    require(pixels==parent['coverage']['covered_pixels'] and weighted==parent['coverage']['weighted_pixels'] and max_overlap==parent['coverage']['maximum_overlap'],'original coverage recount')
                passes.append(dict(budget=budget,window=window,tiles=decoded,pixels=pixels,weighted_pixels=weighted,maximum_overlap=max_overlap,
                    maximum_probability_error=max_error,mask_exact_to_saved_probability=True,independent_float64_threshold_roundoff_disagreements=roundoff_masks,bounds=[left,top,right,bottom]))
        require(encode_count[0]==before,'independent online replay never encodes query')
        result=dict(status='passed_full_WSI_audit_gate' if a.gate else 'accepted_full_WSI_probability_replay_qualified',worker_pid=pid,gate=a.gate,signature=signature,
            cache_files_hashed=len(needed),native_probes=native_probes,original512_probe_replays=probe_replays,passes=passes,
            query_encode_calls_during_decode=0,query_prompt_maps_zero=True,seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            area_fraction_of_rectangular_slide=record['geometry']['covered_area_fraction'],
            limits=['Full planned valid4x-context domain only; edges excluded, not full tissue coverage',
                    'Native pixel reads/encode verified only fixed probes; all cache files hashed/identified and all tile probabilities decoded',
                    'Native-grid output interpolates512 probabilities, adds no spatial detail','Image-only CPU fixedCRC support mechanical task; no TLS semantic accuracy/full-cell/GPU efficiency',
                    'Independent float64 averaging may differ by rounding at .5; actual saved mask threshold exact and all such differences counted'])
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','plan','development','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);run(p.parse_args())
