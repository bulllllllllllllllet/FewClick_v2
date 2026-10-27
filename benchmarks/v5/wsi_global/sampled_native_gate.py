"""Actual three-method remote decode/native-raster/sampled-raster equality gate."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import pyvips
import torch

from .native_remote_gate import raw,encode,decode,prompts
from .sampled_stripe import SampledStripe
from .stripe import StripeWriter
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,reason):
    if not test:
        raise ValueError(reason)


def compare(folder,records,native_wh,gt_wh):
    boxes=[r['native_box'] for r in records]
    bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
    native=StripeWriter(folder/'native',bounds,boxes[0][2:]);sampled=SampledStripe(folder/'sampled',bounds,boxes[0][2:],native_wh,gt_wh)
    for record in records:
        require(sha(record['path'])==record['sha256'],'frozen saved512 probability')
        probability=np.load(record['path'],allow_pickle=False)
        native.push(probability,record['native_box']);sampled.push(probability,record['native_box'])
    coverage=native.finish();selected=sampled.finish()
    w,h=native_wh;gw,gh=gt_wh;left,top,right,bottom=bounds
    gx=[x for x in range(gw) if x*w/gw>=left and (x+1)*w/gw<=right]
    gy=[y for y in range(gh) if y*h/gh>=top and (y+1)*h/gh<=bottom]
    require(selected['GT_cell_bounds']==[gx[0],gy[0],gx[-1]+1,gy[-1]+1],'independent complete original GT-cell footprint')
    # Recompute coordinate indices independently; producer uses array*(w/gw).
    # Use that fixed extent-affine multiplication order at nearest-pixel lookup.
    px=np.array([int(np.floor((x+.5)*(w/gw)))-left for x in gx],np.int64)
    py=np.array([int(np.floor((y+.5)*(h/gh)))-top for y in gy],np.int64)
    for row,y in enumerate(py):
        require(np.array_equal(sampled.prob[row],native.prob[y,px]),'every sampled probability equals finalized full-native FP32 mean')
        require(np.array_equal(sampled.mask[row],native.mask[y,px]) and np.array_equal(sampled.count[row],native.count[y,px]),'every sampled binary/overlap equals full-native projection')
    files={}
    for name in ('native','sampled'):
        files[name]={p.name:dict(path=str(p),sha256=sha(p)) for p in (folder/name).glob('*.npy')}
    del native,sampled
    return dict(native_coverage=coverage,sampled_coverage=selected,probability_error=0.,binary_disagreements=0,overlap_disagreements=0,files=files)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);pid=os.getpid();started=time.perf_counter()
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        require(not torch.cuda.is_available(),'actual CPU4 gate with CUDA hidden');torch.set_num_threads(4);pyvips.cache_set_max(0)
        sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/sampled_stripe.py'),Path('benchmarks/v5/wsi_global/stripe.py'),Path('benchmarks/v5/wsi_global/native_remote_gate.py'),Path('benchmarks/v5/transfer_comparison/prototypes.py')]}
        dump('contract.json',dict(worker_pid=pid,sources=sources,protocol_sha256=sha(a.root/'support_query_full_1027_173800/summary.json'),data_summary_sha256=sha(a.root/'main30_full_1027_171918/summary.json'),query_GT_access=False,selection='first mainCRC30 predeclared central2x2 and last2x2, target0 and10 group0, all4 budgets; all methods and both NoCell variants',scope='Exact GT-grid projection of unchanged full-native probability and threshold; not full30 quality/efficiency'))
        # Noninteger anisotropic scale, irregular last-row/column overlap, near .5.
        fixture=a.output/'anisotropic_fixture';fixture.mkdir();records=[]
        for i,(y,x) in enumerate((y,x) for y in (5,13,19) for x in (5,13,21,24)):
            p=fixture/f'tile{i:02d}.npy';np.save(p,((np.arange(63).reshape(7,9)+i*7)%101).astype(np.float32)/100)
            records.append(dict(path=str(p),sha256=sha(p),native_box=[x,y,8,8]))
        synthetic=compare(fixture,records,[37,29],[19,14]);dump('synthetic.json',synthetic)
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        row=read(a.root/'support_query_full_1027_173800/query_inputs.json')[0];sid=row['wsi_id']
        data=next(r for r in read(a.root/'main30_full_1027_171918/slides_private.json') if r['wsi_id']==sid)
        grid={t['number']:t for t in (json.loads(line) for line in Path(row['grid']).open())}
        windows=read(a.root/'semantic_raster_gate_1027_220000/summary.json')['results']
        require(len(windows)==2 and [g['window'] for g in windows]==['central','last_overlap'],'fixed native window gate')
        plans=[p for p in read(a.root/'support_query_full_1027_173800/support_tasks_maximum.json') if p['wsi_id']==sid and p['prompt_group']==0 and p['target'] in (0,10)]
        require(len(plans)==2,'both frozen source targets')
        methods=[('fewclick_m3d_image','nocell','interactive_cpu_gate_nocell_1028_000700'),('sam_ft','sam','interactive_cpu_gate_sam_ft_1028_000900'),('wsi_sam_ft','wsisam','interactive_cpu_gate_wsi_sam_ft_1028_000900')]
        outputs=[];results=[];total_native=total_gt=0
        for method,short,source_name in methods:
            source=a.root/source_name;accepted=a.root/f'interactive_cpu_accept_{short}_1028_001800';receipt=read(accepted/'summary.json')
            require(receipt['status']=='accepted_interactive_cache_fixture' and receipt['source_summary_sha256']==sha(source/'summary.json'),'independently accepted actual source caches')
            saved={e['identity']['key']:e for e in (json.loads(line) for line in (source/'cache_manifest.jsonl').open())}
            best=read(base/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
            require(best['checkpoint_sha256']==read(source/'summary.json')['signature']['weight_sha256'] and sha(best['path'])==best['checkpoint_sha256'],'same selected model weight')
            model=load(method,Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'))
            folder=a.output/method;folder.mkdir();supports={}
            for plan in plans:
                d=saved[plan['support_id']];require(sha(d['path'])==d['sha256'],'actual accepted source feature bytes')
                supports[plan['support_id']]=torch.load(d['path'],map_location='cpu',weights_only=False)['encoded']
            queries={};image=pyvips.Image.new_from_file(row['source_wsi'],access='random')
            view_count=3 if method=='fewclick_m3d_image' else 1 if method=='sam_ft' else 2
            with torch.inference_mode():
                for number in dict.fromkeys(n for g in windows for n in g['tiles']):
                    tile=grid[number];images=[]
                    require(tile['native_boxes'][2][0]>=row['source_cut_native']+row['gap_native'],'true remote 4x footprint')
                    for box in tile['native_boxes'][:view_count]:
                        crop=image.crop(*box).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(box[3],box[2],3)
                        images.append(np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)).copy())
                    value=encode(model,method,raw(images));p=folder/f'query{number:06d}.pth'
                    torch.save(dict(encoded=value,RGB_array_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],native_boxes=tile['native_boxes'][:view_count],checkpoint_sha256=best['checkpoint_sha256']),p)
                    queries[number]=dict(path=str(p),sha256=sha(p),native_box=tile['native_boxes'][0])
                calls=[0]
                def hook(module,args):
                    calls[0]+=1
                encoders=[e.backbone for e in model.core.encoders] if method=='fewclick_m3d_image' else [model.sam.image_encoder]
                handles=[m.register_forward_pre_hook(hook) for m in encoders]
                records={}
                for g in windows:
                    for number in g['tiles']:
                        d=queries[number];require(sha(d['path'])==d['sha256'],'frozen actual query cache')
                        query=torch.load(d['path'],map_location='cpu',weights_only=False)['encoded']
                        for plan in plans:
                            for budget in ('3+2','1+1','3+1','5+1'):
                                probabilities=decode(model,method,supports[plan['support_id']],query,prompts(plan,budget))
                                for variant,p in probabilities.items():
                                    key=(g['window'],plan['target'],budget,variant)
                                    path=folder/f'{g["window"]}_target{plan["target"]}_{budget}_{variant}_tile{number:06d}.npy'
                                    np.save(path,p.numpy(),allow_pickle=False)
                                    record=dict(path=str(path),sha256=sha(path),native_box=d['native_box']);records.setdefault(key,[]).append(record)
                                    outputs.append(dict(method=method,window=g['window'],target=plan['target'],budget=budget,variant=variant,tile=number,**record))
                    dump('status.json',dict(status='actual_cached_remote_decode',worker_pid=pid,method=method,window=g['window'],outputs=len(outputs),online_encoder_calls=calls[0]))
                require(calls[0]==0,'actual online encoder hooks zero for every task')
                for h in handles:
                    h.remove()
            for (window,target,budget,variant),items in records.items():
                result=compare(folder/f'{window}_target{target}_{budget}_{variant}',items,data['native_wh'],data['gt_wh'])
                result.update(method=method,window=window,target=target,budget=budget,variant=variant,online_encoder_calls=0)
                results.append(result);total_native+=result['native_coverage']['covered_pixels'];total_gt+=result['sampled_coverage']['GT_cells']
                dump('status.json',dict(status='compare_full_native_and_sampled',worker_pid=pid,method=method,completed_cases=len(results),total_cases=64))
            dump('results.json',results);dump('outputs.json',outputs)
            print(json.dumps(dict(method=method,cases=len(records),outputs=len(outputs),online_encoder_calls=calls[0])),flush=True)
            del model,supports,queries,query,value;gc.collect()
        require(len(results)==64 and len(outputs)==256,'all native model/variant/window/target/budget cases')
        summary=dict(status='passed_actual_interactive_native_GT_projection_gate',worker_pid=pid,sources=sources,cases=len(results),outputs=len(outputs),native_pixels=total_native,GT_cells=total_gt,all_probabilities_exact=True,all_masks_exact=True,all_overlap_counts_exact=True,online_query_encode_calls=0,query_GT_access=False,results_sha256=sha(a.output/'results.json'),outputs_sha256=sha(a.output/'outputs.json'),seconds=time.perf_counter()-started,remaining='Independent fresh projection acceptance and full main30 interactive streaming/decode/GT scoring; no quality result from this gate')
        dump('summary.json',summary);dump('status.json',dict(status=summary['status'],worker_pid=pid));print(json.dumps(summary),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
