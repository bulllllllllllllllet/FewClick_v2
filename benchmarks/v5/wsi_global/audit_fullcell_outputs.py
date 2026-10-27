"""Fresh GPU5 full-cell encoder/zero-cell and complete binary WSI output audit."""
import argparse
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
import torch

from .probability_replay import replay,require
from .audit_fullcell_inputs import geometric_features
from .fullcell_gpu_gate import transfer
from benchmarks.v5.evidence_closure.audit_full_wsi import tree_error
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4)
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='loading',worker_pid=pid))
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='5' and torch.cuda.is_available(),'actual allowed GPU5 required')
        done=read(a.source/'summary.json');contract=read(a.source/'contract.json');complete=read(a.source/'cache_complete.json')
        require(done['status']=='complete_fullcell_GPU_WSI_execution' and done['tiles']==3050 and Path(str(a.source)+'.log').read_text().rstrip().endswith('EXIT_CODE=0'),'full producer completed before any GPU audit')
        require(done['signature']==contract['signature'],'producer result signature')
        for p,d in contract['signature']['source_sha256'].items():require(sha(p)==d,'producer source unchanged')
        generic=read(a.replay_gate/'summary.json');require(generic['status']=='passed_actual_float32_probability_replay_gate' and generic['helper_sha256']==sha('benchmarks/v5/wsi_global/probability_replay.py'),'actual independent FP32 replay core gate')
        signature=dict(auditor_sha256=sha(__file__),helper_sha256=generic['helper_sha256'],cell_geometry_helper_sha256=sha('benchmarks/v5/wsi_global/audit_fullcell_inputs.py'),producer_contract_sha256=sha(a.source/'contract.json'),producer_summary_sha256=sha(a.source/'summary.json'))
        if not a.gate:
            require(a.verified_gpu_gate is not None,'actual same-source native GPU audit gate required');gate=read(a.verified_gpu_gate/'summary.json');require(gate['status']=='passed_fullcell_GPU_probability_audit_gate' and gate['signature']==signature,'actual native/zero/overlap gate identity')
        entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()];require(len(entries)==3050 and sha(a.source/'cache_manifest.jsonl')==complete['manifest_sha256'],'all3050 encoded cache identities')
        record=read(a.plan/'slides_private.json')[0];require(sha(a.plan/'slides_private.json')==contract['signature']['plan_sha256'] and sha(record['grid'])==contract['signature']['grid_sha256'],'original complete true4x plan')
        tiles=[json.loads(line) for line in Path(record['grid']).open()];boxes=[t['native_boxes'][0] for t in tiles];xs=sorted({b[0] for b in boxes});ys=sorted({b[1] for b in boxes})
        require(len(xs)*len(ys)==3050 and [(b[0],b[1]) for b in boxes]==[(x,y) for y in ys for x in xs],'complete row-major grid')
        zero=[]
        for i,(e,t) in enumerate(zip(entries,tiles)):
            require(e['tile']==i and e['number']==i and e['identity']['geometry']['native_boxes']==t['native_boxes'] and e['identity']['geometry']['native_mpp']==record['geometry']['native_mpp'] and e['identity']['geometry']['padding'] is False,'all cached actual input footprints')
            for c in e['identity']['cell_records']:
                if c['details']['total_cell_count']==0:zero.append(dict(tile=i,scale=c['scale']))
        require(zero==done['zero_cell_cases']==complete['zero_cell_cases'],'all producer zero-cell records retained')
        zero_probes={next(z['tile'] for z in zero if z['scale']==scale) for scale in (0,1,2) if any(z['scale']==scale for z in zero)}
        entirely_empty=next((e['tile'] for e in entries if all(c['details']['total_cell_count']==0 for c in e['identity']['cell_records'])),None)
        if entirely_empty is not None:zero_probes.add(entirely_empty)
        central=[y*len(xs)+x for y in (len(ys)//2,len(ys)//2+1) for x in (len(xs)//2,len(xs)//2+1)];last=[y*len(xs)+x for y in (len(ys)-2,len(ys)-1) for x in (len(xs)-2,len(xs)-1)]
        zero_windows=[]
        for i in sorted(zero_probes):
            x,y,w,h=boxes[i]
            # Include every contributing neighbour if the zero tile lies in a final overlap row/column.
            cluster=[j for j,(bx,by,bw,bh) in enumerate(boxes) if bx<x+w and bx+bw>x and by<y+h and by+bh>y]
            zero_windows.append((f'actual_zero_{i}',cluster))
        windows=[('central',central),('last_overlap',last)]+zero_windows if a.gate else [('full',list(range(3050)))]
        probe_ids=sorted({0,1525,3049,*zero_probes} if a.gate else {0,762,787,1525,2262,2287,3049,*zero_probes})
        needed=sorted({i for _,indices in windows for i in indices}|set(probe_ids))
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,probes=probe_ids,windows=[dict(name=n,tiles=len(ix)) for n,ix in windows],device='GPU5 FP32',query_GT_access=False,detector_rerun=False))
        for n,i in enumerate(needed):
            e=entries[i];require(Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256'],'all required encoded cache bytes')
            if n%100==0:dump('status.json',dict(status='hashing_encoded_cache',worker_pid=pid,completed=n+1,total=len(needed)))
        best=read(a.training/'best_checkpoint.json');require(sha(best['path'])==best['checkpoint_sha256']==contract['signature']['checkpoint_sha256'],'actual selected fullcell weights')
        require(sha(a.development/'contract.json')==contract['signature']['development_protocol_sha256'],'source support protocol')
        original_gpu=read(a.source.parent/'fullcell_gpu_gate_1027_150000/run.json')
        require(torch.__version__==original_gpu['torch_version'] and torch.version.cuda==original_gpu['cuda_version'] and torch.backends.cuda.matmul.allow_tf32==original_gpu['matmul_allow_tf32'] and torch.backends.cudnn.allow_tf32==original_gpu['cudnn_allow_tf32'],'same original GPU numeric runtime')
        device=torch.device('cuda:0');model=load('fewclick_m3d_xcell',Path(best['path']),device,contract['signature']['development_protocol_sha256']);require(model.prior['cell_mode']=='xcell','full cell branch only')
        calls=[0];original=model.core.encode_image
        def encode(raw):calls[0]+=1;return original(raw)
        model.core.encode_image=encode;data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
        raw,_,_=data.materialize(nested(maximum,'1+1'),device,True,True)
        with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(raw))
        slide=StrictSlide(record['provenance']['wsi_path']);probes=[]
        try:
            with torch.inference_mode():
                for i in probe_ids:
                    e=entries[i];images,geometry=slide.fields(tiles[i]['centre_level0']);require(geometry==e['identity']['geometry'] and [hashlib.sha256(im.tobytes()).hexdigest() for im in images]==e['identity']['RGB_array_sha256'],'fresh actual WSI RGB/geometry')
                    raw={};totals=[];zeros=[]
                    for c,box,image in zip(e['identity']['cell_records'],geometry['native_boxes'],images):
                        require(sha(c['path'])==c['sha256'],'fresh probe cell bytes');native=slide.read(box);require(hashlib.sha256(native.tobytes()).hexdigest()==c['native_RGB_sha256'],'fresh native cell source RGB')
                        with np.load(c['path'],allow_pickle=False) as z:
                            features=z['features'];expected,total=geometric_features(z['instances'],c['details']['detection_mpp_xy'],z['reg']);require(total==int(z['total']) and np.array_equal(features,expected),'independent true cells and raw74 native probe')
                        scale=c['scale'];suffix='' if scale==0 else f'_{scale}';raw['image'+suffix]=he_to_tensor(image.copy())[None].to(device);raw['cells'+suffix]=torch.from_numpy(features)[None].to(device);raw['cell_valid'+suffix]=torch.ones(1,len(features),dtype=torch.bool,device=device)
                        raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([total],dtype=torch.float32,device=device);totals.append(total)
                        if total==0:zeros.append(scale)
                    density=[];hook=model.core.cell.register_forward_hook(lambda m,args,out:density.append(float(out['region_cell_count'].sum())))
                    try:fresh=model.core.encode_image(raw)
                    finally:hook.remove()
                    require(len(density)==3 and all(math.isclose(x,y,abs_tol=1e-3,rel_tol=1e-6) for x,y in zip(density,totals)),'true distributed regional counts, including actual zeros')
                    payload=torch.load(e['path'],map_location='cpu',weights_only=False);cached=transfer(payload['encoded'],device);require(payload['identity']==e['identity'] and payload['checkpoint_sha256']==best['checkpoint_sha256'],'native cache method/input')
                    error=tree_error(fresh,cached);require(error<=2e-6,'fresh fullcell GPU encoder/cache parity');checks=[]
                    for budget in ('1+1','3+2'):
                        _,p,_=data.materialize(nested(maximum,budget),device,True,True);task=model.core.support_task(source,p);aout,am=model.core.decode(fresh,task,p,False);bout,bm=model.core.decode(cached,task,p,False)
                        require(not am.count_nonzero() and not bm.count_nonzero(),'native actual zero query maps');ap=aout['pixel_probability'][0];bp=bout['pixel_probability'][0];delta=float((ap-bp).abs().max());require(torch.isfinite(ap).all() and delta<=2e-6 and torch.equal(ap>=.5,bp>=.5),'native empty/full cell query probability/cache mask')
                        checks.append(dict(budget=budget,maximum_error=delta))
                    probes.append(dict(tile=i,encoded_maximum_error=error,true_counts=totals,regional_count_sums=density,actual_empty_scales=zeros,probability_checks=checks))
        finally:slide.close()
        before=calls[0];passes=[];saved=[];bounds=contract['bounds_level0'];shape=(bounds[3]-bounds[1],bounds[2]-bounds[0])
        for budget in ('1+1','3+2'):
            parent=next(p for p in done['passes'] if p['budget']==budget);require(read(a.source/f'pass_complete_{budget}.json')==parent,'producer complete pass identity');arrays={}
            for name in ('probability','mask','overlap'):
                p=parent['outputs'][name]
                if not a.gate:require(sha(p['path'])==p['sha256'],'all six complete output bytes')
                arrays[name]=np.load(p['path'],mmap_mode='r');require(arrays[name].shape==shape,'original native output extent')
            require(arrays['probability'].dtype==np.float32 and arrays['mask'].dtype==np.bool_ and arrays['overlap'].dtype==np.uint16,'actual probability/mask/count dtype')
            _,p,_=data.materialize(nested(maximum,budget),device,True,True)
            with torch.inference_mode():task=model.core.support_task(source,p)
            def decode(i):
                e=entries[i];payload=torch.load(e['path'],map_location='cpu',weights_only=False);require(payload['identity']==e['identity'] and payload['checkpoint_sha256']==best['checkpoint_sha256'],'every actual GPU decoded cache identity')
                with torch.inference_mode():out,maps=model.core.decode(transfer(payload['encoded'],device),task,p,False)
                require(not maps.count_nonzero(),'no online source-coordinate query maps');prob=out['pixel_probability'][0].cpu().numpy()
                original_probe=a.source/f'probe_{budget}_{i:06d}.npy'
                if original_probe.exists():
                    old=np.load(original_probe,allow_pickle=False);err=float(np.max(np.abs(prob-old)));require(err<=2e-6 and np.array_equal(prob>=.5,old>=.5),'original512 saved probability/mask');saved.append(dict(tile=i,budget=budget,maximum_error=err))
                return prob
            for name,indices in windows:
                result=replay(indices,boxes,decode,arrays,bounds,atol=2e-6,progress=lambda value:dump('status.json',dict(status='full_GPU_probability_replay',worker_pid=pid,budget=budget,window=name,**value)))
                if not a.gate:require(result['pixels']==parent['coverage']['covered_pixels'] and result['weighted_pixels']==parent['coverage']['weighted_pixels'] and result['maximum_overlap']==parent['coverage']['maximum_overlap'],'all native area/overlap conservation')
                passes.append(dict(budget=budget,window=name,**result));print(json.dumps(dict(budget=budget,window=name,pixels=result['pixels'],error=result['maximum_probability_error'])),flush=True)
        require(calls[0]==before and done['online_query_encode_calls']==done['online_detector_calls']==0,'no online detector or query image encoding')
        result=dict(status='passed_fullcell_GPU_probability_audit_gate' if a.gate else 'accepted_fullcell_full_WSI_probability_cache_qualified',worker_pid=pid,signature=signature,gate=a.gate,native_probes=probes,passes=passes,original512_probes=saved,zero_cell_cases=zero,all_three_scales_empty_tile=entirely_empty,cache_files_hashed=len(needed),seconds=time.perf_counter()-start,
            remaining='All9150 saved-cell geometry audit/aggregation must also pass; valid4x native domain not full tissue or TLS semantic accuracy, no detection-history invariance claim')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','plan','development','training','replay-gate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gpu-gate',type=Path);run(p.parse_args())
