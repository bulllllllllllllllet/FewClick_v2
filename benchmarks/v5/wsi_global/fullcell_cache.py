"""Full CRC30 native cells/CTP/XCell/three-view region-cache producer.

The detector and 74D feature recipe are unchanged. Native RGB is read once per
view and supplies both HE512 and prepare_variant. No query labels are read.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pyvips
import torch

from .cell_cache_inputs import native_views
from .audit_fullcell_inputs import geometric_features
from .completion import atomic_json
from .native_remote_gate import prompts
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector,CachedCenterCropper,DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models,WEIGHTS
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:
        raise ValueError(why)


def transfer(x,device):
    if isinstance(x,torch.Tensor):return x.detach().to(device)
    if isinstance(x,dict):return {k:transfer(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(transfer(v,device) for v in x)
    raise TypeError('unexpected encoded node '+str(type(x)))


def finite(x):
    if isinstance(x,torch.Tensor):require(torch.isfinite(x).all(),'finite actual fullcell encoded tensor')
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    else:
        for v in x:finite(v)


def append_view(raw,scale,image,arrays,device):
    """Assemble real74/true total inputs, including genuine zero-cell views."""
    features=arrays['features'];total=int(arrays['total']);n=len(features)
    require(image.shape==(512,512,3) and image.dtype==np.uint8,'native HE512 RGB')
    require(features.dtype==np.float32 and features.shape==(n,74) and np.isfinite(features).all() and n==min(255,total),'real74 retained features and uncapped true cell count')
    suffix='' if scale==0 else f'_{scale}'
    raw['image'+suffix]=he_to_tensor(image.copy())[None].to(device)
    raw['cells'+suffix]=torch.from_numpy(features.copy())[None].to(device)
    raw['cell_valid'+suffix]=torch.ones(1,n,dtype=torch.bool,device=device)
    raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([total],dtype=torch.float32,device=device)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);pid=os.getpid();start=time.perf_counter()
    def dump(n,d):atomic_json(a.output/n,d)
    try:
        torch.set_num_threads(4);pyvips.cache_set_max(0);torch.manual_seed(20261025);np.random.seed(20261025)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5') and torch.cuda.is_available(),'one permitted physical GPU required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
        root=a.root;inputs=root/'cell_native_inputs_full_1028_012700';accepted=read(root/'cell_native_inputs_acceptance_1028_023400/summary.json');input_summary=read(inputs/'summary.json')
        require(accepted['status']=='accepted_full30_fullcell_native_input_manifest' and accepted['source_summary_sha256']==sha(inputs/'summary.json'),'all20456 native inputs accepted')
        require(sha(inputs/'inputs.jsonl')==input_summary['input_manifest_sha256'],'complete native input manifest bytes')
        records=[json.loads(line) for line in (inputs/'inputs.jsonl').open()]
        if a.gate:
            keys={r['key'] for r in (json.loads(line) for line in (root/'cell_native_input_gate_1028_012500/inputs.jsonl').open())};records=[r for r in records if r['key'] in keys]
        require(len(records)==(8 if a.gate else 20456),'all fixed fullcell input slots')
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');training=base/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451';best=read(training/'best_checkpoint.json');done=read(training/'summary.json')
        require(done['status']=='complete' and done['step']==10000 and done['training_consumed_sha256']==sha(training/'training_consumed.jsonl')=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','same completed full-label40k supervision')
        require(sha(best['path'])==best['checkpoint_sha256'],'same development-selected fullcell model')
        sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/cell_cache_inputs.py'),Path(inspect.getfile(inspect.unwrap(prepare_variant))),Path(inspect.getfile(InstanSegDetector)),Path(inspect.getfile(load_feature_models)),Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py')]}
        weights={k:dict(path=str(v),sha256=sha(v)) for k,v in WEIGHTS.items() if k in ('ctranspath','xcell')};weights['instanseg']=dict(path=str(DEFAULT_INSTANSEG_CHECKPOINT),sha256=sha(DEFAULT_INSTANSEG_CHECKPOINT))
        signature=dict(sources=sources,weights=weights,checkpoint_sha256=best['checkpoint_sha256'],native_input_summary_sha256=sha(inputs/'summary.json'),input_acceptance_sha256=sha(root/'cell_native_inputs_acceptance_1028_023400/summary.json'),device=str(device),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cuda=torch.version.cuda,matmul_TF32=False,cudnn_TF32=False,seed=20261025,cell_order='manifest ascending then scale0/1/2; one serial RNG stream; no partition-invariance claim')
        if not a.gate:
            require(a.verified_gate is not None and a.gate_acceptance is not None,'actual same-GPU fullcell cache and independent cell-input gates required')
            g=read(a.verified_gate/'summary.json');ga=read(a.gate_acceptance/'summary.json')
            require(g['status']=='passed_main30_fullcell_cache_gate' and g['signature']==signature and ga['status']=='accepted_main30_fullcell_cache_gate' and ga['source_summary_sha256']==sha(a.verified_gate/'summary.json'),'same fullcell GPU recipe and independent gate receipt')
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,inputs=len(records),query_GT_access=False,method='fewclick_m3d_xcell',XCell_provenance='Internal training data excludes current CRC/TIGER per user; upstream CTP priors recorded separately',cell_labels='unknown0 only; no tissueGT nuclear mapping',failure_policy='stop preserve; no retry/resume/fallback'))
        model=load('fewclick_m3d_xcell',Path(best['path']),device,sha(base/'tasks_1025_125400/contract.json'));require(model.prior['cell_mode']=='xcell','full molecular/cell branch')
        detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device);ctp,xcell,compatibility=load_feature_models(device);cropper=CachedCenterCropper();dump('feature_compatibility.json',compatibility)
        timing=dict(native_read_seconds=0.,cell_prepare_seconds=0.,cell_write_seconds=0.,encode_seconds=0.,encoded_write_seconds=0.,hash_seconds=0.)
        current=None;slide=None;zero=[];true_total=retained=0;cache_bytes=0;gate_cpu={};gate_native={}
        with torch.inference_mode(),(a.output/'cache_manifest.jsonl').open('x') as manifest:
            for index,r in enumerate(records):
                for p,h in sources.items():require(sha(p)==h,'frozen fullcell source recipe')
                if current!=r['wsi_id']:
                    slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');current=r['wsi_id'];require([slide.width,slide.height]==r['native_wh'],'actual native slide header')
                folder=a.output/r['wsi_id']/r['key'].replace(':','_');folder.mkdir(parents=True);raw={};cell_records=[];HE_hashes=[]
                iterator=iter(native_views(slide,r))
                for scale in range(3):
                    tick=time.perf_counter();k,native,image,details=next(iterator);timing['native_read_seconds']+=time.perf_counter()-tick;require(k==scale,'fixed three-view order')
                    torch.cuda.synchronize();tick=time.perf_counter();arrays,cell_details=prepare_variant(native,r['native_mpp'],details['metadata'],detector,ctp,xcell,device,cropper=cropper);torch.cuda.synchronize();timing['cell_prepare_seconds']+=time.perf_counter()-tick
                    require(cell_details['detection_size']==details['detection_size'] and cell_details['detection_mpp_xy']==details['detection_mpp_xy'],'actual canonical size and physical pitch')
                    if a.gate:
                        expected,count=geometric_features(arrays['instances'],cell_details['detection_mpp_xy'],arrays['reg']);require(np.array_equal(expected,arrays['features']) and count==int(arrays['total']),'independent actual instance geometry/true count/74 assembly gate')
                    tick=time.perf_counter();path=folder/f'cells_scale{scale}.npz';np.savez_compressed(path,**arrays);timing['cell_write_seconds']+=time.perf_counter()-tick
                    if a.gate:
                        with np.load(path,allow_pickle=False) as z:require(all(np.array_equal(z[k],v) for k,v in arrays.items()),'all real saved cell arrays exact')
                    append_view(raw,scale,image,arrays,device);total=int(arrays['total']);true_total+=total;retained+=len(arrays['features'])
                    if total==0:zero.append(dict(key=r['key'],scale=scale))
                    tick=time.perf_counter();digest=sha(path);timing['hash_seconds']+=time.perf_counter()-tick
                    cell_records.append(dict(scale=scale,path=str(path),sha256=digest,native_RGB_sha256=hashlib.sha256(native.tobytes()).hexdigest(),details=cell_details));HE_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    dump('status.json',dict(status='preparing_CRC30_actual_cells',worker_pid=pid,completed_inputs=index,total_inputs=len(records),key=r['key'],scale=scale,true_cells=total))
                torch.cuda.synchronize();tick=time.perf_counter();encoded=model.core.encode_image(raw);torch.cuda.synchronize();timing['encode_seconds']+=time.perf_counter()-tick;finite(encoded)
                identity=dict(wsi_id=r['wsi_id'],kind=r['kind'],key=r['key'],geometry=dict(native_boxes=r['native_boxes'],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),RGB_array_sha256=HE_hashes,cell_records=cell_records)
                path=folder/'encoded.pth';tick=time.perf_counter();torch.save(dict(encoded=transfer(encoded,'cpu'),identity=identity,checkpoint_sha256=best['checkpoint_sha256'],method='fewclick_m3d_xcell'),path);timing['encoded_write_seconds']+=time.perf_counter()-tick
                if index==0:
                    estimate=(path.stat().st_size+sum(Path(c['path']).stat().st_size for c in cell_records))*len(records)
                    require(shutil.disk_usage(a.output).free>1.5*estimate,'actual fullcell NPZ/encoded disk headroom');dump('size_estimate.json',dict(first_input_estimated_all_bytes=estimate,inputs=len(records)))
                tick=time.perf_counter();digest=sha(path);timing['hash_seconds']+=time.perf_counter()-tick;cache_bytes+=path.stat().st_size
                manifest.write(json.dumps(dict(index=r['index'],path=str(path),sha256=digest,bytes=path.stat().st_size,identity=identity))+'\n');manifest.flush()
                if a.gate:gate_native[r['key']]=encoded;gate_cpu[r['key']]=torch.load(path,map_location='cpu',weights_only=False)['encoded']
                dump('status.json',dict(status='encoding_CRC30_actual_fullcell_cache',worker_pid=pid,completed_inputs=index+1,total_inputs=len(records),key=r['key'],seconds=time.perf_counter()-start));print(json.dumps(dict(input=index+1,total=len(records),key=r['key'])),flush=True)
                del raw,encoded,arrays;gc.collect()
        checks=[]
        if a.gate:
            available={r['key']:r for r in records};plans=read(root/'support_query_full_1027_173800/support_tasks_maximum.json');before=[0]
            def hook(module,args):before[0]+=1
            handles=[e.backbone.register_forward_pre_hook(hook) for e in model.core.encoders]
            with torch.inference_mode():
                for key,r in available.items():
                    if r['kind']!='query':continue
                    chosen=[p for p in plans if p['wsi_id']==r['wsi_id'] and p['prompt_group']==0 and (p['target']==0 or (p['target']==10 and p['wsi_id']==records[0]['wsi_id'])) and p['support_id'] in available]
                    for plan in chosen:
                        for budget in ('3+2','1+1','3+1','5+1'):
                            p=transfer(prompts(plan,budget),device);task=model.core.support_task(gate_native[plan['support_id']],p);direct,maps=model.core.decode(gate_native[key],task,p,False)
                            cached_task=model.core.support_task(transfer(gate_cpu[plan['support_id']],device),p);cached,cm=model.core.decode(transfer(gate_cpu[key],device),cached_task,p,False)
                            error=float((direct['pixel_probability']-cached['pixel_probability']).abs().max())
                            require(not maps.count_nonzero() and not cm.count_nonzero() and error<=2e-6 and torch.equal(direct['pixel_probability']>=.5,cached['pixel_probability']>=.5),'actual fullcell support/query GPU/native disk parity')
                            checks.append(dict(query=key,target=plan['target'],budget=budget,maximum_error=error,binary_exact=True))
            for h in handles:h.remove()
            require(before[0]==0 and len(checks)==24,'all24 remote native/fullcell budgets and no online encoder')
            dump('decode_checks.json',checks)
        result=dict(status='passed_main30_fullcell_cache_gate' if a.gate else 'complete_main30_fullcell_cache_unaccepted',worker_pid=pid,signature=signature,inputs=len(records),views=3*len(records),query_inputs=sum(r['kind']=='query' for r in records),support_inputs=sum(r['kind']=='support' for r in records),zero_cell_cases=zero,sum_true_instances_across_overlapping_views=true_total,sum_retained_across_views=retained,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),encoded_bytes=cache_bytes,timing=timing,decode_checks=len(checks),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start,remaining='Independent fullcell cache/native-input/actual-feature gate acceptance and full30 originalGT prediction scores; no partition invariant detection claim')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps({k:v for k,v in result.items() if k not in ('signature','zero_cell_cases')}),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);p.add_argument('--gate-acceptance',type=Path)
    run(p.parse_args())
