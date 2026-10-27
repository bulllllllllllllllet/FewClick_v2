"""Real native WSI cell preparation and GPU resident-cache click timing gate."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import torch

from benchmarks.v5.evidence_closure.audit_heldout_local import read,sha,require
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector,CachedCenterCropper,DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models,WEIGHTS
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def transfer(tree,device):
    if torch.is_tensor(tree):return tree.detach().to(device)
    if isinstance(tree,dict):return {k:transfer(v,device) for k,v in tree.items()}
    if isinstance(tree,list):return [transfer(v,device) for v in tree]
    if isinstance(tree,tuple):return tuple(transfer(v,device) for v in tree)
    return tree


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4)
    torch.manual_seed(20261025);np.random.seed(20261025)
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='loading',worker_pid=pid))
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='5' and torch.cuda.is_available(),'explicit allowed GPU5')
        device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();best=read(a.training/'best_checkpoint.json')
        require(sha(best['path'])==best['checkpoint_sha256'],'selected XCell model bytes')
        training=read(a.training/'summary.json');require(training['status']=='complete' and training['step']==10000 and not training['gate'],'full accepted training')
        require(training['training_consumed_sha256']=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','matched full training')
        protocol_sha=sha(str(a.development/'contract.json'));tick=time.perf_counter();model=load('fewclick_m3d_xcell',Path(best['path']),device,protocol_sha)
        torch.cuda.synchronize();model_load=time.perf_counter()-tick;require(model.prior['cell_mode']=='xcell','full cell branch required')
        tick=time.perf_counter();detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device);ctp,xcell,compatibility=load_feature_models(device)
        torch.cuda.synchronize();cell_model_load=time.perf_counter()-tick;cropper=CachedCenterCropper()
        ps=read(a.plan/'summary.json');require(sha(str(a.plan/'slides_private.json'))==ps['slides_sha256'],'fixed plan')
        record=read(a.plan/'slides_private.json')[0];require(sha(record['grid'])==record['grid_sha256'],'grid SHA')
        tiles=[json.loads(line) for line in Path(record['grid']).open()];selected=sorted({len(tiles)//2,len(tiles)-1});require(selected==[1525,3049],'fixed tissue/edge test cases')
        sources=[Path(__file__),Path(inspect.getfile(inspect.unwrap(prepare_variant))),Path(inspect.getfile(InstanSegDetector)),
            Path(inspect.getfile(load_feature_models)),Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/external_evaluation/wsi_reader.py')]
        meta=dict(worker_pid=pid,device=str(device),visible_devices='5',GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,
            cuda_version=torch.version.cuda,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            source_sha256={str(p):sha(str(p)) for p in sources},checkpoint_sha256=best['checkpoint_sha256'],development_protocol_sha256=protocol_sha,
            plan_sha256=ps['slides_sha256'],grid_sha256=record['grid_sha256'],selected_tiles=selected,
            weights={k:dict(path=str(p),sha256=sha(str(p))) for k,p in WEIGHTS.items() if k in ('ctranspath','xcell')},
            detector_sha256=sha(str(DEFAULT_INSTANSEG_CHECKPOINT)),cell_model_compatibility=compatibility,query_GT_available=False,
            precision='FP32 inference; detector official autocast internally',seed=20261025,detector_call_order='tile ascending then scale0/1/2',
            scope='Two actual WSI windows full-cell mechanical/cache/GPU per-window timing, not full-slide speed or TLS accuracy',
            XCell_provenance='User confirmed internal data excludes current CRC/TIGER; TLS upstream exposure not established',
            cache_scope='Online timing assumes encoded query cache already resident on GPU; excludes disk/H2D/global stitch')
        dump('run.json',meta);data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
        count=[0];original=model.core.encode_image
        def encode(raw):count[0]+=1;return original(raw)
        model.core.encode_image=encode;tick=time.perf_counter();support,_,_=data.materialize(nested(maximum,'3+2'),device,True,True)
        with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(support))
        torch.cuda.synchronize();support_prepare=time.perf_counter()-tick
        slide=StrictSlide(record['provenance']['wsi_path']);queries=[];detector_calls=0
        try:
            require([slide.width,slide.height]==[record['width'],record['height']] and [slide.mx,slide.my]==record['geometry']['native_mpp'],'physical slide header')
            for i in selected:
                dump('status.json',dict(status='preparing_full_cells',worker_pid=pid,tile=i,detector_calls=detector_calls));tick=time.perf_counter()
                images,geometry=slide.fields(tiles[i]['centre_level0']);image_read=time.perf_counter()-tick
                require(geometry['native_boxes']==tiles[i]['native_boxes'],'planned real context')
                raw={};cell_records=[]
                for scale,(image,box) in enumerate(zip(images,geometry['native_boxes'])):
                    tick=time.perf_counter();native=slide.read(box);native_read=time.perf_counter()-tick;w,h=box[2:]
                    metadata=dict(input_to_image_affine=[w/512,0,0,0,h/512,0],source_image_size=[w,h])
                    torch.cuda.synchronize();tick=time.perf_counter();detector_calls+=1
                    arrays,details=prepare_variant(native,geometry['native_mpp'],metadata,detector,ctp,xcell,device,cropper=cropper)
                    torch.cuda.synchronize();cell_time=time.perf_counter()-tick
                    p=a.output/f'cells_{i:06d}_scale{scale}.npz';np.savez_compressed(p,**arrays)
                    with np.load(p,allow_pickle=False) as disk:require(all(np.array_equal(disk[k],v) for k,v in arrays.items()),'real cell cache roundtrip')
                    suffix='' if scale==0 else f'_{scale}'
                    raw['image'+suffix]=he_to_tensor(image.copy())[None].to(device)
                    raw['cells'+suffix]=torch.from_numpy(arrays['features'])[None].to(device)
                    raw['cell_valid'+suffix]=torch.ones(1,len(arrays['features']),dtype=torch.bool,device=device)
                    raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([int(arrays['total'])],dtype=torch.float32,device=device)
                    cell_records.append(dict(scale=scale,path=str(p),sha256=sha(str(p)),native_RGB_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                        native_read_seconds=native_read,cell_prepare_seconds=cell_time,details=details))
                    dump('status.json',dict(status='preparing_full_cells',worker_pid=pid,tile=i,scale=scale,detector_calls=detector_calls,total_cells=int(arrays['total'])))
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.inference_mode():encoded=model.core.encode_image(raw)
                torch.cuda.synchronize();encode_time=time.perf_counter()-tick;path=a.output/f'encoded_{i:06d}.pth'
                tick=time.perf_counter();cpu=transfer(encoded,'cpu');torch.save(dict(encoded=cpu,geometry=geometry,checkpoint_sha256=best['checkpoint_sha256']),path);cache_write=time.perf_counter()-tick
                disk=torch.load(path,map_location='cpu',weights_only=False);require(disk['geometry']==geometry and disk['checkpoint_sha256']==best['checkpoint_sha256'],'encoded disk identity')
                cached=transfer(disk['encoded'],device);torch.cuda.synchronize()
                queries.append(dict(tile=i,encoded=encoded,cached=cached,encoded_path=str(path),encoded_sha256=sha(str(path)),
                    images_read_seconds=image_read,model_encode_seconds=encode_time,cache_write_seconds=cache_write,cell_records=cell_records))
        finally:slide.close()
        before=count[0];before_detector=detector_calls;checks=[]
        with torch.inference_mode():
            for query in queries:
                for budget in ('1+1','3+2'):
                    _,prompts,_=data.materialize(nested(maximum,budget),device,True,True);task=model.core.support_task(source,prompts)
                    native,maps=model.core.decode(query['encoded'],task,prompts,False);replayed,replaymaps=model.core.decode(query['cached'],task,prompts,False)
                    require(not maps.count_nonzero() and not replaymaps.count_nonzero(),'zero query prompt maps')
                    error=float((native['pixel_probability']-replayed['pixel_probability']).abs().max());require(error<=2e-6 and torch.equal(native['pixel_probability']>=.5,replayed['pixel_probability']>=.5),'GPU cache/native output parity')
                    np.save(a.output/f'probability_{query["tile"]:06d}_{budget}.npy',replayed['pixel_probability'][0].cpu().numpy(),allow_pickle=False)
                    for _ in range(3):model.core.decode(query['cached'],model.core.support_task(source,prompts),prompts,False)
                    timings=[]
                    for _ in range(20):
                        torch.cuda.synchronize();wall=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                        events[0].record();updated=model.core.support_task(source,prompts);events[1].record()
                        result,heat=model.core.decode(query['cached'],updated,prompts,False);events[2].record();torch.cuda.synchronize()
                        require(not heat.count_nonzero(),'timed query map zero')
                        timings.append(dict(task_ms=events[0].elapsed_time(events[1]),decode_ms=events[1].elapsed_time(events[2]),wall_ms=(time.perf_counter()-wall)*1000))
                    checks.append(dict(tile=query['tile'],budget=budget,native_cache_max_error=error,binary_equal=True,timings=timings,
                        task_ms_median=float(np.median([t['task_ms'] for t in timings])),decode_ms_median=float(np.median([t['decode_ms'] for t in timings])),
                        wall_ms_median=float(np.median([t['wall_ms'] for t in timings])),wall_ms_p95=float(np.quantile([t['wall_ms'] for t in timings],.95))))
        require(count[0]==before and detector_calls==before_detector==6,'no query preparation during timed interaction')
        for q in queries:q.pop('encoded');q.pop('cached')
        result=dict(status='passed_real_fullcell_GPU_cache_and_click_gate',worker_pid=pid,queries=queries,checks=checks,model_load_seconds=model_load,
            cell_models_load_seconds=cell_model_load,support_prepare_seconds=support_prepare,detector_calls=detector_calls,
            online_query_encode_calls=0,online_detector_calls=0,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),total_wall_seconds=time.perf_counter()-start,
            source_sha256=sha(__file__),limits=['Two fixed windows only; full-slide cell preparation and global GPU output/IO not benchmarked',
                'Warm20resident-cache per-window timing, not WSI latency; initialization/detection/encoding/cache transfer/global stitching excluded from online metric',
                'Fixed CRC source prompt mechanical task on TLS slide, no TLS semantic accuracy','InstanSeg call history/seed recorded, not claiming partition-invariant detections'])
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','development','training','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
