"""Versioned native full-cell preparation and disk-streamed GPU WSI inference."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import pyvips
import numpy as np
import torch

from .fullcell_gpu_gate import transfer
from .stripe import StripeWriter
from .cache_gate import stitch
from benchmarks.v5.evidence_closure.audit_heldout_local import read,sha,require
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector,CachedCenterCropper,DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models,WEIGHTS
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def finite_tree(value):
    if torch.is_tensor(value) and value.is_floating_point():require(torch.isfinite(value).all(),'finite encoded tensor')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4)
    torch.manual_seed(20261025);np.random.seed(20261025)
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid,gate=a.gate))
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='5' and torch.cuda.is_available(),'allowed actual GPU5 required')
        accepted=read(a.native_audit/'summary.json');native=read(a.native_gate/'summary.json');nm=read(a.native_gate/'run.json')
        require(accepted['status']=='passed_independent_fullcell_GPU_gate_audit' and accepted['gate_summary_sha256']==sha(str(a.native_gate/'summary.json')),'accepted fullcell native prerequisite')
        for p,v in nm['source_sha256'].items():require(sha(p)==v,'native preparation sources unchanged')
        ps=read(a.plan/'summary.json');require(sha(str(a.plan/'slides_private.json'))==ps['slides_sha256'],'full plan')
        record=read(a.plan/'slides_private.json')[0];require(sha(record['grid'])==record['grid_sha256'],'original all3050 grid')
        tiles=[json.loads(line) for line in Path(record['grid']).open()];require(len(tiles)==3050,'full grid count')
        if a.gate:
            nx=len(record['geometry']['x_origins']);ny=len(record['geometry']['y_origins'])
            indices=[y*nx+x for y in (ny//2,ny//2+1) for x in (nx//2,nx//2+1)]
        else:indices=list(range(len(tiles)))
        selected=[tiles[i] for i in indices];boxes=[t['native_boxes'][0] for t in selected];fw,fh=record['geometry']['fine_native_wh']
        bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
        require(a.gate or bounds==record['geometry']['valid_fine_bounds_xyxy'],'full planned prediction domain')
        best=read(a.training/'best_checkpoint.json');done=read(a.training/'summary.json')
        require(done['status']=='complete' and done['step']==10000 and best['checkpoint_sha256']==nm['checkpoint_sha256']==sha(best['path']),'accepted selected fullcell weight')
        protocol_sha=sha(str(a.development/'contract.json'));require(protocol_sha==nm['development_protocol_sha256'],'same source support protocol')
        sources=[Path(__file__),Path('benchmarks/v5/wsi_global/fullcell_gpu_gate.py'),Path('benchmarks/v5/wsi_global/stripe.py'),
            Path(inspect.getfile(inspect.unwrap(prepare_variant))),Path(inspect.getfile(InstanSegDetector)),Path(inspect.getfile(load_feature_models)),
            Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/fair_comparison/local_data.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py')]
        signature=dict(source_sha256={str(p):sha(str(p)) for p in sources},checkpoint_sha256=best['checkpoint_sha256'],
            plan_sha256=ps['slides_sha256'],grid_sha256=record['grid_sha256'],development_protocol_sha256=protocol_sha,
            native_audit_sha256=sha(str(a.native_audit/'summary.json')))
        if not a.gate:
            require(a.stream_gate is not None,'real fullcell streaming gate required')
            gate=read(a.stream_gate/'summary.json');require(gate['status']=='passed_fullcell_GPU_stream_gate' and gate['signature']==signature,'same-source production gate')
            require(a.stream_gate.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'),'stream gate shell exit')
        estimated=(sum(round(b[2]*record['geometry']['native_mpp'][0]/.25)*round(b[3]*record['geometry']['native_mpp'][1]/.25)*4 for b in selected[0]['native_boxes'])+8_000_000)*len(selected)+2*(bounds[2]-bounds[0])*(bounds[3]-bounds[1])*7
        free=shutil.disk_usage(a.output).free;require(free>1.5*estimated,'actual disk headroom for immutable complete artifacts')
        dump('contract.json',dict(worker_pid=pid,gate=a.gate,signature=signature,tiles=len(selected),indices=indices,bounds_level0=bounds,
            device='cuda:0',visible_devices='5',GPU=torch.cuda.get_device_name(),precision='FP32; official detector internal autocast',
            seed=20261025,detector_call_order='row-major tile then scale0/1/2; no old gate cell reuse',
            estimated_preserved_bytes=estimated,actual_free_bytes=free,failure_policy='Stop and preserve; no retry/fallback/resume',
            query_GT_used=False,task_scope='fixed CRC support mechanical task on TLS WSI, not TLS semantic accuracy',
            field_scope='true4x context domain only; no tissue filter/padding',cache_state='OS/NFS uncontrolled; online reads CPU disk caches into GPU'))
        torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0');tick=time.perf_counter()
        model=load('fewclick_m3d_xcell',Path(best['path']),device,protocol_sha);torch.cuda.synchronize();model_load=time.perf_counter()-tick
        tick=time.perf_counter();detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device);ctp,xcell,compatibility=load_feature_models(device);torch.cuda.synchronize();cell_models_load=time.perf_counter()-tick
        dump('cell_recipe.json',dict(weights={k:dict(path=str(p),sha256=sha(str(p))) for k,p in WEIGHTS.items() if k in ('ctranspath','xcell')},
            detector_sha256=sha(str(DEFAULT_INSTANSEG_CHECKPOINT)),compatibility=compatibility,
            source_sha256=signature['source_sha256'],cell_feature_dim=74,max_retained=255,unknown_nuclear_class=0))
        data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0);native_probabilities={}
        encode_calls=[0];original=model.core.encode_image
        def encode(raw):encode_calls[0]+=1;return original(raw)
        model.core.encode_image=encode;tick=time.perf_counter();raw,_,_=data.materialize(nested(maximum,'1+1'),device,True,True)
        with torch.inference_mode():support=model.core.encode_image(fewclick_inputs(raw))
        torch.cuda.synchronize();support_seconds=time.perf_counter()-tick;cropper=CachedCenterCropper();manifest=[];zero_cases=[]
        timing=dict(model_load_seconds=model_load,cell_models_load_seconds=cell_models_load,support_prepare_seconds=support_seconds,
            image_read_seconds=0.,cell_native_read_seconds=0.,cell_prepare_seconds=0.,model_encode_seconds=0.,cache_write_seconds=0.,cache_hash_seconds=0.)
        cells_dir=a.output/'cells';cells_dir.mkdir();encoded_dir=a.output/'encoded';encoded_dir.mkdir();slide=StrictSlide(record['provenance']['wsi_path'])
        try:
            require([slide.width,slide.height]==[record['width'],record['height']] and [slide.mx,slide.my]==record['geometry']['native_mpp'],'actual slide header')
            with (a.output/'cache_manifest.jsonl').open('x') as handle:
                for number,(i,tile) in enumerate(zip(indices,selected)):
                    for p,v in signature['source_sha256'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v,'frozen production source')
                    tick=time.perf_counter();images,geometry=slide.fields(tile['centre_level0']);timing['image_read_seconds']+=time.perf_counter()-tick
                    require(geometry['native_boxes']==tile['native_boxes'],'planned native image footprint');raw={};cell_records=[]
                    for scale,(image,box) in enumerate(zip(images,geometry['native_boxes'])):
                        tick=time.perf_counter();native_rgb=slide.read(box);timing['cell_native_read_seconds']+=time.perf_counter()-tick;w,h=box[2:]
                        metadata=dict(input_to_image_affine=[w/512,0,0,0,h/512,0],source_image_size=[w,h])
                        torch.cuda.synchronize();tick=time.perf_counter();arrays,details=prepare_variant(native_rgb,geometry['native_mpp'],metadata,detector,ctp,xcell,device,cropper=cropper);torch.cuda.synchronize();timing['cell_prepare_seconds']+=time.perf_counter()-tick
                        if int(arrays['total'])==0:zero_cases.append(dict(tile=i,scale=scale))
                        if a.gate and i==1525:
                            old=next(q for q in native['queries'] if q['tile']==i)['cell_records'][scale]
                            with np.load(old['path'],allow_pickle=False) as reference:
                                require(np.array_equal(reference['instances'],arrays['instances']) and np.array_equal(reference['features'],arrays['features']) and int(reference['total'])==int(arrays['total']),'same initial native-gate cell history parity')
                        path=cells_dir/f'tile_{i:06d}_scale{scale}.npz';tick=time.perf_counter();np.savez_compressed(path,**arrays);timing['cache_write_seconds']+=time.perf_counter()-tick
                        with np.load(path,allow_pickle=False) as disk:require(all(np.array_equal(disk[k],v) for k,v in arrays.items()),'actual cell disk roundtrip')
                        suffix='' if scale==0 else f'_{scale}';raw['image'+suffix]=he_to_tensor(image.copy())[None].to(device)
                        raw['cells'+suffix]=torch.from_numpy(arrays['features'])[None].to(device);raw['cell_valid'+suffix]=torch.ones(1,len(arrays['features']),dtype=torch.bool,device=device)
                        raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([int(arrays['total'])],dtype=torch.float32,device=device)
                        tick=time.perf_counter();digest=sha(str(path));timing['cache_hash_seconds']+=time.perf_counter()-tick
                        cell_records.append(dict(scale=scale,path=str(path),sha256=digest,native_RGB_sha256=hashlib.sha256(native_rgb.tobytes()).hexdigest(),details=details))
                    torch.cuda.synchronize();tick=time.perf_counter()
                    with torch.inference_mode():encoded=model.core.encode_image(raw)
                    torch.cuda.synchronize();timing['model_encode_seconds']+=time.perf_counter()-tick
                    finite_tree(encoded)
                    if a.gate:
                        with torch.inference_mode():
                            for budget in ('1+1','3+2'):
                                _,p,_=data.materialize(nested(maximum,budget),device,True,True);task=model.core.support_task(support,p)
                                native_result,maps=model.core.decode(encoded,task,p,False);require(not maps.count_nonzero(),'native gate query maps zero')
                                native_probabilities[i,budget]=native_result['pixel_probability'][0].cpu().numpy().copy()
                    path=encoded_dir/f'tile_{i:06d}.pth';identity=dict(slide_id=record['slide_id'],geometry=geometry,RGB_array_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],cell_records=cell_records)
                    tick=time.perf_counter();torch.save(dict(encoded=transfer(encoded,'cpu'),identity=identity,checkpoint_sha256=best['checkpoint_sha256']),path);timing['cache_write_seconds']+=time.perf_counter()-tick
                    tick=time.perf_counter();digest=sha(str(path));timing['cache_hash_seconds']+=time.perf_counter()-tick
                    entry=dict(number=number,tile=i,path=str(path),sha256=digest,bytes=path.stat().st_size,identity=identity);manifest.append(entry);handle.write(json.dumps(entry)+'\n');handle.flush()
                    dump('status.json',dict(status='preparing_fullcell_cache',worker_pid=pid,completed=number+1,total=len(selected),tile=i,zero_cell_cases=len(zero_cases),seconds=time.perf_counter()-start))
                    if number%10==0 or a.gate:print(json.dumps(dict(stage='cell_cache',completed=number+1,total=len(selected),tile=i,counts=[r['details']['total_cell_count'] for r in cell_records],seconds=time.perf_counter()-start)),flush=True)
                    del encoded,raw
        finally:slide.close()
        require(encode_calls[0]==1+len(selected),'offline encoding count');offline_peak=torch.cuda.max_memory_allocated();dump('cache_complete.json',dict(status='complete_fullcell_cache',tiles=len(selected),manifest_sha256=sha(str(a.output/'cache_manifest.jsonl')),timing=timing,zero_cell_cases=zero_cases))
        del detector,ctp,xcell;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();before=encode_calls[0];passes=[]
        for budget in ('1+1','3+2'):
            wall=time.perf_counter();_,prompts,_=data.materialize(nested(maximum,budget),device,True,True);torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():task=model.core.support_task(support,prompts)
            torch.cuda.synchronize();task_seconds=time.perf_counter()-tick;writer=StripeWriter(a.output/('online_'+budget),bounds,[fw,fh]);io=move=decode=stitch_time=validation=0.;gate_p=[]
            with torch.inference_mode():
                for index,entry in enumerate(manifest):
                    tick=time.perf_counter();require(hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],'online immutable encoded bytes');validation+=time.perf_counter()-tick
                    tick=time.perf_counter();payload=torch.load(entry['path'],map_location='cpu',weights_only=False);io+=time.perf_counter()-tick
                    require(payload['identity']==entry['identity'] and payload['checkpoint_sha256']==best['checkpoint_sha256'],'streamed cache identity')
                    torch.cuda.synchronize();tick=time.perf_counter();encoded=transfer(payload['encoded'],device);torch.cuda.synchronize();move+=time.perf_counter()-tick
                    tick=time.perf_counter();result,maps=model.core.decode(encoded,task,prompts,False);torch.cuda.synchronize();decode+=time.perf_counter()-tick;require(not maps.count_nonzero(),'query maps zero')
                    p=result['pixel_probability'][0].cpu().numpy();require(np.isfinite(p).all(),'finite fullcell prediction')
                    if a.gate:
                        direct=native_probabilities[entry['tile'],budget]
                        require(np.max(np.abs(direct-p))<=2e-6 and np.array_equal(direct>=.5,p>=.5),'real native GPU versus reloaded cache probability/mask');gate_p.append(p.copy())
                    if index in (0,len(manifest)//2,len(manifest)-1):np.save(a.output/f'probe_{budget}_{entry["tile"]:06d}.npy',p,allow_pickle=False)
                    tick=time.perf_counter();writer.push(p,entry['identity']['geometry']['native_boxes'][0]);stitch_time+=time.perf_counter()-tick
                    dump('status.json',dict(status='online_GPU',worker_pid=pid,budget=budget,completed=index+1,total=len(manifest),seconds=time.perf_counter()-start))
            coverage=writer.finish();pass_seconds=time.perf_counter()-wall
            require(encode_calls[0]==before,'no query encoding online')
            if a.gate:
                expected,_,_=stitch(gate_p,boxes);actual=np.load(writer.folder/'probability.npy',mmap_mode='r');require(np.max(np.abs(expected-actual))<=1e-7,'gate independent complete stitched probability')
            outputs={n:dict(path=str(writer.folder/(n+'.npy')),sha256=sha(str(writer.folder/(n+'.npy')))) for n in ('probability','mask','overlap')}
            passes.append(dict(budget=budget,coverage=coverage,task_seconds=task_seconds,cache_validation_seconds=validation,cache_load_seconds=io,H2D_seconds=move,decode_seconds=decode,
                resize_stitch_seconds=stitch_time,wall_seconds=pass_seconds,output_hash_seconds=time.perf_counter()-wall-pass_seconds,outputs=outputs))
            dump('pass_complete_'+budget+'.json',passes[-1]);print(json.dumps(dict(stage='online_complete',budget=budget,wall_seconds=pass_seconds)),flush=True)
        result=dict(status='passed_fullcell_GPU_stream_gate' if a.gate else 'complete_fullcell_GPU_WSI_execution',worker_pid=pid,gate=a.gate,signature=signature,
            tiles=len(selected),timing=timing,passes=passes,zero_cell_cases=zero_cases,online_query_encode_calls=0,online_detector_calls=0,
            offline_peak_GPU_allocated_bytes=offline_peak,online_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            total_wall_seconds=time.perf_counter()-start,scope='Full cell branch, fixedCRC mechanical source on TLS; valid4x domain, no TLS accuracy or other29slide performance',
            remaining='Independent full encoded/probability/coverage/true-cell geometry acceptance; query GT and TLS completeness not certified')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],tiles=len(selected),seconds=result['total_wall_seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','development','training','native-gate','native-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--stream-gate',type=Path);run(p.parse_args())
