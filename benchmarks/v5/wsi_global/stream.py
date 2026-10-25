"""Full CPU image-only WSI cache preparation then disk-streamed click passes."""
import argparse
import hashlib
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

from .stripe import StripeWriter
from .cache_gate import raw_images,load_cache
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested


def read(path):return json.loads(Path(path).read_text())


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU stream must hide CUDA; GPU benchmarking requires its own native gate')
    if a.gate_tiles is not None and not a.gate:raise ValueError('partial tile override only for explicit execution gate')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.perf_counter()
    files=[Path(__file__),Path(__file__).with_name('stripe.py'),Path(__file__).with_name('cache_gate.py'),
        Path('benchmarks/v5/external_evaluation/wsi_reader.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),
        Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/paper_multiscale/model.py'),
        Path('benchmarks/v4/phase_4_cross_scale/src/export_tokens.py')]
    sources={str(p):digest(p) for p in files}
    def locked():
        for path,sha in sources.items():
            if digest(path)!=sha:raise ValueError('locked WSI execution source changed '+path)
    try:
        accepted=read(a.stream_gate/'summary.json')
        if accepted['status']!='passed_real_streaming_writer_gate':raise ValueError('actual streaming writer gate required')
        for path,sha in accepted['source_sha256'].items():
            if digest(path)!=sha:raise ValueError('stream-gated source changed')
        plan=read(a.plan/'summary.json');rows=read(a.plan/'slides_private.json')
        if plan['status']!='complete_spatial_plan' or digest(a.plan/'slides_private.json')!=plan['slides_sha256']:raise ValueError('full spatial plan changed')
        record=rows[a.slide_index]
        if digest(record['grid'])!=record['grid_sha256']:raise ValueError('full tile grid changed')
        tiles=[json.loads(line) for line in Path(record['grid']).open()]
        if len(tiles)!=record['geometry']['tiles']:raise ValueError('full grid count mismatch')
        if a.gate_tiles is not None:tiles=read(a.gate_tiles)
        if not tiles or len({tuple(t['native_boxes'][0]) for t in tiles})!=len(tiles):raise ValueError('empty/duplicate tiles')
        boxes=[t['native_boxes'][0] for t in tiles];fw,fh=record['geometry']['fine_native_wh']
        if any(b[2:]!=[fw,fh] for b in boxes) or [b[1] for b in boxes]!=sorted(b[1] for b in boxes):raise ValueError('fine geometry or row ordering')
        bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
        if not a.gate and bounds!=record['geometry']['valid_fine_bounds_xyxy']:raise ValueError('full-domain coverage changed')
        training=read(a.training/'summary.json');best=read(a.training/'best_checkpoint.json');expected='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe'
        if training['status']!='complete' or training['step']!=10000 or not training['full_manifest_parity'] or training['training_consumed_sha256']!=expected:
            raise ValueError('full G3 training evidence required')
        if digest(a.training/'training_consumed.jsonl')!=expected or digest(best['path'])!=best['checkpoint_sha256']:
            raise ValueError('actual training/selected weights changed')
        if best['checkpoint_sha256']!=accepted['selected_checkpoint_sha256']:raise ValueError('stream gate uses different model weights')
        pixels=(bounds[2]-bounds[0])*(bounds[3]-bounds[1]);estimate=len(tiles)*8_000_000+2*pixels*7
        free=shutil.disk_usage(a.output).free
        if free<estimate*1.5:raise ValueError('insufficient actual disk headroom for preserved outputs')
        write(a.output/'contract.json',dict(worker_pid=os.getpid(),gate=a.gate,slide_id=record['slide_id'],slide_index=a.slide_index,
            source_sha256=sources,plan_sha256=plan['slides_sha256'],grid_sha256=record['grid_sha256'],tiles=len(tiles),
            gate_tiles_sha256=None if a.gate_tiles is None else digest(a.gate_tiles),bounds_level0=bounds,native_mpp=record['geometry']['native_mpp'],
            checkpoint_sha256=best['checkpoint_sha256'],checkpoint=best['path'],development_protocol_sha256=digest(a.development/'contract.json'),
            estimated_preserved_bytes=estimate,observed_disk_free_bytes=free,device='cpu',threads=4,budgets=['1+1','3+2'],
            cell_detection_performed=False,training_scope='Selected G3 after all-label local adaptation; noJ5 model initialization, not unseen-label TLS',
            task_scope='Fixed CRC support task applied to all query tiles; mechanical execution/efficiency, not TLS prompts or accuracy',
            file_system_cache='OS/NFS cache state uncontrolled; no cold-cache claim',failure_policy='Preserve incomplete artifacts; no retry or partial-cache reuse'))
        tick=time.perf_counter();model=load('fewclick_m3d_image',Path(best['path']),'cpu',digest(a.development/'contract.json'));model_load=time.perf_counter()-tick
        if model.prior['cell_mode']!='none':raise ValueError('placeholder cells only legal for image-only control')
        data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
        count=[0];original=model.core.encode_image
        def encode(raw):count[0]+=1;return original(raw)
        model.core.encode_image=encode
        cache_dir=a.output/'encoded';cache_dir.mkdir();manifest=[];timing=dict(model_load_seconds=model_load,WSI_read_seconds=0.,normalize_seconds=0.,encode_seconds=0.,cache_write_seconds=0.,cache_hash_seconds=0.)
        slide=StrictSlide(record['provenance']['wsi_path'])
        try:
            if [slide.width,slide.height]!=[record['width'],record['height']] or [slide.mx,slide.my]!=record['geometry']['native_mpp']:
                raise ValueError('actual slide header changed')
            with torch.inference_mode(),(a.output/'cache_manifest.jsonl').open('x') as handle:
                for index,tile in enumerate(tiles):
                    locked();tick=time.perf_counter();images,geometry=slide.fields(tile['centre_level0']);timing['WSI_read_seconds']+=time.perf_counter()-tick
                    if geometry['native_boxes']!=tile['native_boxes']:raise ValueError('actual input footprint differs from plan')
                    identity=dict(slide_id=record['slide_id'],geometry=geometry,RGB_array_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images])
                    tick=time.perf_counter();raw=raw_images(images);timing['normalize_seconds']+=time.perf_counter()-tick
                    tick=time.perf_counter();encoded=model.core.encode_image(raw);timing['encode_seconds']+=time.perf_counter()-tick
                    path=cache_dir/f'tile_{index:06d}.pth';tick=time.perf_counter();torch.save(dict(encoded=encoded,identity=identity,checkpoint_sha256=best['checkpoint_sha256']),path);timing['cache_write_seconds']+=time.perf_counter()-tick
                    tick=time.perf_counter();sha=digest(path);timing['cache_hash_seconds']+=time.perf_counter()-tick
                    entry=dict(number=index,path=str(path),sha256=sha,bytes=path.stat().st_size,identity=identity)
                    manifest.append(entry);handle.write(json.dumps(entry,sort_keys=True)+'\n');handle.flush()
                    write(a.output/'status.json',dict(status='preparing_cache',worker_pid=os.getpid(),completed=index+1,total=len(tiles),seconds=time.perf_counter()-started))
                    if (index+1)%20==0 or a.gate:print(json.dumps(dict(stage='cache',completed=index+1,total=len(tiles),seconds=time.perf_counter()-started)),flush=True)
        finally:slide.close()
        if count[0]!=len(tiles):raise ValueError('offline encode coverage')
        write(a.output/'cache_complete.json',dict(status='complete_cache',tiles=len(manifest),manifest_sha256=digest(a.output/'cache_manifest.jsonl'),
            cache_bytes=sum(m['bytes'] for m in manifest),timings=timing))
        tick=time.perf_counter();support,prompts,_=data.materialize(nested(maximum,'1+1'),'cpu',True,True)
        with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(support))
        support_prepare=time.perf_counter()-tick;before=count[0];passes=[]
        for budget in ('1+1','3+2'):
            locked();pass_start=time.perf_counter();_,prompts,_=data.materialize(nested(maximum,budget),'cpu',True,True)
            tick=time.perf_counter()
            with torch.inference_mode():task=model.core.support_task(source,prompts)
            task_seconds=time.perf_counter()-tick;writer=StripeWriter(a.output/('online_'+budget),bounds,[fw,fh]);cache_io=0.;decode=0.;stitch=0.
            with torch.inference_mode():
                for index,entry in enumerate(manifest):
                    locked();tick=time.perf_counter()
                    if digest(entry['path'])!=entry['sha256']:raise ValueError('cache file changed before online pass')
                    encoded=load_cache(Path(entry['path']),best['checkpoint_sha256'],entry['identity']);cache_io+=time.perf_counter()-tick
                    tick=time.perf_counter();result,maps=model.core.decode(encoded,task,prompts,False);decode+=time.perf_counter()-tick
                    if maps.count_nonzero():raise ValueError('query prompt maps nonzero')
                    tick=time.perf_counter();writer.push(result['pixel_probability'][0].numpy(),entry['identity']['geometry']['native_boxes'][0]);stitch+=time.perf_counter()-tick
                    if index in (0,len(manifest)//2,len(manifest)-1):np.save(a.output/f'probe_{budget}_{index:06d}.npy',result['pixel_probability'][0].numpy())
                    write(a.output/'status.json',dict(status='online',worker_pid=os.getpid(),budget=budget,completed=index+1,total=len(manifest),seconds=time.perf_counter()-started))
            tick=time.perf_counter();coverage=writer.finish();finalize=time.perf_counter()-tick
            if count[0]!=before:raise ValueError('online query encoder executed')
            pass_seconds=time.perf_counter()-pass_start;tick=time.perf_counter()
            outputs={name:dict(path=str(writer.folder/(name+'.npy')),sha256=digest(writer.folder/(name+'.npy'))) for name in ('probability','mask','overlap')}
            passes.append(dict(budget=budget,coverage=coverage,task_seconds=task_seconds,cache_validation_load_seconds=cache_io,
                decode_seconds=decode,resize_accumulate_flush_seconds=stitch,finalize_seconds=finalize,online_wall_seconds=pass_seconds,
                output_hash_seconds=time.perf_counter()-tick,outputs=outputs))
            write(a.output/f'pass_complete_{budget}.json',passes[-1])
        write(a.output/'summary.json',dict(status='complete_stream_execution',gate=a.gate,worker_pid=os.getpid(),tiles=len(tiles),
            complete_slide=not a.gate,slide_id=record['slide_id'],cache_bytes=sum(m['bytes'] for m in manifest),offline_timings=timing,
            support_prepare_seconds=support_prepare,passes=passes,online_query_encode_calls=0,query_prompt_maps_zero=True,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,total_seconds=time.perf_counter()-started,
            scope='Image-only CPU full planned prediction domain, excluded context edges; no TLS semantic accuracy/full-cell/GPU efficiency'))
        write(a.output/'status.json',dict(status='complete_stream_execution',gate=a.gate,worker_pid=os.getpid(),tiles=len(tiles)))
    except Exception:
        write(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','development','training','stream-gate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--slide-index',type=int,default=0);p.add_argument('--gate',action='store_true');p.add_argument('--gate-tiles',type=Path)
    run(p.parse_args())
