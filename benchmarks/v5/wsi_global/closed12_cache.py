"""CPU full CRC30 closed12 reference cache, with exact real native-path gate."""
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
from PIL import Image
import torch

from .native_remote_gate import raw,encode
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();started=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid))
    try:
        require(not torch.cuda.is_available(),'CPU4 cache only')
        accepted=read(a.acceptance/'summary.json');native=read(a.native_acceptance/'summary.json')
        require(accepted['status']=='accepted_source_band_query_protocol' and accepted['slides']==30 and accepted['protocol_summary_sha256']==sha(a.protocol/'summary.json'),'accepted all30 source/query protocol')
        require(native['status']=='accepted_WSI_native_remote_cache_gate' and native['gate_summary_sha256']==sha(a.native_gate/'summary.json'),'actual native4method gate accepted')
        inputs=read(a.protocol/'query_inputs.json');require(len(inputs)==30,'all30 slides')
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');training=base/'adaptation_1025_131451/train_closed_reference_1025_131451';best=read(training/'best_checkpoint.json');done=read(training/'summary.json')
        require(done['status']=='complete' and done['step']==10000 and done['training_consumed_sha256']==sha(training/'training_consumed.jsonl')=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','accepted full-label training')
        weight=next(d for d in read(a.native_gate/'summary.json')['methods'] if d['method']=='closed_reference')['checkpoint_sha256'];require(sha(best['path'])==best['checkpoint_sha256']==weight,'same selected weight')
        sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/native_remote_gate.py'),Path('benchmarks/v5/wsi_global/cache_gate.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),Path('benchmarks/v5/fair_comparison/adapted_models.py')]}
        signature=dict(source_sha256=sources,protocol_sha256=sha(a.protocol/'summary.json'),acceptance_sha256=sha(a.acceptance/'summary.json'),weight_sha256=weight,native_acceptance_sha256=sha(a.native_acceptance/'summary.json'))
        old=read(a.native_gate/'input_identities.json');queries={k:v for k,v in old.items() if ':query:' in k}
        if not a.gate:
            require(a.verified_gate is not None,'actual same-source cache gate required');g=read(a.verified_gate/'summary.json');require(g['status']=='passed_closed12_production_cache_gate' and g['signature']==signature,'same-recipe4tile gate')
        selected=[]
        for r in inputs:
            require(sha(r['grid'])==r['grid_sha256'],'frozen complete remote grid')
            tiles=[json.loads(line) for line in Path(r['grid']).open()]
            if a.gate:tiles=[t for t in tiles if r['wsi_id']+':query:'+str(t['number']) in queries]
            selected.append((r,tiles))
        total=sum(len(t) for _,t in selected);require(total==(4 if a.gate else 20208),'expected complete query input count')
        estimate=total*12*512*512*4;free=shutil.disk_usage(a.output).free;require(free>1.5*estimate,'real preserved-logit disk headroom')
        dump('contract.json',dict(worker_pid=pid,gate=a.gate,signature=signature,total_tiles=total,estimated_bytes=estimate,actual_free_bytes=free,device='CPU4 FP32',query_GT_access=False,method='closed_reference',extra_supervision='all12-class CE; no support clicks required',failure_policy='stop preserve; no retry/fallback/resume'))
        model=load('closed_reference',Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'))
        timing=dict(read_resize_normalize_seconds=0.,model_encode_seconds=0.,write_seconds=0.,hash_seconds=0.);completed=0;gate_errors=[]
        with (a.output/'cache_manifest.jsonl').open('x') as manifest,torch.inference_mode():
            for r,tiles in selected:
                if not tiles:continue
                native_image=pyvips.Image.new_from_file(r['source_wsi'],access='random');folder=a.output/r['wsi_id'];folder.mkdir()
                for t in tiles:
                    for p,d in sources.items():require(sha(p)==d,'frozen production source')
                    tick=time.perf_counter();box=t['native_boxes'][0];crop=native_image.crop(*box).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(box[3],box[2],3)
                    image=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)).copy();value=raw([image]);timing['read_resize_normalize_seconds']+=time.perf_counter()-tick
                    tick=time.perf_counter();logits=encode(model,'closed_reference',value);timing['model_encode_seconds']+=time.perf_counter()-tick
                    require(logits.shape==(1,12,512,512) and torch.isfinite(logits).all(),'finite exact12class output')
                    identity=dict(wsi_id=r['wsi_id'],tile=t['number'],source_wsi=r['source_wsi'],native_box=box,native_mpp=r['native_mpp'],RGB_array_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                    if a.gate:
                        key=r['wsi_id']+':query:'+str(t['number']);refpath=a.native_gate/'closed_reference'/(key.replace(':','_')+'.pth')
                        require(sha(refpath)==native['cache_sha256'][str(refpath)],'independently accepted native cache bytes')
                        reference=torch.load(refpath,map_location='cpu',weights_only=False)['encoded'];error=float((reference-logits).abs().max());require(error==0,'all12 logits native versus production exact');gate_errors.append(error)
                    path=folder/f'tile_{t["number"]:06d}.pth';tick=time.perf_counter();torch.save(dict(logits=logits,identity=identity,checkpoint_sha256=weight,method='closed_reference'),path);timing['write_seconds']+=time.perf_counter()-tick
                    tick=time.perf_counter();digest=sha(path);timing['hash_seconds']+=time.perf_counter()-tick;completed+=1
                    manifest.write(json.dumps(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,identity=identity))+'\n');manifest.flush()
                    dump('status.json',dict(status='encoding_closed12_full_WSI',worker_pid=pid,completed=completed,total=total,wsi_id=r['wsi_id'],tile=t['number'],seconds=time.perf_counter()-started))
                    if completed%20==1 or a.gate:print(json.dumps(dict(completed=completed,total=total,wsi_id=r['wsi_id'],seconds=time.perf_counter()-started)),flush=True)
        result=dict(status='passed_closed12_production_cache_gate' if a.gate else 'complete_closed12_full_WSI_cache_execution',worker_pid=pid,signature=signature,tiles=completed,gate_errors=gate_errors,timing=timing,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,total_seconds=time.perf_counter()-started,remaining='Independent full-cache acceptance and whole-slide probability raster/GT quality metrics not completed')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],tiles=completed,seconds=result['total_seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','acceptance','native-gate','native-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);run(p.parse_args())
