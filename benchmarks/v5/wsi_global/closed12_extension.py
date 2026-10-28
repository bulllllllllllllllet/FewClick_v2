"""Append8244 fine-view closed-reference caches; preserve original remote cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import pyvips
import torch

from .completion import atomic_json
from .native_remote_gate import raw, encode
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    require(not torch.cuda.is_available(),'original CPU4 FP32 recipe, CUDA hidden')
    root=a.root;inputs=root/'full_valid_inputs_full_1028_034700';accepted=root/'full_valid_inputs_acceptance_1028_034700/summary.json'
    s=read(inputs/'summary.json');receipt=read(accepted)
    require(receipt['status']=='accepted_all28452_full_valid_inputs_and_native24_bridge' and receipt['source_summary_sha256']==sha(inputs/'summary.json') and s['inputs_sha256']==sha(inputs/'inputs.jsonl'),'all28452 native inputs independently accepted')
    rows=[json.loads(line) for line in (inputs/'inputs.jsonl').open()]
    if a.gate:
        chosen={r['key'] for r in read(root/'full_valid_inputs_gate_1028_034700/native_evidence.json')}
        rows=[r for r in rows if r['key'] in chosen]
    else:rows=[r for r in rows if r['preparation']!='remote_reuse_candidate']
    require(len(rows)==(8 if a.gate else 8244),'all fixed additional inputs or representative8')
    base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');training=base/'adaptation_1025_131451/train_closed_reference_1025_131451'
    best=read(training/'best_checkpoint.json');done=read(training/'summary.json')
    require(done['status']=='complete' and done['step']==10000 and done['training_consumed_sha256']==sha(training/'training_consumed.jsonl')=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','same completed training/actual40k consumption')
    require(sha(best['path'])==best['checkpoint_sha256']=='0aa11b35866262648d2b56340a0d4501165da0516cf21df7bdfebf8da05d06a4','original selected closed-reference weight')
    files=[Path(__file__)]+[Path('benchmarks/v5/wsi_global')/n for n in ('native_remote_gate.py','cache_gate.py','completion.py')]+[Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),Path('benchmarks/v5/fair_comparison/adapted_models.py')]
    signature=dict(sources={str(p):sha(p) for p in files},input_summary_sha256=sha(inputs/'summary.json'),input_acceptance_sha256=sha(accepted),weight_sha256=best['checkpoint_sha256'],device='CPU4 FP32')
    if not a.gate:
        require(a.verified_gate is not None and a.gate_acceptance is not None,'actual gate and independent fresh-model acceptance required')
        g=read(a.verified_gate/'summary.json');ga=read(a.gate_acceptance/'summary.json')
        require(g['status']=='passed_closed12_extension_native_gate' and g['signature']==signature and ga['status']=='accepted_closed12_extension_gate' and ga['source_summary_sha256']==sha(a.verified_gate/'summary.json'),'same append-cache recipe actual8gate accepted')
    require(shutil.disk_usage(a.output).free>1.5*len(rows)*12*512*512*4,'extra logits disk headroom')
    atomic_json(a.output/'contract.json',dict(worker_pid=os.getpid(),signature=signature,gate=a.gate,keys=[r['key'] for r in rows],query_GT_access=False,extra_supervision='privileged12class CE and targetID, no clicks',scope='Complete valid domain includes source area, separate from strict remote benchmark'))
    model=load('closed_reference',Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'))
    native_gate=root/'native_remote_gate_exact_1027_175300';old_accept=read(root/'native_remote_acceptance_1027_175900/summary.json')
    current=None;slide=None;checks=[];timing=dict(native_read_and_normalize=0.,encode=0.,write_and_hash=0.)
    with torch.inference_mode(),(a.output/'cache_manifest.jsonl').open('x') as manifest:
        for n,r in enumerate(rows):
            for path,digest in signature['sources'].items():require(sha(path)==digest,'append-cache source remains frozen')
            tick=time.perf_counter()
            if current!=r['wsi_id']:
                slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');current=r['wsi_id'];require([slide.width,slide.height]==r['native_wh'],'actual source native dimensions')
            box=r['native_boxes'][0];crop=slide.crop(*box).extract_band(0,n=3)
            native=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(box[3],box[2],3)
            image=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR)).copy();values=raw([image]);timing['native_read_and_normalize']+=time.perf_counter()-tick
            tick=time.perf_counter();logits=encode(model,'closed_reference',values);timing['encode']+=time.perf_counter()-tick
            require(logits.shape==(1,12,512,512) and logits.dtype==torch.float32 and torch.isfinite(logits).all(),'all finite FP32 twelve-class logits')
            identity=dict(wsi_id=r['wsi_id'],tile=r['tile'],source_wsi=r['source_wsi'],native_box=box,native_mpp=r['native_mpp'],RGB_array_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            folder=a.output/r['wsi_id'];folder.mkdir(exist_ok=True);path=folder/f'tile_{r["tile"]:06d}.pth';tick=time.perf_counter()
            torch.save(dict(logits=logits,identity=identity,checkpoint_sha256=best['checkpoint_sha256'],method='closed_reference'),path)
            if a.gate:
                disk=torch.load(path,map_location='cpu',weights_only=False);require(torch.equal(disk['logits'],logits) and torch.equal(disk['logits'].softmax(1),logits.softmax(1)) and torch.equal(disk['logits'].argmax(1),logits.argmax(1)),'actual disk all-logit/softmax/categorical equality')
                old=native_gate/'closed_reference'/((str(r['existing_input_key']) if r['existing_input_key'] else '').replace(':','_')+'.pth')
                if old.exists():
                    require(sha(old)==old_accept['cache_sha256'][str(old)],'independently accepted original native cache')
                    expected=torch.load(old,map_location='cpu',weights_only=False)['encoded'];require(torch.equal(logits,expected),'old native versus extension exact12 logits')
                    checks.append(dict(key=r['key'],old_native_exact=True))
            digest=sha(path);timing['write_and_hash']+=time.perf_counter()-tick
            manifest.write(json.dumps(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,identity=identity,full_valid_key=r['key'],preparation=r['preparation']))+'\n');manifest.flush()
            atomic_json(a.output/'status.json',dict(status='encoding_closed12_valid_domain_extension',worker_pid=os.getpid(),completed=n+1,total=len(rows),key=r['key'],seconds=time.perf_counter()-started))
    require(not a.gate or len(checks)>=1,'representative gate bridges previous accepted native logits')
    result=dict(status='passed_closed12_extension_native_gate' if a.gate else 'complete_closed12_extension_cache_unaccepted',worker_pid=os.getpid(),signature=signature,tiles=len(rows),old_native_checks=checks,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),timing=timing,seconds=time.perf_counter()-started,remaining='Independent all-cache and fresh native acceptance, merge28452 manifest and full-valid raster/GT scores; no whole-valid quality accepted')
    atomic_json(a.output/'summary.json',result);atomic_json(a.output/'status.json',dict(status=result['status'],worker_pid=os.getpid()));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);p.add_argument('--gate-acceptance',type=Path);run(p.parse_args())
