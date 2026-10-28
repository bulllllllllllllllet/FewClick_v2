"""Independent all appended closed logits and fresh native model probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyvips
import torch

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
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    require(not torch.cuda.is_available(),'CPU same native selected model')
    s=read(a.source/'summary.json');gate=s['status']=='passed_closed12_extension_native_gate'
    require(gate or s['status']=='complete_closed12_extension_cache_unaccepted','completed actual extension cache')
    for p,h in s['signature']['sources'].items():require(sha(p)==h,'unchanged all append-cache sources')
    if not gate:
        g=read(a.verified_gate/'summary.json') if a.verified_gate else {}
        require(g.get('status')=='accepted_closed12_extension_gate' and g['source_sha256']==sha(__file__),'same independent native/logit audit gate')
    root=a.root;inputs=root/'full_valid_inputs_full_1028_034700'
    require(sha(inputs/'summary.json')==s['signature']['input_summary_sha256'] and sha(root/'full_valid_inputs_acceptance_1028_034700/summary.json')==s['signature']['input_acceptance_sha256'],'full input geometry independently accepted')
    all_inputs=[json.loads(line) for line in (inputs/'inputs.jsonl').open()]
    if gate:
        keys={r['key'] for r in read(root/'full_valid_inputs_gate_1028_034700/native_evidence.json')}
        expected=[r for r in all_inputs if r['key'] in keys]
    else:expected=[r for r in all_inputs if r['preparation']!='remote_reuse_candidate']
    require(sha(a.source/'cache_manifest.jsonl')==s['cache_manifest_sha256'],'all new cache rows bound')
    entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()]
    require(len(entries)==len(expected)==s['tiles']==(8 if gate else 8244),'all original extension input slots')
    base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(base/'adaptation_1025_131451/train_closed_reference_1025_131451/best_checkpoint.json')
    require(sha(best['path'])==best['checkpoint_sha256']==s['signature']['weight_sha256']=='0aa11b35866262648d2b56340a0d4501165da0516cf21df7bdfebf8da05d06a4','same fixed selected closed model')
    for n,(e,r) in enumerate(zip(entries,expected)):
        d=e['identity'];require(e['full_valid_key']==r['key'] and e['preparation']==r['preparation'],'every canonical extension preparation identity')
        require((d['wsi_id'],d['tile'],d['source_wsi'],d['native_box'],d['native_mpp'])==(r['wsi_id'],r['tile'],r['source_wsi'],r['native_boxes'][0],r['native_mpp']),'every original source/box/MPP')
        require(Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256'],'every actual saved cache byte')
        payload=torch.load(e['path'],map_location='cpu',weights_only=False)
        require(payload['identity']==d and payload['checkpoint_sha256']==best['checkpoint_sha256'] and payload['method']=='closed_reference','all real method/weight/identity bindings')
        z=payload['logits'];require(z.shape==(1,12,512,512) and z.dtype==torch.float32 and torch.isfinite(z).all(),'every real full12 FP32 logit tensor')
        if n%20==0:(a.output/'status.json').write_text(json.dumps(dict(status='independent_extension_scan',worker_pid=os.getpid(),completed=n+1,total=len(entries)))+'\n')
    model=load('closed_reference',Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'))
    positions=list(range(len(entries))) if gate else sorted({0,len(entries)//4,len(entries)//2,3*len(entries)//4,len(entries)-1});checks=[]
    mean=torch.tensor([.485,.456,.406],dtype=torch.float32)[:,None,None];std=torch.tensor([.229,.224,.225],dtype=torch.float32)[:,None,None]
    with torch.inference_mode():
        for position in positions:
            e=entries[position];d=e['identity'];box=d['native_box'];slide=pyvips.Image.new_from_file(d['source_wsi'],access='random')
            im=slide.crop(*box).extract_band(0,n=3);native=np.frombuffer(im.write_to_memory(),np.uint8).reshape(box[3],box[2],3)
            rgb=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR)).copy()
            require(hashlib.sha256(rgb.tobytes()).hexdigest()==d['RGB_array_sha256'],'fresh native RGB pixels')
            image=torch.from_numpy(rgb.transpose(2,0,1)).float().div(255.)
            normalized=((image-mean)/std)[None]
            direct=model.network(normalized)['out'];saved=torch.load(e['path'],map_location='cpu',weights_only=False)['logits']
            error=float((direct-saved).abs().max())
            require(error==0 and torch.equal(direct.softmax(1),saved.softmax(1)) and torch.equal(direct.argmax(1),saved.argmax(1)),'all fresh independent normalization/network logits/softmax/categorical exact')
            checks.append(dict(key=e['full_valid_key'],max_logit_error=error,softmax_exact=True,categorical_exact=True))
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    result=dict(status='accepted_closed12_extension_gate' if gate else 'accepted_closed12_extension_cache_qualified',worker_pid=os.getpid(),source_sha256=sha(__file__),source_summary_sha256=sha(a.source/'summary.json'),tiles=len(entries),native_probes=len(checks),checks_sha256=sha(a.output/'checks.json'),seconds=time.perf_counter()-start,
                scope='Every appended cache identity/bytes/finite12 logits; fresh independent explicit ImageNet normalization/network on all8 gate or fixed5 full probes; full encoder replay not exhaustive',remaining='Merge original20208 plus appended8244 exactly once, full28452 raster/originalGT quality and independent acceptance')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
