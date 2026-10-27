"""Independent complete closed12 cache identity/finite-value and native probe audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

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
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();pid=os.getpid();torch.set_num_threads(4)
    source=read(a.source/'summary.json');contract=read(a.source/'contract.json');gate=source['status']=='passed_closed12_production_cache_gate'
    require(gate or source['status']=='complete_closed12_full_WSI_cache_execution','finished actual cache')
    for p,d in source['signature']['source_sha256'].items():require(sha(p)==d,'producer code signature')
    require(sha(a.protocol/'summary.json')==source['signature']['protocol_sha256'] and sha(a.source/'cache_manifest.jsonl')==source['cache_manifest_sha256'],'complete immutable input/output identities')
    inputs=read(a.protocol/'query_inputs.json');expected={}
    for r in inputs:
        for line in Path(r['grid']).open():
            t=json.loads(line);expected[r['wsi_id'],t['number']]=(r,t)
    lines=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()];seen=set();paths={}
    require(len(lines)==source['tiles']==(4 if gate else 20208),'complete expected tile count')
    for index,entry in enumerate(lines):
        identity=entry['identity'];key=(identity['wsi_id'],identity['tile']);require(key in expected and key not in seen,'unique true query tile');seen.add(key)
        r,t=expected[key];require(identity['native_box']==t['native_boxes'][0] and identity['source_wsi']==r['source_wsi'] and identity['native_mpp']==r['native_mpp'],'all actual input geometries')
        path=Path(entry['path']);require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'every immutable cache byte')
        value=torch.load(path,map_location='cpu',weights_only=False);require(value['identity']==identity and value['checkpoint_sha256']==source['signature']['weight_sha256'] and value['method']=='closed_reference','actual encoded cache method/input/checkpoint')
        z=value['logits'];require(z.shape==(1,12,512,512) and z.dtype==torch.float32 and torch.isfinite(z).all(),'every tensor shape/dtype/finite')
        paths[key]=path
        if index%20==0:(a.output/'status.json').write_text(json.dumps(dict(status='independent_cache_scan',worker_pid=pid,completed=index+1,total=len(lines)))+'\n')
    require(gate or seen==set(expected),'full30 every query tile present')
    base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(base/'adaptation_1025_131451/train_closed_reference_1025_131451/best_checkpoint.json')
    require(sha(best['path'])==best['checkpoint_sha256']==source['signature']['weight_sha256'],'accepted actual selected weights')
    model=load('closed_reference',Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'));keys=list(paths);probes=keys if gate else [keys[i] for i in sorted({0,len(keys)//4,len(keys)//2,3*len(keys)//4,len(keys)-1})];errors=[]
    with torch.inference_mode():
        for key in probes:
            value=torch.load(paths[key],map_location='cpu',weights_only=False);identity=value['identity'];b=identity['native_box'];image=pyvips.Image.new_from_file(identity['source_wsi'],access='random').crop(*b).extract_band(0,n=3)
            rgb=np.frombuffer(image.write_to_memory(),np.uint8).reshape(b[3],b[2],3);rgb=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)).copy()
            require(hashlib.sha256(rgb.tobytes()).hexdigest()==identity['RGB_array_sha256'],'actual fresh native RGB SHA')
            direct=encode(model,'closed_reference',raw([rgb]));error=float((value['logits']-direct).abs().max());require(error==0,'fresh selected-model native12 tensor replay');errors.append(dict(wsi_id=key[0],tile=key[1],maximum_error=error))
    result=dict(status='accepted_closed12_cache_gate' if gate else 'accepted_closed12_full_WSI_cache_qualified',worker_pid=pid,tiles=len(lines),native_probes=errors,source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-start,remaining='No whole-slide quality/GT/component scores yet')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],tiles=len(lines),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','protocol','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
