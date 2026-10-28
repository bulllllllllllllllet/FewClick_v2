"""Independently re-read native extension inputs and selected model encodings."""
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

from .accept_interactive_cache import native_encode, schema, equal_tree
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);start=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    source=read(a.source/'summary.json');contract=read(a.source/'contract.json');gate=source['status']=='passed_interactive_valid_extension_gate'
    require(gate or source['status']=='complete_interactive_valid_extension_unaccepted','completed real extension cache')
    sig=source['signature'];method=sig['method'];device=torch.device(sig['device']);views={'fewclick_m3d_image':3,'sam_ft':1,'wsi_sam_ft':2}[method]
    for p,h in sig['sources'].items():require(sha(p)==h,'same producer/encoder bridge sources')
    recipe=dict(source_sha256=sha(__file__),native_encoder_sha256=sha('benchmarks/v5/wsi_global/accept_interactive_cache.py'),signature=sig)
    if not gate:
        g=read(a.verified_gate/'summary.json') if a.verified_gate else {}
        require(g.get('status')=='accepted_interactive_valid_extension_gate' and g['recipe']==recipe,'same device/model/base full-cache independent gate')
    if device.type=='cuda':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'),'one permitted same-model GPU');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    else:require(not torch.cuda.is_available() and gate,'CPU fixture only')
    inputs=a.root/'full_valid_inputs_full_1028_034700';require(sha(inputs/'summary.json')==sig['input_summary_sha256'],'fixed all28452 accepted native input identities')
    all_inputs={r['key']:r for r in map(json.loads,(inputs/'inputs.jsonl').open())}
    require(sha(a.source/'cache_manifest.jsonl')==source['cache_manifest_sha256'],'all new output manifest bytes')
    entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()]
    require(len(entries)==source['items']==(8 if gate else 7996) and [e['identity']['key'] for e in entries]==contract['keys'],'complete ordered native extension records')
    if not gate:require(contract['keys']==[r['key'] for r in all_inputs.values() if r['preparation']=='new_input'],'entire additional7996 geometric set')
    base=Path(contract['base_cache']);ba=Path(contract['base_acceptance']);require(sha(base/'summary.json')==sig['base_summary_sha256'] and sha(ba/'summary.json')==sig['base_acceptance_sha256'] and read(ba/'summary.json')['source_summary_sha256']==sha(base/'summary.json'),'original accepted cache identities')
    original={e['identity']['key']:e for e in map(json.loads,(base/'cache_manifest.jsonl').open())}
    for e in entries:
        d=e['identity'];r=all_inputs[d['key']];require(d['kind']=='full_valid_query' and d['wsi_id']==r['wsi_id'] and e['preparation']==r['preparation'],'exact native field kind/ID')
        require(d['geometry']==dict(tile=r['tile'],native_boxes=r['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),'all physical frame/box/scale identities')
        require(Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256'],'every new cache file byte')
        p=torch.load(e['path'],map_location='cpu',weights_only=False);require(p['identity']==d and p['checkpoint_sha256']==sig['weight_sha256'] and p['method']==method,'all payload identities/model/weight');schema(p['encoded'])
    dev=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(dev/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
    require(sha(best['path'])==best['checkpoint_sha256']==sig['weight_sha256'],'actual selected method model')
    model=load(method,Path(best['path']),device,sha(dev/'tasks_1025_125400/contract.json'));positions=list(range(len(entries))) if gate else sorted({0,len(entries)//4,len(entries)//2,3*len(entries)//4,len(entries)-1});checks=[];aliases=0
    with torch.inference_mode():
        for n in positions:
            e=entries[n];d=e['identity'];r=all_inputs[d['key']];slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');images=[]
            for b in r['native_boxes'][:views]:
                im=slide.crop(*b).extract_band(0,n=3);pixels=np.frombuffer(im.write_to_memory(),np.uint8).reshape(b[3],b[2],3)
                images.append(np.asarray(Image.fromarray(pixels).resize((512,512),Image.Resampling.BILINEAR)).copy())
            require([hashlib.sha256(im.tobytes()).hexdigest() for im in images]==d['RGB_array_sha256'],'all fresh native probe pixels')
            p=torch.load(e['path'],map_location='cpu',weights_only=False);fresh=native_encode(model,method,images,device);equal_tree(p['encoded'],fresh)
            if r['existing_input_key'] in original:
                old=original[r['existing_input_key']];require(sha(old['path'])==old['sha256'],'original source/query alias bytes')
                prior=torch.load(old['path'],map_location='cpu',weights_only=False);require(prior['method']==method and prior['checkpoint_sha256']==sig['weight_sha256'] and prior['identity']['RGB_array_sha256']==d['RGB_array_sha256'],'actual same input/model independent alias')
                equal_tree(prior['encoded'],fresh);aliases+=1
            checks.append(dict(key=r['key'],fresh_native_encoder_exact=True,old_alias_exact=r['existing_input_key'] in original))
    require(not gate or aliases==4,'actual source/query-as-query alias equivalence')
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    result=dict(status='accepted_interactive_valid_extension_gate' if gate else 'accepted_interactive_valid_extension_qualified',worker_pid=os.getpid(),recipe=recipe,source_summary_sha256=sha(a.source/'summary.json'),items=len(entries),native_probes=len(checks),alias_probes=aliases,checks_sha256=sha(a.output/'checks.json'),seconds=time.perf_counter()-start,
                scope='All new encoded/identity/file hashes and schema, independent native encoders all8 fixture/fixed5 full; fixture four original source/query aliases exact. No detector/fullcell or exhaustive full encoder replay claim.',remaining='Bind original20208+248 supports and new7996 to full28452 index, then full-valid model predictions/originalGT quality and remote invariance')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
