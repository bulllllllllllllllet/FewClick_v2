"""Append image-only valid4x caches with native support-as-query equivalence."""
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
from .interactive_cache import device_encode, tree, tensor_leaves
from .native_remote_gate import raw
from benchmarks.v5.external_evaluation.evaluate_adapted import load

VIEWS={'fewclick_m3d_image':3,'sam_ft':1,'wsi_sam_ft':2}


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    base=read(a.base_cache/'summary.json');receipt=read(a.base_acceptance/'summary.json');sig=base['signature'];method=sig['method'];device=torch.device(sig['device']);views=VIEWS[method]
    require(receipt['source_summary_sha256']==sha(a.base_cache/'summary.json') and receipt['status'] in ('accepted_interactive_cache_fixture','accepted_interactive_main30_cache_qualified'),'independently accepted base cache')
    require(a.gate or receipt['status']=='accepted_interactive_main30_cache_qualified','full20456 base cache required for extension production')
    for p,d in sig['sources'].items():require(sha(p)==d,'unchanged original method/encoder source')
    if device.type=='cuda':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'),'one permitted GPU');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    else:require(not torch.cuda.is_available() and a.gate,'CPU representative gate only')
    root=a.root;full=root/'full_valid_inputs_full_1028_034700';accepted=root/'full_valid_inputs_acceptance_1028_034700/summary.json'
    require(read(accepted)['source_summary_sha256']==sha(full/'summary.json') and read(full/'summary.json')['inputs_sha256']==sha(full/'inputs.jsonl'),'all28452 geometry independently accepted')
    require(sha(a.base_cache/'cache_manifest.jsonl')==base['cache_manifest_sha256'],'complete original cache manifest')
    old={e['identity']['key']:e for e in map(json.loads,(a.base_cache/'cache_manifest.jsonl').open())}
    all_rows=[json.loads(line) for line in (full/'inputs.jsonl').open()]
    if a.gate:
        aliases=[]
        for prep in ('support_reuse_candidate','remote_reuse_candidate'):
            choices=[r for r in all_rows if r['preparation']==prep and r['existing_input_key'] in old]
            require(len(choices)>=2,'two actual existing source and query aliases');aliases+=choices[:2]
        new=[r for r in all_rows if r['preparation']=='new_input'];chosen={r['key']:r for r in [new[0],new[-1],new[len(new)//2],next(r for r in new if r['native_boxes'][0][2]%2)]}
        for r in new:
            if len(chosen)==4:break
            chosen.setdefault(r['key'],r)
        rows=aliases+list(chosen.values())
    else:rows=[r for r in all_rows if r['preparation']=='new_input']
    require(len(rows)==(8 if a.gate else 7996),'all native gate or additional inputs')
    sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/interactive_cache.py'),Path('benchmarks/v5/wsi_global/native_remote_gate.py')]}
    signature=dict(torch=torch.__version__,cuda=torch.version.cuda,sources=sources,method=method,device=str(device),weight_sha256=sig['weight_sha256'],base_summary_sha256=sha(a.base_cache/'summary.json'),base_acceptance_sha256=sha(a.base_acceptance/'summary.json'),input_summary_sha256=sha(full/'summary.json'))
    if not a.gate:
        require(a.verified_gate is not None and a.gate_acceptance is not None,'actual same-device extension and independent gate required')
        g=read(a.verified_gate/'summary.json');ga=read(a.gate_acceptance/'summary.json')
        require(g['status']=='passed_interactive_valid_extension_gate' and g['signature']==signature and ga['status']=='accepted_interactive_valid_extension_gate' and ga['source_summary_sha256']==sha(a.verified_gate/'summary.json'),'same full-base GPU recipe gate accepted')
    atomic_json(a.output/'contract.json',dict(worker_pid=os.getpid(),signature=signature,gate=a.gate,base_cache=str(a.base_cache),base_acceptance=str(a.base_acceptance),keys=[r['key'] for r in rows],query_GT_access=False,method_scope='NoCell graph/prototype or custom SAM/WSISAM prototype caches; no fullcell detector/feature evidence',evaluation_scope='complete_valid4x_including_support'))
    dev=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(dev/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
    require(sha(best['path'])==best['checkpoint_sha256']==signature['weight_sha256'],'same selected method checkpoint')
    model=load(method,Path(best['path']),device,sha(dev/'tasks_1025_125400/contract.json'));current=None;slide=None;alias_checks=[]
    with torch.inference_mode(),(a.output/'cache_manifest.jsonl').open('x') as manifest:
        for n,r in enumerate(rows):
            for p,h in sources.items():require(sha(p)==h,'frozen extension encoder')
            if current!=r['wsi_id']:
                slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');current=r['wsi_id'];require([slide.width,slide.height]==r['native_wh'],'actual native source dimensions')
            images=[]
            for b in r['native_boxes'][:views]:
                crop=slide.crop(*b).extract_band(0,n=3);pixels=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(b[3],b[2],3)
                images.append(np.asarray(Image.fromarray(pixels).resize((512,512),Image.Resampling.BILINEAR)).copy())
            encoded=device_encode(model,method,raw(images),device);cpu=tree(encoded,lambda t:t.detach().cpu());leaves=tensor_leaves(cpu)
            require(leaves and all(torch.isfinite(t).all() for t in leaves),'all encoded values finite')
            identity=dict(wsi_id=r['wsi_id'],kind='full_valid_query',key=r['key'],geometry=dict(tile=r['tile'],native_boxes=r['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),RGB_array_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images])
            if a.gate and r['existing_input_key'] in old:
                e=old[r['existing_input_key']];require(sha(e['path'])==e['sha256'],'original accepted alias bytes')
                p=torch.load(e['path'],map_location='cpu',weights_only=False)
                require(p['checkpoint_sha256']==sig['weight_sha256'] and p['method']==method and p['identity']['RGB_array_sha256']==identity['RGB_array_sha256'],'same model and exact native pixels for source/query alias')
                expected=tensor_leaves(p['encoded']);require(len(leaves)==len(expected) and all(torch.equal(x,y) for x,y in zip(leaves,expected)),'source-as-query or original-query encoder tensors exactly equal')
                alias_checks.append(dict(key=r['key'],original_key=r['existing_input_key'],native_encoded_exact=True))
            folder=a.output/r['wsi_id'];folder.mkdir(exist_ok=True);path=folder/f'tile_{r["tile"]:06d}.pth';torch.save(dict(encoded=cpu,identity=identity,checkpoint_sha256=sig['weight_sha256'],method=method),path)
            if n==0:require(shutil.disk_usage(a.output).free>1.5*path.stat().st_size*len(rows),'additional native cache disk headroom')
            manifest.write(json.dumps(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,identity=identity,preparation=r['preparation']))+'\n');manifest.flush()
            atomic_json(a.output/'status.json',dict(status='encoding_interactive_valid_extension',worker_pid=os.getpid(),method=method,completed=n+1,total=len(rows),key=r['key']))
    require(not a.gate or len(alias_checks)==4,'both support and remote cache equivalence gates')
    result=dict(status='passed_interactive_valid_extension_gate' if a.gate else 'complete_interactive_valid_extension_unaccepted',worker_pid=os.getpid(),signature=signature,items=len(rows),alias_checks=alias_checks,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),seconds=time.perf_counter()-start,remaining='Independent fresh/native cache acceptance and all28452 source/query alias index; full-valid predictions and quality unaccepted')
    atomic_json(a.output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','base-cache','base-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);p.add_argument('--gate-acceptance',type=Path);run(p.parse_args())
