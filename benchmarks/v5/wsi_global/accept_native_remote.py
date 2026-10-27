"""Replay every WSI cache gate output with live encoder hooks and bad-cache rejection."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import time

import numpy as np
import torch
from .cache_gate import load_cache
from .native_remote_gate import decode,prompts
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4);check(not torch.cuda.is_available(),'CPU independent gate')
    accepted=read(a.gate/'summary.json');check(accepted['status']=='passed_WSI_native_remote_cache_gate','all four methods completed')
    for p,d in accepted['source_sha256'].items():check(sha(p)==d,'source unchanged')
    check(sha(a.gate/'outputs.json')==accepted['output_manifest_sha256'],'output manifest')
    outputs=read(a.gate/'outputs.json');identities=read(a.gate/'input_identities.json');cases=read(a.gate/'cases.json')
    base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');results=[];checksums={};replayed=0
    for item in accepted['methods']:
        method=item['method'];folder=a.gate/method;best=read(base/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
        check(sha(best['path'])==best['checkpoint_sha256']==item['checkpoint_sha256'],'real development weight binding')
        model=load(method,Path(best['path']),torch.device('cpu'),sha(base/'tasks_1025_125400/contract.json'));cached={}
        for identity,record in identities.items():
            path=folder/(identity.replace(':','_')+'.pth')
            if method=='closed_reference' and ':query:' not in identity:continue
            check(path.exists(),'complete expected cache');cached[identity]=load_cache(path,best['checkpoint_sha256'],record);checksums[str(path)]=sha(path)
        # Negative cases run the same cache consumer used for the real replay.
        identity=next(iter(cached));path=folder/(identity.replace(':','_')+'.pth');rejected=0
        for weight,metadata in [('0'*64,identities[identity]),(best['checkpoint_sha256'],identities[identity]|{'wrong_input_probe':True})]:
            try:load_cache(path,weight,metadata)
            except ValueError:rejected+=1
        check(rejected==2,'wrong checkpoint and wrong input rejected')
        calls=[0]
        def hook(module,args):calls[0]+=1
        modules=[e.backbone for e in model.core.encoders] if method=='fewclick_m3d_image' else [model.network.backbone] if method=='closed_reference' else [model.sam.image_encoder]
        hooks=[module.register_forward_pre_hook(hook) for module in modules];errors=[]
        with torch.inference_mode():
            if method=='closed_reference':
                for row in [q for q in outputs if q['method']==method]:
                    check(sha(row['path'])==row['sha256'],'saved closed probability SHA');saved=np.load(row['path'],allow_pickle=False);expected=cached[row['query']].softmax(1)[0].numpy()
                    check(np.array_equal(saved,expected) and np.array_equal(saved.argmax(0),expected.argmax(0)),'all closed channels and argmax');replayed+=1;errors.append(0.)
            else:
                for index,case in enumerate(cases):
                    for budget in ('3+2','1+1','3+1','5+1'):
                        actual=decode(model,method,cached[case['source']],cached[case['query']],prompts(case['plan'],budget))
                        for name,value in actual.items():
                            row=next(q for q in outputs if q['method']==method and q['case']==index and q['budget']==budget and q['variant']==name)
                            check(sha(row['path'])==row['sha256'],'saved remote probability SHA');saved=np.load(row['path'],allow_pickle=False);expected=value.numpy()
                            error=float(np.max(np.abs(saved-expected)));check(error==0 and np.array_equal(saved>=.5,expected>=.5),'every probability and mask replay');replayed+=1;errors.append(error)
        for hook_handle in hooks:hook_handle.remove()
        check(calls[0]==0,'actual image encoder hooks observed zero during cached decode')
        results.append(dict(method=method,outputs=len(errors),maximum_error=max(errors),encoder_forward_hook_calls=calls[0],negative_cache_cases=rejected,cache_files=len(cached)))
        del model,cached
    check(replayed==len(outputs)==accepted['outputs'],'every output accepted')
    result=dict(status='accepted_WSI_native_remote_cache_gate',worker_pid=os.getpid(),methods=results,outputs=replayed,cache_sha256=checksums,gate_summary_sha256=sha(a.gate/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-started,
        scope='Fresh selected model, all saved512 probability/mask outputs, encoder-hook-zero cached decode, wrong-weight/input rejection; NoCell/custom SAM prototypes/closed12 only',remaining='Full-cell main30 gate and full WSI quality/GT/component scoring not completed')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],outputs=replayed,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
