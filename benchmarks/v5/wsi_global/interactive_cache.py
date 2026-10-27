"""Native main30 image-only/SAM/WSI-SAM frozen source/query feature caches.

SAM transfer uses the explicit encoder-prototype wrapper, not its native local
prompt decoder. This producer neither reads query labels nor computes quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import pyvips
import torch
import torch.nn.functional as F

from .native_remote_gate import raw, encode
from benchmarks.v5.external_evaluation.evaluate_adapted import load


METHODS = ('fewclick_m3d_image','sam_ft','wsi_sam_ft')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def require(test, reason):
    if not test:
        raise ValueError(reason)


def tree(value, fn):
    if isinstance(value,torch.Tensor):
        return fn(value)
    if isinstance(value,dict):
        return {k:tree(v,fn) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(tree(v,fn) for v in value)
    raise TypeError('unexpected encoded cache node '+str(type(value)))


def tensor_leaves(value):
    leaves=[]
    tree(value,lambda t:leaves.append(t) or t)
    return leaves


def device_encode(model, method, values, device):
    if device.type == 'cpu':
        return encode(model,method,values)
    if method == 'fewclick_m3d_image':
        return model.core.encode_image({k:v.to(device) for k,v in values.items()})
    if method == 'sam_ft':
        # Keep public CPU RGB resize exactly as in sam_features.
        rgb=values['rgb'][0].permute(1,2,0).numpy()
        image=torch.as_tensor(model.transform.apply_image(rgb)).permute(2,0,1).to(device)
        return model.sam.image_encoder(model.sam.preprocess(image)[None])
    features=[]
    for key in ('rgb','rgb_1'):
        # The original WSI-SAM wrapper performs this resize/round on CPU.
        image=F.interpolate(values[key].float(),size=(1024,1024),mode='bilinear',align_corners=False).round().clamp(0,255).to(torch.uint8)
        embedding,_=model.sam.image_encoder(model.sam.preprocess(image[0].to(device))[None])
        features.append(embedding.float())
    fine,low=features;h,w=fine.shape[-2:]
    yy,xx=torch.meshgrid((torch.arange(h,device=device)+.5)/h,(torch.arange(w,device=device)+.5)/w,indexing='ij')
    aligned=F.grid_sample(low,torch.stack([xx-.5,yy-.5],-1)[None],align_corners=False)
    return torch.cat([F.normalize(fine,dim=1),F.normalize(aligned,dim=1)],1)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        torch.set_num_threads(4);pyvips.cache_set_max(0)
        device=torch.device(a.device)
        if device.type=='cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'), 'one permitted physical GPU required')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        else:
            require(not torch.cuda.is_available(), 'CPU gate must hide CUDA')
        accepted=read(a.acceptance/'summary.json');native=read(a.native_acceptance/'summary.json')
        require(accepted['status']=='accepted_source_band_query_protocol' and accepted['slides']==30 and accepted['protocol_summary_sha256']==sha(a.protocol/'summary.json'),'accepted full30 source/query protocol')
        require(native['status']=='accepted_WSI_native_remote_cache_gate' and native['gate_summary_sha256']==sha(a.native_gate/'summary.json'),'independent actual four-method cache gate')
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        training=base/f'adaptation_1025_131451/train_{a.method}_1025_131451';best=read(training/'best_checkpoint.json');done=read(training/'summary.json')
        require(done['status']=='complete' and done['step']==10000 and done['training_consumed_sha256']==sha(training/'training_consumed.jsonl')=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','completed full-label shared40k training')
        require(sha(best['path'])==best['checkpoint_sha256']==next(r['checkpoint_sha256'] for r in read(a.native_gate/'summary.json')['methods'] if r['method']==a.method),'same development-selected checkpoint')
        paths=[Path(__file__),Path('benchmarks/v5/wsi_global/native_remote_gate.py'),Path('benchmarks/v5/wsi_global/cache_gate.py'),Path('benchmarks/v5/heldout_transfer/features.py'),Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py')]
        sources={str(p):sha(p) for p in paths}
        signature=dict(sources=sources,method=a.method,protocol_sha256=sha(a.protocol/'summary.json'),protocol_acceptance_sha256=sha(a.acceptance/'summary.json'),weight_sha256=best['checkpoint_sha256'],device=str(device),torch=torch.__version__,cuda=torch.version.cuda,matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,native_acceptance_sha256=sha(a.native_acceptance/'summary.json'))
        if not a.gate:
            require(a.verified_gate is not None,'actual same-method/device producer gate required')
            g=read(a.verified_gate/'summary.json');require(g['status']=='passed_interactive_native_cache_fixture' and g['signature']==signature,'same frozen cache producer/device gate')
        inputs=read(a.protocol/'query_inputs.json');old=read(a.native_gate/'input_identities.json');old_queries={k for k in old if ':query:' in k};old_supports={k for k in old if ':query:' not in k}
        work=[]
        for r in inputs:
            require(sha(r['grid'])==r['grid_sha256'],'frozen complete remote grid')
            supports=r['support_inputs']
            if a.gate:
                supports=[s for s in supports if s['support_id'] in old_supports]
            for s in supports:
                work.append((r,'support',s))
            for t in (json.loads(line) for line in Path(r['grid']).open()):
                if not a.gate or r['wsi_id']+':query:'+str(t['number']) in old_queries:
                    require(t['native_boxes'][2][0]>=r['source_cut_native']+r['gap_native'],'full query context outside source band')
                    work.append((r,'query',t))
        require(sum(kind=='query' for _,kind,_ in work)==(4 if a.gate else 20208),'all fixed query tiles')
        require(len({(r['wsi_id'],kind,s['support_id'] if kind=='support' else s['number']) for r,kind,s in work})==len(work),'no duplicate cache identities')
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,items=len(work),query_GT_access=False,view_count=3 if a.method=='fewclick_m3d_image' else 1 if a.method=='sam_ft' else 2,scope='NoCell graph+regionproto or explicit custom SAM/WSI-SAM encoder-prototype transfer; no local mask decoder or full-cell claims',failure_policy='stop preserve; no retry/resume/fallback'))
        model=load(a.method,Path(best['path']),device,sha(base/'tasks_1025_125400/contract.json'))
        timing=dict(input_seconds=0.,encode_seconds=0.,write_seconds=0.,hash_seconds=0.)
        native_image=None;current_sid=None;total_bytes=0;gate_errors=[]
        with (a.output/'cache_manifest.jsonl').open('x') as manifest,torch.inference_mode():
            for number,(r,kind,item) in enumerate(work):
                for p,h in sources.items():require(sha(p)==h,'frozen producer source')
                tick=time.perf_counter();sid=r['wsi_id'];views=3 if a.method=='fewclick_m3d_image' else 1 if a.method=='sam_ft' else 2
                if kind=='support':
                    key=item['support_id'];images=[];descriptors=[]
                    for d in item['images'][:views]:
                        require(sha(d['path'])==d['sha256'],'accepted native support RGB bytes')
                        images.append(np.asarray(Image.open(d['path']).convert('RGB')).copy());descriptors.append(dict(d))
                    geometry=dict(source_record=item,views=descriptors)
                else:
                    key=sid+':query:'+str(item['number']);images=[]
                    if current_sid!=sid:
                        native_image=pyvips.Image.new_from_file(r['source_wsi'],access='random');current_sid=sid
                    for box in item['native_boxes'][:views]:
                        crop=native_image.crop(*box).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(box[3],box[2],3)
                        images.append(np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)).copy())
                    geometry=dict(tile=item['number'],native_boxes=item['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi'])
                identity=dict(wsi_id=sid,kind=kind,key=key,geometry=geometry,RGB_array_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images])
                values=raw(images);timing['input_seconds']+=time.perf_counter()-tick
                if device.type=='cuda':torch.cuda.synchronize()
                tick=time.perf_counter();encoded=device_encode(model,a.method,values,device)
                if device.type=='cuda':torch.cuda.synchronize()
                timing['encode_seconds']+=time.perf_counter()-tick
                tensors=tensor_leaves(encoded);require(tensors and all(torch.isfinite(t).all() for t in tensors),'all finite encoded tensors')
                cpu=tree(encoded,lambda t:t.detach().cpu())
                if number==0:
                    estimate=sum(t.numel()*t.element_size() for t in tensor_leaves(cpu))*len(work)
                    require(shutil.disk_usage(a.output).free>1.5*estimate,'actual encoded-cache disk headroom')
                    dump('size_estimate.json',dict(first_encoded_bytes=estimate//len(work),items=len(work),estimated_total_bytes=estimate))
                folder=a.output/sid;folder.mkdir(exist_ok=True);path=folder/(key.replace(':','_')+'.pth')
                tick=time.perf_counter();torch.save(dict(encoded=cpu,identity=identity,checkpoint_sha256=best['checkpoint_sha256'],method=a.method),path);timing['write_seconds']+=time.perf_counter()-tick
                if a.gate:
                    saved=torch.load(path,map_location='cpu',weights_only=False);require(saved['identity']==identity and saved['checkpoint_sha256']==best['checkpoint_sha256'],'actual saved identity/weight')
                    if device.type=='cpu':
                        reference=a.native_gate/a.method/(key.replace(':','_')+'.pth')
                        require(sha(reference)==native['cache_sha256'][str(reference)],'old independently accepted encoded bytes')
                        expected=torch.load(reference,map_location='cpu',weights_only=False)['encoded']
                    else:
                        expected=tree(device_encode(model,a.method,values,device),lambda t:t.detach().cpu())
                    left=tensor_leaves(saved['encoded']);right=tensor_leaves(expected)
                    require(len(left)==len(right) and all(torch.equal(x,y) for x,y in zip(left,right)),'all saved encoded tensors exact to native CPU/fresh GPU path')
                    gate_errors.append(0.)
                tick=time.perf_counter();digest=sha(path);timing['hash_seconds']+=time.perf_counter()-tick;total_bytes+=path.stat().st_size
                manifest.write(json.dumps(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,identity=identity))+'\n');manifest.flush()
                dump('status.json',dict(status='encoding_interactive_main30_cache',worker_pid=pid,method=a.method,completed=number+1,total=len(work),kind=kind,wsi_id=sid,key=key,seconds=time.perf_counter()-started))
                if a.gate or number%20==0:print(json.dumps(dict(completed=number+1,total=len(work),method=a.method,key=key)),flush=True)
                del encoded,cpu,tensors
        result=dict(status='passed_interactive_native_cache_fixture' if a.gate else 'complete_interactive_main30_cache_unaccepted',worker_pid=pid,signature=signature,items=len(work),query_tiles=sum(kind=='query' for _,kind,_ in work),support_tiles=sum(kind=='support' for _,kind,_ in work),cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),bytes=total_bytes,timing=timing,gate_errors=gate_errors,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.perf_counter()-started,remaining='Independent full-cache acceptance, cached remote task/decode and native WSI probability/quality; full-cell main30 separate')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','acceptance','native-gate','native-acceptance','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path)
    run(p.parse_args())
