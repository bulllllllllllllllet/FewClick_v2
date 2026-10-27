"""Independent cache identity/footprint/tensor audit and native model probes.

Does not import the cache producer or its encoding bridge. All cache bytes and
schemas are checked. Fresh native query RGB is reread only at fixed probes;
all saved support RGB images are reread and hashed. No query GT is accessed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import pyvips
import torch
import torch.nn.functional as F

from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def schema(node):
    if isinstance(node,torch.Tensor):
        require(node.device.type=='cpu' and torch.isfinite(node).all(), 'all saved tensors finite on CPU')
        return dict(shape=list(node.shape),dtype=str(node.dtype))
    if isinstance(node,dict):
        return {k:schema(v) for k,v in node.items()}
    if isinstance(node,(tuple,list)):
        return dict(container=type(node).__name__,values=[schema(v) for v in node])
    raise TypeError('unexpected cache tree node '+str(type(node)))


def equal_tree(saved, fresh):
    if isinstance(saved,torch.Tensor):
        require(isinstance(fresh,torch.Tensor) and saved.shape==fresh.shape and saved.dtype==fresh.dtype and torch.equal(saved,fresh.detach().cpu()), 'every native-recomputed tensor equals saved cache exactly')
    elif isinstance(saved,dict):
        require(isinstance(fresh,dict) and saved.keys()==fresh.keys(), 'encoded dict identity')
        for k in saved:
            equal_tree(saved[k],fresh[k])
    else:
        require(type(saved)==type(fresh) and len(saved)==len(fresh), 'encoded sequence structure')
        for x,y in zip(saved,fresh):
            equal_tree(x,y)


def native_encode(model, method, images, device):
    if method=='fewclick_m3d_image':
        values={}
        for level,im in enumerate(images):
            suffix='' if level==0 else f'_{level}'
            values['image'+suffix]=he_to_tensor(im.copy())[None].to(device)
            values['cells'+suffix]=torch.zeros(1,1,74,device=device)
            values['cell_valid'+suffix]=torch.zeros(1,1,dtype=torch.bool,device=device)
            values['total_cell_count' if level==0 else f'total_{level}']=torch.zeros(1,device=device)
        return model.core.encode_image(values)
    if method=='sam_ft':
        transformed=model.transform.apply_image(images[0])
        chw=torch.from_numpy(transformed).permute(2,0,1).to(device)
        return model.sam.image_encoder(model.sam.preprocess(chw)[None])
    embeddings=[]
    for im in images:
        tensor=torch.from_numpy(im.copy()).permute(2,0,1)[None]
        resized=F.interpolate(tensor.float(),size=(1024,1024),mode='bilinear',align_corners=False)
        pixels=resized.round().clamp(0,255).to(dtype=torch.uint8)[0].to(device)
        value,_=model.sam.image_encoder(model.sam.preprocess(pixels)[None]);embeddings.append(value.float())
    fine,low=embeddings;h,w=fine.shape[2:]
    rows=(torch.arange(h,device=device)+.5)/h-.5
    cols=(torch.arange(w,device=device)+.5)/w-.5
    grid_y,grid_x=torch.meshgrid(rows,cols,indexing='ij')
    low_at_fine=F.grid_sample(low,torch.stack((grid_x,grid_y),dim=-1)[None],align_corners=False)
    return torch.cat((F.normalize(fine,dim=1),F.normalize(low_at_fine,dim=1)),dim=1)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        torch.set_num_threads(4);pyvips.cache_set_max(0)
        source=read(a.source/'summary.json');fixture=source['status']=='passed_interactive_native_cache_fixture'
        require(fixture or source['status']=='complete_interactive_main30_cache_unaccepted', 'actual completed cache producer')
        sig=source['signature'];method=sig['method'];device=torch.device(sig['device'])
        require(method in ('fewclick_m3d_image','sam_ft','wsi_sam_ft'), 'image-only or explicit encoder prototype method')
        require(sig['protocol_sha256']==sha(a.protocol/'summary.json') and sig['native_acceptance_sha256']==sha(a.native_acceptance/'summary.json'), 'original protocols/native acceptance bound')
        require(read(a.native_acceptance/'summary.json')['gate_summary_sha256']==sha(a.native_gate/'summary.json'), 'actual independently accepted native four-method gate')
        for p,d in sig['sources'].items():
            require(sha(p)==d, 'all cache producer sources frozen')
        if device.type=='cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'), 'one permitted physical GPU')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        else:
            require(not torch.cuda.is_available(), 'explicit CPU gate with CUDA hidden')
        require(sig['torch']==torch.__version__ and sig['cuda']==torch.version.cuda and sig['matmul_TF32']==torch.backends.cuda.matmul.allow_tf32 and sig['cudnn_TF32']==torch.backends.cudnn.allow_tf32, 'same actual numeric device/runtime contract')
        audit_recipe=dict(source_sha256=sha(__file__),producer_sources=sig['sources'],method=method,device=str(device),numpy=np.__version__,torch=torch.__version__)
        if not fixture:
            require(a.verified_gate is not None, 'actual same-method/device independent cache audit fixture')
            g=read(a.verified_gate/'summary.json')
            require(g['status']=='accepted_interactive_cache_fixture' and g['recipe']==audit_recipe, 'independent cache auditor gate matches')
        require(sha(a.source/'cache_manifest.jsonl')==source['cache_manifest_sha256'], 'complete committed manifest bytes')
        entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()]
        indexed={e['identity']['key']:e for e in entries}
        require(len(indexed)==len(entries)==(7 if fixture else 20456), 'all unique native support/query cache keys')
        native_keys=set(read(a.native_gate/'input_identities.json'))
        inputs=read(a.protocol/'query_inputs.json');expected={};specs={};views=3 if method=='fewclick_m3d_image' else 1 if method=='sam_ft' else 2
        support_rgb={}
        for r in inputs:
            sid=r['wsi_id'];require(sha(r['grid'])==r['grid_sha256'], 'complete accepted query geometry')
            for s in r['support_inputs']:
                key=s['support_id']
                if fixture and key not in native_keys:
                    continue
                images=[]
                for d in s['images'][:views]:
                    require(sha(d['path'])==d['sha256'], 'every actual support PNG byte')
                    image=np.asarray(Image.open(d['path']).convert('RGB')).copy();require(image.shape==(512,512,3), 'actual native support shape');images.append(image)
                hashes=[hashlib.sha256(im.tobytes()).hexdigest() for im in images]
                expected[key]=dict(wsi_id=sid,kind='support',key=key,geometry=dict(source_record=s,views=s['images'][:views]),RGB_array_sha256=hashes)
                support_rgb[key]=hashes;specs[key]=(r,'support',s)
            for t in (json.loads(line) for line in Path(r['grid']).open()):
                key=sid+':query:'+str(t['number'])
                if fixture and key not in native_keys:
                    continue
                require(t['native_boxes'][2][0]>=r['source_cut_native']+r['gap_native'], 'full 4x query footprint remains outside source')
                expected[key]=dict(wsi_id=sid,kind='query',key=key,geometry=dict(tile=t['number'],native_boxes=t['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']))
                specs[key]=(r,'query',t)
        require(set(expected)==set(indexed), 'complete fixed protocol input set without omissions/extra caches')
        schemas={};total_bytes=0
        for number,e in enumerate(entries):
            key=e['identity']['key'];identity=e['identity'];canonical=expected[key]
            actual_metadata={k:v for k,v in identity.items() if k!='RGB_array_sha256'}
            expected_metadata={k:v for k,v in canonical.items() if k!='RGB_array_sha256'}
            require(actual_metadata==expected_metadata, 'every native slide/source/query key and input footprint')
            hashes=identity['RGB_array_sha256'];require(len(hashes)==views and all(len(h)==64 and set(h)<=set('0123456789abcdef') for h in hashes), 'all query RGB identity schemas')
            if key in support_rgb:
                require(hashes==support_rgb[key], 'all support image array identities')
            path=Path(e['path']);require(path.stat().st_size==e['bytes'] and sha(path)==e['sha256'], 'every actual encoded file byte/hash/size')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            require(payload['identity']==identity and payload['checkpoint_sha256']==sig['weight_sha256'] and payload['method']==method, 'actual saved model/input/method identity')
            current=schema(payload['encoded'])
            require(not schemas or current==schemas, 'all source/query tensor shapes/dtypes identical to fixed 512-view encoding')
            schemas=current;total_bytes+=e['bytes'];del payload
            if number%20==0 or fixture:
                dump('status.json',dict(status='independent_interactive_cache_bytes_and_geometry',worker_pid=pid,method=method,completed=number+1,total=len(entries)))
        require(total_bytes==source['bytes'], 'complete actual storage total')
        supports=[k for k in expected if expected[k]['kind']=='support'];queries=[k for k in expected if expected[k]['kind']=='query']
        require((len(supports),len(queries))==((3,4) if fixture else (248,20208)), 'exact source and query coverage')
        probes=list(expected) if fixture else list(dict.fromkeys([supports[0],supports[-1],queries[0],queries[len(queries)//2],queries[-1]]))
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        training=base/f'adaptation_1025_131451/train_{method}_1025_131451';best=read(training/'best_checkpoint.json');done=read(training/'summary.json')
        require(done['status']=='complete' and done['step']==10000 and sha(training/'training_consumed.jsonl')==done['training_consumed_sha256']=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', 'actual common full-label supervision receipt')
        require(sha(best['path'])==best['checkpoint_sha256']==sig['weight_sha256'], 'exact development-selected weights')
        model=load(method,Path(best['path']),device,sha(base/'tasks_1025_125400/contract.json'));evidence=[]
        with torch.inference_mode():
            for key in probes:
                r,kind,item=specs[key];images=[]
                if kind=='support':
                    images=[np.asarray(Image.open(d['path']).convert('RGB')).copy() for d in item['images'][:views]]
                else:
                    slide=pyvips.Image.new_from_file(r['source_wsi'],access='random')
                    for x,y,w,h in item['native_boxes'][:views]:
                        crop=slide.crop(x,y,w,h).extract_band(0,n=3)
                        pixels=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(h,w,3)
                        images.append(np.asarray(Image.fromarray(pixels).resize((512,512),Image.Resampling.BILINEAR)).copy())
                require([hashlib.sha256(im.tobytes()).hexdigest() for im in images]==indexed[key]['identity']['RGB_array_sha256'], 'fresh native image input identity')
                fresh=native_encode(model,method,images,device)
                saved=torch.load(indexed[key]['path'],map_location='cpu',weights_only=False)['encoded'];equal_tree(saved,fresh)
                evidence.append(dict(key=key,native_RGB_exact=True,every_encoded_tensor_exact=True));del fresh,saved
        dump('schema.json',schemas)
        result=dict(status='accepted_interactive_cache_fixture' if fixture else 'accepted_interactive_main30_cache_qualified',worker_pid=pid,recipe=audit_recipe,items=len(entries),query_tiles=len(queries),support_tiles=len(supports),all_cache_bytes_and_tensor_values_checked=True,all_support_RGB_checked=True,fresh_native_probes=evidence,source_summary_sha256=sha(a.source/'summary.json'),schema_sha256=sha(a.output/'schema.json'),seconds=time.perf_counter()-started,
                    scope='Every saved byte, input geometry/method/weight/tensor schema/finiteness; all support RGB and fixed fresh native model probes. No exhaustive re-encoding or queryGT, no WSI quality acceptance',remaining='Cached remote task/decode/native raster/GT scores and full-cell main30; independent output acceptance')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','protocol','native-gate','native-acceptance','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path)
    run(p.parse_args())
