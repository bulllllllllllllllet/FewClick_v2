"""Independent CRC fullcell cache identities, actual nuclei and encoder probes.

Does not import the new cache producer/assembly helper. Saved reg64 values are
bound and checked when rebuilding74; detector and reg64 extraction are not
rerun, so no detection-history invariance or independent reg64 proof is claimed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import pyvips
import torch

from .audit_fullcell_inputs import geometric_features
from .completion import atomic_json
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:raise ValueError(why)


def audit_view(record,box,native_mpp):
    require(sha(record['path'])==record['sha256'],'all actual instance/feature NPZ bytes')
    with np.load(record['path'],allow_pickle=False) as z:
        require(set(z.files)=={'instances','meta','reg','features','total'},'exact saved cell payload')
        arrays={k:z[k] for k in z.files}
    inst=arrays['instances'];d=record['details'];size=d['detection_size'];pitch=d['detection_mpp_xy']
    require(list(inst.shape)==[size[1],size[0]] and inst.dtype==np.int32,'actual canonical categorical instance grid')
    w,h=box[2:];canonical=[max(1,round(w*native_mpp[0]/.25)),max(1,round(h*native_mpp[1]/.25))]
    require(size==canonical and all(math.isclose(p,q,rel_tol=1e-12) for p,q in zip(pitch,(w*native_mpp[0]/size[0],h*native_mpp[1]/size[1]))),'actual physical detection pitch and source view')
    expected,total=geometric_features(inst,pitch,arrays['reg']);n=len(expected)
    require(arrays['total'].shape==() and int(arrays['total'])==d['total_cell_count']==total and d['retained_cell_count']==n==min(255,total),'uncapped actual and retained cell counts')
    require(arrays['features'].dtype==np.float32 and np.array_equal(arrays['features'],expected),'independent integer-centroid/physical-area/unknown+reg74 reconstruction')
    require(arrays['meta'].dtype==np.float32 and arrays['meta'].shape==(n,4) and np.array_equal(arrays['meta'][:,:3],expected[:,:3]) and np.count_nonzero(arrays['meta'][:,3])==0,'actual nuclear meta has unknown class without tissue GT')
    require(d['nuclear_class']=='0 unknown' and d['cell_feature_dim']==74,'explicit cell-class/feature policy')
    return arrays,dict(scale=record['scale'],total=total,retained=n,shape=list(inst.shape),features_exact=True)


def transfer(x,device):
    if isinstance(x,torch.Tensor):return x.to(device)
    if isinstance(x,dict):return {k:transfer(v,device) for k,v in x.items()}
    return type(x)(transfer(v,device) for v in x)


def compare_tree(left,right):
    if isinstance(left,torch.Tensor):
        require(isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype,'all encoded tensor schemas')
        if left.is_floating_point():
            require(torch.isfinite(left).all() and torch.isfinite(right).all(),'all encoded tensors finite')
            return float((left.cpu()-right.cpu()).abs().max()) if left.numel() else 0.
        require(torch.equal(left.cpu(),right.cpu()),'all encoded discrete tensor values exact');return 0.
    if isinstance(left,dict):
        require(isinstance(right,dict) and left.keys()==right.keys(),'all encoded tree dictionary keys')
        return max((compare_tree(left[k],right[k]) for k in left),default=0.)
    require(type(left)==type(right) and len(left)==len(right),'all encoded sequence schemas')
    return max((compare_tree(x,y) for x,y in zip(left,right)),default=0.)


def finite_schema(value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and torch.isfinite(value).all(),'all actual saved encoded tensor values finite on CPU')
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):return {k:finite_schema(v) for k,v in value.items()}
    require(isinstance(value,(tuple,list)),'expected encoded tree container')
    return dict(container=type(value).__name__,values=[finite_schema(v) for v in value])


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4);pyvips.cache_set_max(0)
    def dump(n,d):atomic_json(a.output/n,d)
    try:
        root=a.root;source=read(a.source/'summary.json');gate=source['status']=='passed_main30_fullcell_cache_gate'
        require(gate or source['status']=='complete_main30_fullcell_cache_unaccepted','completed actual fullcell cache')
        sig=source['signature'];require(os.environ.get('CUDA_VISIBLE_DEVICES')==sig['visible_devices'] and os.environ['CUDA_VISIBLE_DEVICES'] in ('1','3','4','5'),'same permitted physical GPU as producer')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.__version__==sig['torch'] and torch.version.cuda==sig['cuda'] and not sig['matmul_TF32'] and not sig['cudnn_TF32'],'same FP32/TF32 numeric runtime')
        for p,h in sig['sources'].items():require(sha(p)==h,'all fullcell source versions bound')
        for d in sig['weights'].values():require(sha(d['path'])==d['sha256'],'detector/CTP/internalXCell exact weights')
        recipe=dict(source_sha256=sha(__file__),geometry_source_sha256=sha('benchmarks/v5/wsi_global/audit_fullcell_inputs.py'),producer_signature=sig)
        if not gate:
            require(a.verified_gate is not None,'actual independent fullcell GPU cache gate required');g=read(a.verified_gate/'summary.json')
            require(g['status']=='accepted_main30_fullcell_cache_gate' and g['recipe']==recipe,'same actual independent GPU audit recipe')
        inputs=root/'cell_native_inputs_full_1028_012700';require(sha(inputs/'summary.json')==sig['native_input_summary_sha256'],'full native input geometry bound')
        accepted=root/'cell_native_inputs_acceptance_1028_023400/summary.json';require(sha(accepted)==sig['input_acceptance_sha256'] and read(accepted)['source_summary_sha256']==sha(inputs/'summary.json'),'independent native cohort receipt')
        require(sha(a.source/'cache_manifest.jsonl')==source['cache_manifest_sha256'],'all encoded/input manifest bytes')
        entries=[json.loads(line) for line in (a.source/'cache_manifest.jsonl').open()];records=[json.loads(line) for line in (inputs/'inputs.jsonl').open()]
        if gate:
            keys={r['key'] for r in (json.loads(line) for line in (root/'cell_native_input_gate_1028_012500/inputs.jsonl').open())};records=[r for r in records if r['key'] in keys]
        require(len(entries)==len(records)==source['inputs']==(8 if gate else 20456),'all canonical fullcell inputs')
        zero=[];total=retained=0;all_checks=[];schemas=None;encoded_bytes=0
        for index,(e,r) in enumerate(zip(entries,records)):
            identity=e['identity'];key=r['key'];require(e['index']==r['index'] and (identity['wsi_id'],identity['kind'],identity['key'])==(r['wsi_id'],r['kind'],key),'all source/query index and key identities')
            require(identity['geometry']==dict(native_boxes=r['native_boxes'],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),'all native multiscale footprints/pitch/source identities')
            require(len(identity['cell_records'])==len(identity['RGB_array_sha256'])==3 and [c['scale'] for c in identity['cell_records']]==[0,1,2],'all3 actual cell input views')
            views=[]
            for c,box in zip(identity['cell_records'],r['native_boxes']):
                arrays,check=audit_view(c,box,r['native_mpp']);total+=check['total'];retained+=check['retained'];views.append(check)
                if check['total']==0:zero.append(dict(key=key,scale=c['scale']))
            require(Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256'],'every encoded cache byte/size')
            payload=torch.load(e['path'],map_location='cpu',weights_only=False)
            require(payload['identity']==identity and payload['checkpoint_sha256']==sig['checkpoint_sha256'] and payload['method']=='fewclick_m3d_xcell','every actual saved fullcell input/method/weight')
            schema=finite_schema(payload['encoded']);require(schemas is None or schemas==schema,'all fixed512 three-view encoded shapes/dtypes');schemas=schema
            encoded_bytes+=e['bytes'];all_checks.append(dict(key=key,views=views))
            dump('status.json',dict(status='independent_CRC_fullcell_inputs_cache',worker_pid=pid,completed=index+1,total=len(entries)))
            del payload,arrays
        require((total,retained,zero,encoded_bytes)==(source['sum_true_instances_across_overlapping_views'],source['sum_retained_across_views'],source['zero_cell_cases'],source['encoded_bytes']),'all exact true/retained/zero/storage totals')
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(base/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
        require(sha(best['path'])==best['checkpoint_sha256']==sig['checkpoint_sha256'],'same development-selected fullcell checkpoint')
        device=torch.device('cuda:0');model=load('fewclick_m3d_xcell',Path(best['path']),device,sha(base/'tasks_1025_125400/contract.json'));require(model.prior['cell_mode']=='xcell','actual cell branch active')
        indexed={e['identity']['key']:e for e in entries};native_records={r['key']:r for r in records};support=[r['key'] for r in records if r['kind']=='support'];query=[r['key'] for r in records if r['kind']=='query']
        probes=list(indexed) if gate else list(dict.fromkeys([support[0],support[-1],query[0],query[len(query)//2],query[-1]]+[next(z['key'] for z in zero if z['scale']==scale) for scale in (0,1,2) if any(z['scale']==scale for z in zero)]))
        fresh_encoded={};probe_checks=[]
        with torch.inference_mode():
            for key in probes:
                e=indexed[key];r=native_records[key];raw={};counts=[];slide=pyvips.Image.new_from_file(r['source_wsi'],access='random')
                for scale,(box,c) in enumerate(zip(r['native_boxes'],e['identity']['cell_records'])):
                    x,y,w,h=box;crop=slide.crop(x,y,w,h).extract_band(0,n=3);native=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(h,w,3)
                    require(hashlib.sha256(native.tobytes()).hexdigest()==c['native_RGB_sha256'],'fresh native RGB bytes for actual cells')
                    image=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR)).copy();require(hashlib.sha256(image.tobytes()).hexdigest()==e['identity']['RGB_array_sha256'][scale],'fresh native HE512 identity')
                    arrays,check=audit_view(c,box,r['native_mpp']);suffix='' if scale==0 else '_'+str(scale)
                    raw['image'+suffix]=he_to_tensor(image)[None].to(device);raw['cells'+suffix]=torch.from_numpy(arrays['features'])[None].to(device);raw['cell_valid'+suffix]=torch.ones(1,len(arrays['features']),dtype=torch.bool,device=device);raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([check['total']],dtype=torch.float32,device=device);counts.append(check['total'])
                density=[];hook=model.core.cell.register_forward_hook(lambda m,args,out:density.append(float(out['region_cell_count'].sum())))
                try:fresh=model.core.encode_image(raw)
                finally:hook.remove()
                require(len(density)==3 and all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-3) for a,b in zip(density,counts)),'true uncapped density/zero reaches each region encoder')
                saved=torch.load(e['path'],map_location='cpu',weights_only=False)['encoded'];error=compare_tree(saved,transfer(fresh,'cpu'));require(error<=2e-6,'all fresh native encoded tensors within original native gate tolerance')
                fresh_encoded[key]=fresh;probe_checks.append(dict(key=key,encoded_max_error=error,true_counts=counts,density_sums=density))
            # Gate exhaustively checks all actual6 support/query tasks×4budgets.
            # Full cache validates fresh encodes at fixed probes; output scoring is separate.
            decode_checks=[]
            if gate:
                from benchmarks.v5.fair_comparison.protocol import nested
                plans=read(root/'support_query_full_1027_173800/support_tasks_maximum.json');calls=[0]
                def hook(module,args):calls[0]+=1
                handles=[m.backbone.register_forward_pre_hook(hook) for m in model.core.encoders]
                for key in query:
                    r=native_records[key]
                    for plan in [p for p in plans if p['wsi_id']==r['wsi_id'] and p['prompt_group']==0 and (p['target']==0 or (p['target']==10 and p['wsi_id']==records[0]['wsi_id'])) and p['support_id'] in indexed]:
                        for budget in ('3+2','1+1','3+1','5+1'):
                            plan_budget=nested(plan,budget);p={}
                            for sign in ('positive','negative'):
                                p[sign+'_xy']=torch.tensor([plan_budget[sign+'_xy']],dtype=torch.float32,device=device);p[sign+'_mask']=torch.ones(1,len(plan_budget[sign+'_xy']),dtype=torch.bool,device=device)
                            p['prompt_size_id']=torch.ones(1,dtype=torch.long,device=device)
                            task=model.core.support_task(fresh_encoded[plan['support_id']],p);direct,maps=model.core.decode(fresh_encoded[key],task,p,False)
                            s=transfer(torch.load(indexed[plan['support_id']]['path'],map_location='cpu',weights_only=False)['encoded'],device);q=transfer(torch.load(indexed[key]['path'],map_location='cpu',weights_only=False)['encoded'],device)
                            cached,cm=model.core.decode(q,model.core.support_task(s,p),p,False);error=float((direct['pixel_probability']-cached['pixel_probability']).abs().max())
                            require(not maps.count_nonzero() and not cm.count_nonzero() and error<=2e-6 and torch.equal(direct['pixel_probability']>=.5,cached['pixel_probability']>=.5),'fresh native selected-model and actual disk source/query probabilities/masks')
                            decode_checks.append(dict(query=key,target=plan['target'],budget=budget,max_error=error))
                for h in handles:h.remove()
                require(calls[0]==0 and len(decode_checks)==24,'all24 native support-query budgets and encoder hooks0')
        dump('entries.json',all_checks);dump('native_probe_checks.json',probe_checks);dump('decode_checks.json',decode_checks)
        result=dict(status='accepted_main30_fullcell_cache_gate' if gate else 'accepted_main30_fullcell_cache_qualified',worker_pid=pid,recipe=recipe,source_summary_sha256=sha(a.source/'summary.json'),inputs=len(entries),views=3*len(entries),zero_cell_cases=zero,native_probes=len(probe_checks),decode_checks=len(decode_checks),entries_sha256=sha(a.output/'entries.json'),seconds=time.perf_counter()-start,
                    scope='All actual NPZ/encoded bytes/identity/physical instance geometry/true counts/74 assembly; fresh native selected encoder and density probes, all24 gate budgets; no detector rerun or independent reg64 recomputation',remaining='Full20456 cache/61368view production and WSI prediction scores, probability replay/quality acceptance and BF1/statistics')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps({k:v for k,v in result.items() if k not in ('recipe','zero_cell_cases')}),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
