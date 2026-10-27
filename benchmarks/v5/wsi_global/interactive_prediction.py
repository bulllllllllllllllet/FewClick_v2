"""Frozen support/query caches -> remote predictions -> original-GT scores.

Full outputs are lossless packed masks and exact probability/overlap digests.
Each task's mask is finalized and hashed before that task's labels are opened.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .native_remote_gate import prompts
from .packed_prediction import PackedSampledStripe,unpack_rows
from .retrieval_stream import RetrievalAccumulator
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype,region_prototype


METHODS={'fewclick_m3d_image':('nocell','interactive_cpu_gate_nocell_1028_000700'),'sam_ft':('sam','interactive_cpu_gate_sam_ft_1028_000900'),'wsi_sam_ft':('wsisam','interactive_cpu_gate_wsi_sam_ft_1028_000900')}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,reason):
    if not test:
        raise ValueError(reason)


def move(x,device):
    if isinstance(x,torch.Tensor):
        return x.to(device)
    if isinstance(x,dict):
        return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return type(x)(move(v,device) for v in x)
    raise TypeError('unexpected cache node '+str(type(x)))


def score_task(root,r,domain,prediction,plan,budget):
    """Scoring-only call, after mask bytes and probability digests are sealed."""
    labels=np.load(root/'main30_full_1027_171918'/r['wsi_id']/'labels.npy',mmap_mode='r')
    cc=np.load(root/'gt_components_full_1027_231500'/r['wsi_id']/'components.npy',mmap_mode='r')
    x0,y0,x1,y1=prediction['GT_cell_bounds'];sx,sy=r['native_wh'][0]/labels.shape[1],r['native_wh'][1]/labels.shape[0]
    require(prediction['GT_shape']==[y1-y0,x1-x0],'original GT prediction shape')
    xs=(np.arange(x0,x1)+.5)*sx;npos=int(budget[0]);acc=RetrievalAccumulator()
    for y in range(y0,y1,64):
        end=min(y+64,y1);ys=(np.arange(y,end)+.5)*sy
        guess=unpack_rows(prediction['mask_path'],slice(y-y0,end-y0),x1-x0)
        acc.update(np.asarray(labels[y:end,x0:x1]),np.asarray(cc[y:end,x0:x1]),guess,plan['target'],domain['positive_component_ids'],xs,ys,r['native_mpp'],plan['positive_native_xy'][:npos])
    return acc.finish()


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        torch.set_num_threads(4);root=a.root;short,cpu_name=METHODS[a.method]
        require(not a.cpu_fixture or a.gate,'CPU fixture only in representative gate')
        cache=root/(cpu_name if a.cpu_fixture else f'interactive_GPU_full_{a.method}_1028_001100')
        accepted=root/(f'interactive_cpu_accept_{short}_1028_001800' if a.cpu_fixture else f'interactive_GPU_accept_full_{a.method}_1028_002000')
        ca=read(accepted/'summary.json');source=read(cache/'summary.json')
        require(ca['status']==('accepted_interactive_cache_fixture' if a.cpu_fixture else 'accepted_interactive_main30_cache_qualified') and ca['source_summary_sha256']==sha(cache/'summary.json'),'independently accepted actual method cache')
        require(source['signature']['method']==a.method,'same method cache')
        device=torch.device(source['signature']['device'])
        if device.type=='cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'),'one permitted physical GPU')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        else:
            require(not torch.cuda.is_available(),'CPU fixture must hide CUDA')
        component_accept=root/'gt_components_acceptance_1027_231500/summary.json';component_source=root/'gt_components_full_1027_231500/summary.json'
        require(read(component_accept)['status']=='accepted_all30_GT_components_and_prompt_domains' and read(component_accept)['source_summary_sha256']==sha(component_source),'all30 scoring components accepted')
        packed_gate=root/'packed_prediction_gate_1028_004000/summary.json'
        require(read(packed_gate)['status']=='passed_lossless_packed_prediction_gate','actual64-case lossless prediction gate')
        reference=root/'sampled_native_gate_1028_002700';projection_receipt=root/'sampled_native_acceptance_1028_003100/summary.json'
        require(read(projection_receipt)['source_summary_sha256']==sha(reference/'summary.json'),'actual64-case independent native projection acceptance')
        sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/packed_prediction.py'),Path('benchmarks/v5/wsi_global/sampled_stripe.py'),Path('benchmarks/v5/wsi_global/retrieval_stream.py'),Path('benchmarks/v5/wsi_global/retrieval_metrics.py'),Path('benchmarks/v5/transfer_comparison/prototypes.py')]}
        signature=dict(sources=sources,method=a.method,cache_acceptance_sha256=sha(accepted/'summary.json'),device=str(device),protocol_sha256=sha(root/'support_query_full_1027_173800/summary.json'),weight_sha256=source['signature']['weight_sha256'],packed_gate_sha256=sha(packed_gate))
        if not a.gate:
            require(a.verified_gate is not None,'actual same-device interactive prediction driver gate required')
            g=read(a.verified_gate/'summary.json');require(g['status']=='passed_interactive_prediction_driver_fixture' and g['signature']==signature,'same full driver cache/device/geometry gate')
        protocol=read(root/'support_query_full_1027_173800/query_inputs.json');plans=read(root/'support_query_full_1027_173800/support_tasks_maximum.json');data={r['wsi_id']:r for r in read(root/'main30_full_1027_171918/slides_private.json')}
        domains={(r['wsi_id'],r['target'],r['prompt_group'],r['budget']):r for r in read(root/'gt_components_full_1027_231500/task_domains.json')}
        require(sha(cache/'cache_manifest.jsonl')==source['cache_manifest_sha256'],'all cache manifest entries bound')
        entries={e['identity']['key']:e for e in (json.loads(line) for line in (cache/'cache_manifest.jsonl').open())}
        if a.gate:
            q=protocol[0];grid={t['number']:t for t in (json.loads(line) for line in Path(q['grid']).open())}
            work=[(g['window'],q,[grid[n] for n in g['tiles']]) for g in read(root/'semantic_raster_gate_1027_220000/summary.json')['results']]
        else:
            work=[(q['wsi_id'],q,[json.loads(line) for line in Path(q['grid']).open()]) for q in protocol]
        nvariants=2 if a.method=='fewclick_m3d_image' else 1
        expected_rows=(32 if a.method=='fewclick_m3d_image' else 16) if a.gate else 5520*nvariants
        reference_outputs=read(reference/'outputs.json') if a.cpu_fixture else []
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,cpu_fixture=a.cpu_fixture,expected_rows=expected_rows,query_GT_role='Each task mask sealed/hashed before labels opened; never passed to model/threshold/prompt selection',output='lossless packed masks, every native-resized mean/count probability block digest, per-tile512 digest; no probability quantization',numeric='PIL native resize; FP32 uniform overlap mean; >=.5; original complete GT-cell centre sampling',method_scope='NoCell graph+regionproto control or explicit custom SAM/WSISAM feature-prototype support-query transfer; not native local SAM decoder or full-cell FewClick',failure_policy='stop preserve; no retry/resume/fallback'))
        base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(base/f'adaptation_1025_131451/train_{a.method}_1025_131451/best_checkpoint.json')
        require(sha(best['path'])==best['checkpoint_sha256']==signature['weight_sha256'],'fixed development selected weights')
        model=load(a.method,Path(best['path']),device,sha(base/'tasks_1025_125400/contract.json'))
        calls=[0]
        def hook(module,args):
            calls[0]+=1
        encoders=[e.backbone for e in model.core.encoders] if a.method=='fewclick_m3d_image' else [model.sam.image_encoder]
        handles=[m.register_forward_pre_hook(hook) for m in encoders];rows=[];case_outputs=[]
        with torch.inference_mode(),(a.output/'tile_probability_digests.jsonl').open('x') as manifest:
            for item_index,(name,q,tiles) in enumerate(work):
                r=data[q['wsi_id']];sid=r['wsi_id'];chosen=[p for p in plans if p['wsi_id']==sid and (not a.gate or (p['target'] in (0,10) and p['prompt_group']==0))]
                boxes=[t['native_boxes'][0] for t in tiles];bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
                gw,gh=r['gt_wh'];w,h=r['native_wh'];estimate=int((bounds[2]-bounds[0])*gw/w)*int((bounds[3]-bounds[1])*gh/h)*len(chosen)*4*nvariants/8
                require(shutil.disk_usage(a.output).free>1.5*estimate,'actual lossless full-budget mask disk headroom')
                query_cpu={};support_cpu={}
                required=[t['number'] for t in tiles]
                if not a.cpu_fixture:
                    query_bytes=sum(entries[sid+':query:'+str(n)]['bytes'] for n in required)
                    available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
                    require(available>3*query_bytes+8*1024**3,'actual RAM headroom to reuse one WSI encoded caches')
                for n in required:
                    if a.cpu_fixture:
                        path=reference/a.method/f'query{n:06d}.pth'
                    else:
                        e=entries[sid+':query:'+str(n)];path=Path(e['path']);require(sha(path)==e['sha256'],'all accepted query encoded bytes')
                    payload=torch.load(path,map_location='cpu',weights_only=False);require(payload['checkpoint_sha256']==best['checkpoint_sha256'],'actual query cache weight')
                    query_cpu[n]=payload['encoded']
                for plan in chosen:
                    e=entries[plan['support_id']];require(sha(e['path'])==e['sha256'],'all source feature bytes')
                    payload=torch.load(e['path'],map_location='cpu',weights_only=False);support_cpu[plan['support_id']]=payload['encoded']
                for plan in chosen:
                    source_device=move(support_cpu[plan['support_id']],device)
                    for budget in ('3+2','1+1','3+1','5+1'):
                        for path,d in sources.items():require(sha(path)==d,'frozen prediction/scoring sources')
                        p=move(prompts(plan,budget),device);task=model.core.support_task(source_device,p) if a.method=='fewclick_m3d_image' else None
                        variants=('graph_dense','region_prototype') if a.method=='fewclick_m3d_image' else ('feature_prototype',)
                        folder=a.output/name/f'target{plan["target"]}_group{plan["prompt_group"]}_{budget}'
                        writers={v:PackedSampledStripe(folder/v,bounds,boxes[0][2:],r['native_wh'],r['gt_wh']) for v in variants}
                        for t in tiles:
                            query=move(query_cpu[t['number']],device)
                            if a.method=='fewclick_m3d_image':
                                output,maps=model.core.decode(query,task,p,False);require(not maps.count_nonzero(),'no query-local prompt maps in remote transfer')
                                probability={'graph_dense':output['pixel_probability'][0],'region_prototype':region_prototype(source_device,query,p)['pixel_probability'][0]}
                            else:
                                probability={'feature_prototype':dense_prototype(source_device,query,p)['pixel_probability'][0]}
                            for variant,value in probability.items():
                                values=value.detach().float().cpu().numpy()
                                if a.cpu_fixture:
                                    ref=next(z for z in reference_outputs if z['method']==a.method and z['window']==name and z['target']==plan['target'] and z['budget']==budget and z['variant']==variant and z['tile']==t['number'])
                                    require(sha(ref['path'])==ref['sha256'] and np.array_equal(values,np.load(ref['path'])),'new actual cached task/decode equals accepted native512 output')
                                writers[variant].push(values,t['native_boxes'][0])
                                manifest.write(json.dumps(dict(item=name,target=plan['target'],prompt_group=plan['prompt_group'],budget=budget,variant=variant,tile=t['number'],probability512_float32_sha256=hashlib.sha256(values.tobytes()).hexdigest()))+'\n')
                        manifest.flush()
                        for variant,writer in writers.items():
                            prediction=writer.finish();prediction['mask_sha256']=sha(prediction['mask_path']);pred_file=folder/variant/'prediction.json';pred_file.write_text(json.dumps(prediction,indent=2)+'\n')
                            # This task's prediction and all probability digests are immutable now.
                            domain=domains[(sid,plan['target'],plan['prompt_group'],budget)];scores=score_task(root,r,domain,prediction,plan,budget)
                            if not a.gate:
                                require((scores['standard']['positive'],scores['unprompted']['positive'],scores['standard']['negative'],scores['components']['count'])==(domain['remote_target_pixels'],domain['unprompted_target_pixels'],domain['negative_GT_pixels'],domain['unprompted_components']),'full frozen component/valid task denominators')
                            comp=scores['components'];path=folder/variant/'component_hits.npz';np.savez_compressed(path,ids=np.array(comp.pop('ids'),np.uint32),areas=np.array(comp.pop('areas'),np.int64),hits=np.array(comp.pop('hits'),np.int64));comp.update(path=str(path),sha256=sha(path))
                            row=dict(item=name,wsi_id=sid,method=a.method,variant=variant,target=plan['target'],prompt_group=plan['prompt_group'],budget=budget,positive_native_xy=plan['positive_native_xy'][:int(budget[0])],negative_native_xy=plan['negative_native_xy'][:int(budget[2])],prediction=str(pred_file),prediction_sha256=sha(pred_file),scores=scores)
                            rows.append(row);case_outputs.append(dict(prediction=str(pred_file),sha256=sha(pred_file),mask_sha256=prediction['mask_sha256']))
                        dump('rows.json',rows);dump('status.json',dict(status='interactive_native_prediction_and_GT_scoring',worker_pid=pid,method=a.method,completed_rows=len(rows),total_rows=expected_rows,item=name,items_complete=item_index,total_items=len(work),online_encoder_calls=calls[0],seconds=time.perf_counter()-started))
                        del writers
                    del source_device
                print(json.dumps(dict(item=name,rows=len(rows),online_encoder_calls=calls[0])),flush=True)
                del query_cpu,support_cpu;gc.collect()
        for handle in handles:handle.remove()
        require(calls[0]==0 and len(rows)==expected_rows,'all fixed tasks and no online image encoding')
        dump('outputs.json',case_outputs)
        result=dict(status='passed_interactive_prediction_driver_fixture' if a.gate else 'complete_interactive_main30_predictions_scores_unaccepted',worker_pid=pid,signature=signature,rows=len(rows),items=len(work),online_query_encode_calls=calls[0],rows_sha256=sha(a.output/'rows.json'),outputs_sha256=sha(a.output/'outputs.json'),tile_digests_sha256=sha(a.output/'tile_probability_digests.jsonl'),seconds=time.perf_counter()-started,remaining='Independent full prediction/probability/score acceptance, BF1/statistics and full-cell main30; sampled export is not native full-output IO timing')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=tuple(METHODS),required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--cpu-fixture',action='store_true');p.add_argument('--verified-gate',type=Path)
    run(p.parse_args())
