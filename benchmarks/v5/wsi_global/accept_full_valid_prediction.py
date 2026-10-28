"""Independent full-valid packed-mask/GT recount and original-remote invariance.

Gates re-decode every output and independently project native PIL probabilities.
Formal full runs accept every saved mask/score, not exhaustive new-field model
replay; original remote512 digests and all original remote mask cells must match.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype, region_prototype
from .native_remote_gate import prompts


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def move(x,device):
    if isinstance(x,torch.Tensor):return x.to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(move(v,device) for v in x)
    raise TypeError(type(x))


def metrics(tp,fp,positive,negative):
    valid=positive+negative
    return dict(tp=tp,fp=fp,fn=positive-tp,positive=positive,negative=negative,valid=valid,
        dice=2*tp/(positive+tp+fp) if positive else None,recall=tp/positive if positive else None,
        precision=tp/(tp+fp) if tp+fp else 0.,false_positive_fraction=fp/valid if valid else None,false_positive_rate=fp/negative if negative else None)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid();torch.set_num_threads(4)
    def dump(n,d):(a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        root=a.root;source=read(a.source/'summary.json');contract=read(a.source/'contract.json')
        fixture=source['status']=='passed_full_valid_prediction_driver_gate'
        require(fixture or source['status']=='complete_full_valid_predictions_scores_unaccepted','completed full-valid execution')
        sig=source['signature'];method=sig['method'];variants=('graph_dense','region_prototype') if method=='fewclick_m3d_image' else ('feature_prototype',)
        require(sig['torch_cpu_threads']==torch.get_num_threads()==4,'same original CPU4 numerical contract')
        require(sig['evaluation_scope']=='complete_valid4x_including_support_separate_diagnostic','separate from strict remote main table')
        recipe=dict(source_sha256=sha(__file__),producer_signature=sig)
        if not fixture:
            require(a.verified_gate is not None,'independent actual same-device full-valid gate required')
            ga=read(a.verified_gate/'summary.json');require(ga['status']=='accepted_full_valid_prediction_gate' and ga['recipe']==recipe,'same full acceptance recipe')
        for path,digest in sig['sources'].items():require(sha(path)==digest,'unchanged producer/model/scoring helpers')
        for name,key in [('rows.json','rows_sha256'),('outputs.json','outputs_sha256'),('tile_probability_digests.jsonl','tile_digests_sha256')]:require(sha(a.source/name)==source[key],'all output manifests bound')
        rows=read(a.source/'rows.json');outputs=read(a.source/'outputs.json');expected_count=(24 if fixture else 5520)*len(variants)
        require(len(rows)==len(outputs)==source['rows']==expected_count and source['online_query_encode_calls']==0,'all fixed tasks and no image encoder calls')
        require({(r['prediction'],r['prediction_sha256']) for r in rows}=={(o['prediction'],o['sha256']) for o in outputs},'every prediction output matches scoring row')
        protocol=root/'full_valid_protocol_1028_154400';require(sha(protocol/'summary.json')==sig['protocol_summary_sha256'],'same independently accepted full-valid protocol')
        receipt=read(root/'full_valid_protocol_acceptance_1028_154500/summary.json');require(receipt['status']=='accepted_full_valid_metadata_protocol_not_predictions' and receipt['source_summary_sha256']==sha(protocol/'summary.json'),'full-valid all-task acceptance')
        for name,digest in read(protocol/'summary.json')['files'].items():require(sha(protocol/name)==digest,'all fixed full-valid domain/task bytes')
        ci=read(root/'gt_components_acceptance_1027_231500/summary.json');require(ci['status']=='accepted_all30_GT_components_and_prompt_domains' and ci['source_summary_sha256']==sha(root/'gt_components_full_1027_231500/summary.json'),'original whole-GT components')
        index=Path(contract['index']);ix=read(index/'summary.json');require(sha(index/'summary.json')==sig['index_summary_sha256'] and sha(index/'cache_manifest.jsonl')==ix['cache_manifest_sha256'],'actual accepted complete cache index')
        require(ix['status']==('passed_interactive_full_valid_index_fixture' if contract['cpu_fixture'] else 'accepted_interactive_full_valid_cache_index_qualified'),'actual fixture/formal full-valid index')
        for path,key in [(Path(ix['base_cache']),'base_summary_sha256'),(Path(ix['base_acceptance']),'base_acceptance_sha256'),(Path(ix['extension']),'extension_summary_sha256'),(Path(ix['extension_acceptance']),'extension_acceptance_sha256')]:require(sha(path/'summary.json')==ix[key],'all underlying cache/acceptance receipts')
        entries={e['full_valid_key']:e for e in map(json.loads,(index/'cache_manifest.jsonl').open())}
        require((ix['recipe']['method'],ix['device'],ix['weight_sha256'])==(method,sig['device'],sig['weight_sha256']),'same method, device and selected weights')
        base=Path(ix['base_cache']);base_entries={e['identity']['key']:e for e in map(json.loads,(base/'cache_manifest.jsonl').open())}
        require(sha(base/'cache_manifest.jsonl')==read(base/'summary.json')['cache_manifest_sha256'],'original supports remain immutable')
        plans={(p['wsi_id'],p['target'],p['prompt_group']):p for p in read(root/'support_query_full_1027_173800/support_tasks_maximum.json')}
        tasks={(t['wsi_id'],t['target'],t['prompt_group'],t['budget']):t for t in map(json.loads,(protocol/'tasks.jsonl').open())}
        data={r['wsi_id']:r for r in read(root/'main30_full_1027_171918/slides_private.json')}
        component_items={r['wsi_id']:r for r in read(root/'gt_components_full_1027_231500/items.json')}
        slide_domains={r['wsi_id']:r for r in read(protocol/'slides.json')}
        domains={(d['wsi_id'],d['target'],d['prompt_group'],d['budget']):d for d in read(root/'gt_components_full_1027_231500/task_domains.json')}
        grid={sid:list(map(json.loads,Path(r['grid']).open())) for sid,r in data.items()}
        sid0=next(iter(data));first=[e for e in entries.values() if e['identity']['wsi_id']==sid0]
        if fixture:
            exact=[('new_left',next(e for e in first if e['cache_origin']=='extension')),('source_alias',next(e for e in first if e['identity']['kind']=='support')),('remote_single',next(e for e in first if e['identity']['kind']=='query'))]
            expected_windows=[dict(item=n,wsi_id=sid0,tiles=[e['full_valid_geometry']['tile']]) for n,e in exact]
        else:expected_windows=[dict(item=sid,wsi_id=sid,tiles=[g['number'] for g in grid[sid]]) for sid in data]
        require(contract['windows']==expected_windows,'independently derive all gate cases or every full-grid tile')
        windows={w['item']:set(w['tiles']) for w in expected_windows}
        expected=set()
        for sid,target,group in plans:
            if fixture and (sid!=sid0 or target not in (0,10) or group!=0):continue
            for name in windows if fixture else (sid,):
                for budget in ('3+2','1+1','3+1','5+1'):
                    for variant in variants:expected.add((name,sid,target,group,budget,variant))
        require(len(rows)==len({(r['item'],r['wsi_id'],r['target'],r['prompt_group'],r['budget'],r['variant']) for r in rows}) and {(r['item'],r['wsi_id'],r['target'],r['prompt_group'],r['budget'],r['variant']) for r in rows}==expected,'no task or variant omission')
        digest_index={}
        for d in map(json.loads,(a.source/'tile_probability_digests.jsonl').open()):digest_index.setdefault((d['item'],d['target'],d['prompt_group'],d['budget'],d['variant']),[]).append(d)
        primary_rows={};primary_digests={};reference_512=[]
        if contract['cpu_fixture']:
            require(fixture and not torch.cuda.is_available(),'CPU fixture only; no disguised CUDA result')
            reference_512=read(root/'sampled_native_gate_1028_002700/outputs.json')
        else:
            primary=Path(contract['primary']);pa=Path(contract['primary_acceptance']);pr=read(primary/'summary.json');rc=read(pa/'summary.json')
            require(sha(primary/'summary.json')==sig['primary_summary_sha256'] and sha(pa/'summary.json')==sig['primary_acceptance_sha256'] and rc['status']=='accepted_interactive_main30_GT_scores_qualified' and rc['source_summary_sha256']==sha(primary/'summary.json'),'actual accepted original remote outputs')
            for name,key in [('rows.json','rows_sha256'),('tile_probability_digests.jsonl','tile_digests_sha256')]:require(sha(primary/name)==pr[key],'original remote rows/probability digest manifests')
            primary_rows={(r['wsi_id'],r['target'],r['prompt_group'],r['budget'],r['variant']):r for r in read(primary/'rows.json')}
            for d in map(json.loads,(primary/'tile_probability_digests.jsonl').open()):
                key=(d['item'],d['target'],d['prompt_group'],d['budget'],d['variant'],d['tile']);require(key not in primary_digests,'unique original remote probability contribution');primary_digests[key]=d['probability512_float32_sha256']
        model=None;encodes=[0];handles=[];encoded={};fresh=0;remote_digest_checks=0;remote_mask_cells=0
        if fixture:
            device=torch.device(sig['device'])
            if device.type=='cuda':
                require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'),'actual permitted GPU for independent gate replay')
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            adapt=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');best=read(adapt/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
            require(sha(best['path'])==best['checkpoint_sha256']==sig['weight_sha256'],'fresh same selected model bytes')
            model=load(method,Path(best['path']),device,sha(adapt/'tasks_1025_125400/contract.json'));model.eval()
            def hook(module,args):encodes[0]+=1
            handles=[e.register_forward_pre_hook(hook) for e in ([z.backbone for z in model.core.encoders] if method=='fewclick_m3d_image' else [model.sam.image_encoder])]
            def cached(e):
                if e['path'] not in encoded:
                    require(sha(e['path'])==e['sha256'],'each actual accepted replay cache byte');payload=torch.load(e['path'],map_location='cpu',weights_only=False)
                    require(payload['identity']==e['identity'] and payload['checkpoint_sha256']==sig['weight_sha256'],'actual same native cache identity');encoded[e['path']]=move(payload['encoded'],device)
                return encoded[e['path']]
        evidence=[];audited=set();digest_total=0
        for row_number,row in enumerate(rows):
            sid=row['wsi_id'];r=data[sid];target=row['target'];budget=row['budget'];plan=plans[sid,target,row['prompt_group']];frozen=tasks[sid,target,row['prompt_group'],budget]
            require(row['method']==method and row['evaluation_scope']==sig['evaluation_scope'] and row['positive_native_xy']==frozen['positive_native_xy'] and row['negative_native_xy']==frozen['negative_native_xy'],'unchanged source point prefixes and scope')
            require(sha(row['prediction'])==row['prediction_sha256'],'prediction metadata bytes');pred=read(row['prediction']);require(sha(pred['mask_path'])==pred['mask_sha256'],'every actual packed mask byte')
            bits=np.load(pred['mask_path'],mmap_mode='r');height,width=pred['GT_shape'];x0,y0,x1,y1=pred['GT_cell_bounds']
            require(bits.dtype==np.uint8 and list(bits.shape)==[height,(width+7)//8] and [height,width]==[y1-y0,x1-x0] and pred['bitorder']=='big','packed original GT geometry')
            gtpath=root/'main30_full_1027_171918'/sid/'labels.npy';ccpath=root/'gt_components_full_1027_231500'/sid/'components.npy'
            if sid not in audited:
                require(sha(gtpath)==r['files'][str(gtpath)]['sha256'] and sha(ccpath)==component_items[sid]['files']['components.npy']['sha256'],'every original full-GT and connected component raster hash');audited.add(sid)
            labels=np.load(gtpath,mmap_mode='r');cc=np.load(ccpath,mmap_mode='r');gh,gw=labels.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh
            positive_ids=[]
            for px,py in row['positive_native_xy']:
                x,y=math.floor(px/sx),math.floor(py/sy);require(labels[y,x]==target and cc[y,x]>0,'actual source-positive component');positive_ids.append(int(cc[y,x]))
            require(positive_ids==frozen['scoring_only_positive_component_ids'],'same full-GT positive IDs for scoring only')
            selected_grid=[t for t in grid[sid] if t['number'] in windows[row['item']]];boxes=[t['native_boxes'][0] for t in selected_grid]
            xs=sorted({b[0] for b in boxes});ys=sorted({b[1] for b in boxes});fw,fh=boxes[0][2:]
            require(len(boxes)==len(xs)*len(ys) and all(b[2:]==[fw,fh] for b in boxes),'complete Cartesian prediction footprint')
            bounds=[xs[0],ys[0],xs[-1]+fw,ys[-1]+fh]
            require([x0,y0,x1,y1]==[math.ceil(bounds[0]/sx),math.ceil(bounds[1]/sy),math.floor(bounds[2]/sx),math.floor(bounds[3]/sy)],'independent whole-GT-cell score domain')
            if not fixture:require([x0,y0,x1,y1]==slide_domains[sid]['full_GT_bounds'],'full valid domain without cropping for metrics')
            native_x=(np.arange(x0,x1)+.5)*sx;native_y=(np.arange(y0,y1)+.5)*sy;nx=np.floor(native_x).astype(np.int64);ny=np.floor(native_y).astype(np.int64)
            count_x=sum(((nx>=x)&(nx<x+fw)).astype(np.uint16) for x in xs)
            selected=digest_index.get((row['item'],target,row['prompt_group'],budget,row['variant']),[])
            require(len(selected)==len(selected_grid) and {d['tile'] for d in selected}=={t['number'] for t in selected_grid},'every query512 contribution exactly once');digest_total+=len(selected)
            gate_native=None
            for d in selected:
                key=f'{sid}:full_valid:{d["tile"]}';require(d['full_valid_key']==key and key in entries,'actual indexed full-valid field')
                if fixture:
                    require(sha(d['gate_probability_path'])==d['gate_probability_file_sha256'],'actual gate probability file');value=np.load(d['gate_probability_path'],allow_pickle=False)
                    require(value.shape==(512,512) and value.dtype==np.float32 and hashlib.sha256(value.tobytes()).hexdigest()==d['probability512_float32_sha256'],'actual FP32 gate512 probability')
                    with torch.inference_mode():
                        source_cache=cached(base_entries[plan['support_id']]);query_cache=cached(entries[key]);p=move(prompts(plan,budget),device)
                        if method=='fewclick_m3d_image':
                            task=model.core.support_task(source_cache,p);out,maps=model.core.decode(query_cache,task,p,False);require(not maps.count_nonzero(),'no query-local prompt maps')
                            actual=out['pixel_probability'][0] if row['variant']=='graph_dense' else region_prototype(source_cache,query_cache,p)['pixel_probability'][0]
                        else:actual=dense_prototype(source_cache,query_cache,p)['pixel_probability'][0]
                        require(np.array_equal(value,actual.detach().float().cpu().numpy()),'fresh every actual gate decoder probability');fresh+=1
                    native=np.asarray(Image.fromarray(value).resize((fw,fh),Image.Resampling.BILINEAR));gate_native=native[np.ix_(ny-ys[0],nx-xs[0])]
                    if contract['cpu_fixture'] and row['item']=='remote_single':
                        refs=[z for z in reference_512 if (z['method'],z['target'],z['budget'],z['variant'],z['tile'])==(method,target,budget,row['variant'],d['tile'])]
                        require(bool(refs),'previous accepted original remote512 reference present')
                        for ref in refs:require(sha(ref['path'])==ref['sha256'] and np.array_equal(value,np.load(ref['path'])),'original remote512 unchanged');remote_digest_checks+=1
                if not contract['cpu_fixture']:
                    pk=(sid,target,row['prompt_group'],budget,row['variant'],d['tile'])
                    if entries[key]['identity']['kind']=='query':
                        require(primary_digests.get(pk)==d['probability512_float32_sha256'],'every original remote512 probability unchanged');remote_digest_checks+=1
            end=0
            for block in pred['blocks']:
                start=block['start_GT_y']-y0;stop=start+block['rows'];require(start==end and start<stop<=height,'all packed prediction rows exactly once');end=stop
                packed=np.asarray(bits[start:stop]);mask=np.unpackbits(packed,axis=1,count=width,bitorder='big').astype(bool)
                count_y=sum(((ny[start:stop]>=y)&(ny[start:stop]<y+fh)).astype(np.uint16) for y in ys);counts=count_y[:,None]*count_x[None,:]
                require(hashlib.sha256(packed.tobytes()).hexdigest()==block['packed_uint8_sha256'] and hashlib.sha256(mask.tobytes()).hexdigest()==block['binary_bool_sha256'],'all original saved mask digests')
                require(np.all(counts>0) and hashlib.sha256(counts.tobytes()).hexdigest()==block['overlap_uint16_sha256'],'independent native overlap counts')
                if width%8:require(np.all((packed[:,-1]&((1<<(8-width%8))-1))==0),'padding bits excluded')
                if fixture:require(hashlib.sha256(np.ascontiguousarray(gate_native[start:stop]).tobytes()).hexdigest()==block['probability_float32_sha256'] and np.array_equal(mask,gate_native[start:stop]>=.5),'every independently projected native gate probability/mask')
            require(end==height,'all native prediction blocks complete')
            standard=Counter();distance=[Counter() for _ in range(3)];areas=Counter();hits=Counter()
            for offset in range(0,height,173):
                stop=min(offset+173,height);truth=np.asarray(labels[y0+offset:y0+stop,x0:x1]);ids=np.asarray(cc[y0+offset:y0+stop,x0:x1]);guess=np.unpackbits(np.asarray(bits[offset:stop]),axis=1,count=width,bitorder='big').astype(bool)
                valid=truth!=255;positive=truth==target
                standard.update(tp=int(np.count_nonzero(valid&positive&guess)),fp=int(np.count_nonzero(valid&~positive&guess)),positive=int(np.count_nonzero(valid&positive)),negative=int(np.count_nonzero(valid&~positive)))
                keys,n=np.unique(ids[positive],return_counts=True);areas.update(dict(zip(map(int,keys),map(int,n))))
                keys,n=np.unique(ids[positive&guess],return_counts=True);hits.update(dict(zip(map(int,keys),map(int,n))))
                eligible=valid.copy()
                for gid in set(positive_ids):eligible &= ids!=gid
                nearest=np.full(truth.shape,np.inf)
                for px,py in row['positive_native_xy']:
                    nearest=np.minimum(nearest,np.hypot((native_x[None,:]-px)*r['native_mpp'][0],(native_y[offset:stop,None]-py)*r['native_mpp'][1]))
                for counter,condition in zip(distance,(nearest<1000,(nearest>=1000)&(nearest<5000),nearest>=5000)):
                    take=eligible&condition;counter.update(tp=int(np.count_nonzero(take&positive&guess)),fp=int(np.count_nonzero(take&~positive&guess)),positive=int(np.count_nonzero(take&positive)),negative=int(np.count_nonzero(take&~positive)))
            scores=row['scores'];require(scores['standard']==metrics(**{k:standard[k] for k in ('tp','fp','positive','negative')}),'all standard Dice/recall/precision/false positive values')
            removed=sum(areas[k] for k in set(positive_ids));removed_hits=sum(hits[k] for k in set(positive_ids))
            require(scores['unprompted']==metrics(standard['tp']-removed_hits,standard['fp'],standard['positive']-removed,standard['negative']) and scores['censored_target_pixels']==removed,'all post-censor metrics preserve non-target false positives')
            eligible_ids=sorted(k for k,n in areas.items() if n>0 and k not in positive_ids);c=scores['components'];require(sha(c['path'])==c['sha256'],'saved component hit counter bytes')
            with np.load(c['path'],allow_pickle=False) as z:require(z['ids'].tolist()==eligible_ids and z['areas'].tolist()==[areas[k] for k in eligible_ids] and z['hits'].tolist()==[hits[k] for k in eligible_ids],'every full-image component ID/area/prediction hit')
            ratios=np.array([hits[k]/areas[k] for k in eligible_ids]);require(c['count']==len(eligible_ids) and c['mean_area_recall']==(float(ratios.mean()) if len(ratios) else None) and c['detected_at_50']==int(np.count_nonzero(ratios>=.5)) and c['detected_at_10']==int(np.count_nonzero(ratios>=.1)),'every fixed component summary')
            for label,count,actual in zip(('<1mm','1-5mm','>=5mm'),distance,scores['distance']):require(actual==dict(bin=label,**metrics(**{k:count[k] for k in ('tp','fp','positive','negative')})),'all independent physical pixel-distance metrics')
            if not fixture:
                old=primary_rows[sid,target,row['prompt_group'],budget,row['variant']];require(sha(old['prediction'])==old['prediction_sha256'],'original remote prediction metadata');previous=read(old['prediction']);require(sha(previous['mask_path'])==previous['mask_sha256'],'original remote mask bytes')
                rx0,ry0,rx1,ry1=previous['GT_cell_bounds'];require([rx0,ry0,rx1,ry1]==slide_domains[sid]['remote_GT_bounds'],'entire original remote GT rectangle')
                old_bits=np.load(previous['mask_path'],mmap_mode='r');ow=rx1-rx0;oh=ry1-ry0
                require(old_bits.dtype==np.uint8 and list(old_bits.shape)==[oh,(ow+7)//8],'old remote packed shape')
                for start in range(0,oh,137):
                    stop=min(start+137,oh);new=np.unpackbits(np.asarray(bits[ry0-y0+start:ry0-y0+stop]),axis=1,count=width,bitorder='big')[:,rx0-x0:rx1-x0];prior=np.unpackbits(np.asarray(old_bits[start:stop]),axis=1,count=ow,bitorder='big')
                    require(np.array_equal(new,prior),'every original remote mask cell identical to full-valid crop');remote_mask_cells+=new.size
            evidence.append(dict(item=row['item'],target=target,budget=budget,variant=row['variant'],valid_pixels=standard['positive']+standard['negative']))
            dump('status.json',dict(status='independent_full_valid_GT_and_remote_invariance',worker_pid=pid,method=method,completed_rows=row_number+1,total=len(rows)))
            del labels,cc,bits
        require(digest_total==sum(map(len,digest_index.values())),'no orphan probability contribution')
        if not fixture:require(remote_digest_checks==len(primary_digests),'every original remote512 contribution compared exactly once')
        for handle in handles:handle.remove()
        require(encodes[0]==0 and (not fixture or fresh==len(rows)),'zero image encoders and every gate probability freshly redecoded')
        result=dict(status='accepted_full_valid_prediction_gate' if fixture else 'accepted_full_valid_saved_predictions_GT_and_remote_invariance_qualified',worker_pid=pid,recipe=recipe,source_summary_sha256=sha(a.source/'summary.json'),rows=len(rows),fresh_gate512_outputs=fresh,online_encoder_calls=encodes[0],remote_probability_digest_checks=remote_digest_checks,remote_mask_cells_checked=remote_mask_cells,seconds=time.perf_counter()-started,evidence=evidence,
            scope='Every saved full-domain mask/count/GT/component/distance score; every gate512 output independently redecoded and native-projected. Formal full: every original remote512 digest and mask cell unchanged; new fields are not exhaustively redecoded.',remaining='Actual GPU/full28452 execution, full-valid BF1/statistics and fullcell native input/cache extension. Source-inclusive scores remain a separate diagnostic.')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
