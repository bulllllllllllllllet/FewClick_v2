"""Independent fullcell packed-mask/original-GT/component/distance recount.

No producer, unpacking helper or score kernel is imported. This accepts the
evaluation of saved predictions; it does not claim exhaustive model replay.
CPU fixtures additionally compare every mask and probability digest with the
previous independently accepted full-native prediction windows.
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


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:
        raise ValueError(why)


def metrics(tp,fp,positive,negative):
    valid=positive+negative
    return dict(tp=tp,fp=fp,fn=positive-tp,positive=positive,negative=negative,valid=valid,
                dice=2*tp/(positive+tp+fp) if positive else None,recall=tp/positive if positive else None,
                precision=tp/(tp+fp) if tp+fp else 0.,false_positive_fraction=fp/valid if valid else None,
                false_positive_rate=fp/negative if negative else None)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();pid=os.getpid()
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        root=a.root;source=read(a.source/'summary.json');contract=read(a.source/'contract.json')
        fixture=source['status']=='passed_interactive_prediction_driver_fixture'
        require(fixture or source['status']=='complete_interactive_main30_predictions_scores_unaccepted','completed prediction/scoring execution')
        sig=source['signature'];method=sig['method']
        require(method=='fewclick_m3d_xcell' or (a.reference_fixture and fixture and method=='fewclick_m3d_image'),'fullcell predictions or explicit original NoCell score-kernel fixture only')
        variants=('graph_dense','region_prototype')
        recipe=dict(source_sha256=sha(__file__),producer_sources=sig['sources'],method=method,protocol_sha256=sig['protocol_sha256'])
        if not fixture:
            require(a.verified_gate is not None,'actual independent complete-mask score fixture required')
            g=read(a.verified_gate/'summary.json');require(g['status']=='accepted_interactive_score_fixture' and g['recipe']==recipe,'same independent score audit recipe')
        for p,h in sig['sources'].items():require(sha(p)==h,'prediction/scoring sources remain frozen')
        for name,key in [('rows.json','rows_sha256'),('outputs.json','outputs_sha256'),('tile_probability_digests.jsonl','tile_digests_sha256')]:
            require(sha(a.source/name)==source[key],'complete prediction and score manifests bound')
        rows=read(a.source/'rows.json');outputs=read(a.source/'outputs.json')
        expected_count=(16 if fixture else 5520)*len(variants)
        require(len(rows)==len(outputs)==source['rows']==expected_count and source['online_query_encode_calls']==0,'all task outputs and no query image encoding')
        require({(r['prediction'],r['prediction_sha256']) for r in rows}=={(r['prediction'],r['sha256']) for r in outputs},'every prediction output exactly bound to scoring rows')
        protocol=root/'support_query_full_1027_173800';require(sha(protocol/'summary.json')==sig['protocol_sha256'],'fixed source/query protocol identity')
        require(read(root/'gt_components_full_1027_231500/contract.json')['protocol_summary_sha256']==sig['protocol_sha256'],'same source-point/component protocol')
        ca=read(root/'gt_components_acceptance_1027_231500/summary.json');require(ca['status']=='accepted_all30_GT_components_and_prompt_domains' and ca['source_summary_sha256']==sha(root/'gt_components_full_1027_231500/summary.json'),'whole original GT component acceptance')
        if method=='fewclick_m3d_xcell':
            require(not contract['cpu_fixture'] and not a.reference_fixture,'actual fullcell GPU predictions required')
            cp=root/'fullcell_CRC_cache_accept_full_1028_031000';cache=root/'fullcell_CRC_cache_full_1028_031000'
            cache_source=read(cache/'summary.json');receipt=read(cp/'summary.json')
            require(receipt['status']=='accepted_main30_fullcell_cache_qualified' and receipt['source_summary_sha256']==sha(cache/'summary.json') and sha(cp/'summary.json')==sig['cache_acceptance_sha256'],'independent actual fullcell cache receipt')
            require(cache_source['inputs']==20456 and cache_source['views']==61368 and cache_source['signature']['checkpoint_sha256']==sig['weight_sha256'] and cache_source['signature']['device']==sig['device'],'fullcell cache/model/device binding')
            require(read(root/'cell_native_inputs_full_1028_012700/summary.json')['signature']['protocol_sha256']==sig['protocol_sha256'],'same actual fullcell input/source/query protocol')
        else:
            short={'fewclick_m3d_image':'nocell','sam_ft':'sam','wsi_sam_ft':'wsisam'}[method]
            cp=root/(f'interactive_cpu_accept_{short}_1028_001800' if contract['cpu_fixture'] else f'interactive_GPU_accept_full_{method}_1028_002000')
            require(sha(cp/'summary.json')==sig['cache_acceptance_sha256'],'actual native feature cache receipt identity')
            cpu_names={'fewclick_m3d_image':'interactive_cpu_gate_nocell_1028_000700','sam_ft':'interactive_cpu_gate_sam_ft_1028_000900','wsi_sam_ft':'interactive_cpu_gate_wsi_sam_ft_1028_000900'}
            cache=root/(cpu_names[method] if contract['cpu_fixture'] else f'interactive_GPU_full_{method}_1028_001100')
            cache_source=read(cache/'summary.json')
            require(read(cp/'summary.json')['source_summary_sha256']==sha(cache/'summary.json') and cache_source['signature']['protocol_sha256']==sig['protocol_sha256'] and cache_source['signature']['weight_sha256']==sig['weight_sha256'] and cache_source['signature']['device']==sig['device'],'prediction uses the independently accepted method/protocol/weight/device cache')
        plans={(p['wsi_id'],p['target'],p['prompt_group']):p for p in read(protocol/'support_tasks_maximum.json')}
        data={r['wsi_id']:r for r in read(root/'main30_full_1027_171918/slides_private.json')}
        domains={(r['wsi_id'],r['target'],r['prompt_group'],r['budget']):r for r in read(root/'gt_components_full_1027_231500/task_domains.json')}
        grid={q['wsi_id']:[json.loads(line) for line in Path(q['grid']).open()] for q in read(protocol/'query_inputs.json')}
        windows={w['window']:set(w['tiles']) for w in read(root/'semantic_raster_gate_1027_220000/summary.json')['results']}
        expected=set()
        for sid,target,group in plans:
            if fixture and (sid!=next(iter(data)) or target not in (0,10) or group!=0):continue
            for name in windows if fixture else (sid,):
                for budget in ('3+2','1+1','3+1','5+1'):
                    for variant in variants:expected.add((name,sid,target,group,budget,variant))
        actual={(r['item'],r['wsi_id'],r['target'],r['prompt_group'],r['budget'],r['variant']) for r in rows}
        require(len(actual)==len(rows) and actual==expected,'entire predeclared task/variant identity set without omission')
        reference=read(root/'sampled_native_gate_1028_002700/results.json') if contract['cpu_fixture'] else []
        reference_512=read(root/'sampled_native_gate_1028_002700/outputs.json') if contract['cpu_fixture'] else []
        digests=[json.loads(line) for line in (a.source/'tile_probability_digests.jsonl').open()]
        digest_index={}
        for d in digests:
            key=(d['item'],d['target'],d['prompt_group'],d['budget'],d['variant'])
            digest_index.setdefault(key,[]).append(d)
        evidence=[];audited=set();digest_total=0
        for row_number,row in enumerate(rows):
            sid=row['wsi_id'];r=data[sid];target=row['target'];budget=row['budget'];plan=plans[(sid,target,row['prompt_group'])]
            require(row['method']==method and row['positive_native_xy']==plan['positive_native_xy'][:int(budget[0])] and row['negative_native_xy']==plan['negative_native_xy'][:int(budget[2])],'method and actual support positive/negative prefixes')
            require(sha(row['prediction'])==row['prediction_sha256'],'actual prediction geometry/digest bytes')
            pred=read(row['prediction']);require(sha(pred['mask_path'])==pred['mask_sha256'],'all actual packed mask bytes')
            bits=np.load(pred['mask_path'],mmap_mode='r');height,width=pred['GT_shape'];x0,y0,x1,y1=pred['GT_cell_bounds']
            require(bits.dtype==np.uint8 and list(bits.shape)==[height,(width+7)//8] and [height,width]==[y1-y0,x1-x0] and pred['bitorder']=='big','actual packed geometry/schema')
            gtpath=root/'main30_full_1027_171918'/sid/'labels.npy';ccpath=root/'gt_components_full_1027_231500'/sid/'components.npy'
            if sid not in audited:
                require(sha(gtpath)==r['files'][str(gtpath)]['sha256'],'original GT bytes')
                ci=next(z for z in read(root/'gt_components_full_1027_231500/items.json') if z['wsi_id']==sid)
                require(sha(ccpath)==ci['files']['components.npy']['sha256'],'all whole-GT component raster bytes');audited.add(sid)
            labels=np.load(gtpath,mmap_mode='r');cc=np.load(ccpath,mmap_mode='r');gh,gw=labels.shape;w,h=r['native_wh'];sx,sy=w/gw,h/gh
            positive_ids=[]
            for px,py in row['positive_native_xy']:
                x,y=math.floor(px/sx),math.floor(py/sy);require(labels[y,x]==target and cc[y,x]>0,'actual positive source-point component');positive_ids.append(int(cc[y,x]))
            domain=domains[(sid,target,row['prompt_group'],budget)]
            require(domain['positive_component_ids']==positive_ids,'all source-positive component prefix IDs')
            boxes=[t['native_boxes'][0] for t in grid[sid] if not fixture or t['number'] in windows[row['item']]]
            xs=sorted({b[0] for b in boxes});ys=sorted({b[1] for b in boxes});fw,fh=boxes[0][2:]
            require(len(boxes)==len(xs)*len(ys) and all(b[2:]==[fw,fh] for b in boxes),'complete Cartesian prediction grid')
            left,top,right,bottom=min(xs),min(ys),max(xs)+fw,max(ys)+fh
            require([x0,y0,x1,y1]==[math.ceil(left/sx),math.ceil(top/sy),math.floor(right/sx),math.floor(bottom/sy)],'independent full native-cell footprint')
            native_x=(np.arange(x0,x1)+.5)*sx;native_y=(np.arange(y0,y1)+.5)*sy
            nx=np.floor(native_x).astype(np.int64);ny=np.floor(native_y).astype(np.int64)
            count_x=sum(((nx>=x)&(nx<x+fw)).astype(np.uint16) for x in xs)
            prior=None
            if contract['cpu_fixture']:
                prior=next(z for z in reference if z['method']==method and (z['window'],z['target'],z['budget'],z['variant'])==(row['item'],target,budget,row['variant']))
                previous_prob=np.load(prior['files']['sampled']['probability.npy']['path'],mmap_mode='r');previous_mask=np.load(prior['files']['sampled']['mask.npy']['path'],mmap_mode='r')
            end=0
            for block in pred['blocks']:
                start=block['start_GT_y']-y0;stop=start+block['rows'];require(start==end and start<stop<=height,'complete nonoverlapping prediction blocks');end=stop
                packed=np.asarray(bits[start:stop]);mask=np.unpackbits(packed,axis=1,count=width,bitorder='big').astype(bool)
                count_y=sum(((ny[start:stop]>=y)&(ny[start:stop]<y+fh)).astype(np.uint16) for y in ys);counts=count_y[:,None]*count_x[None,:]
                require(hashlib.sha256(packed.tobytes()).hexdigest()==block['packed_uint8_sha256'] and hashlib.sha256(mask.tobytes()).hexdigest()==block['binary_bool_sha256'],'every saved packed and decoded Boolean block digest')
                require(np.all(counts>0) and hashlib.sha256(counts.tobytes()).hexdigest()==block['overlap_uint16_sha256'],'every sampled native overlap count independently recounted')
                if width%8:require(np.all((packed[:,-1]&((1<<(8-width%8))-1))==0),'zero padding excluded from evaluation')
                if prior:
                    require(np.array_equal(mask,previous_mask[start:stop]) and hashlib.sha256(np.ascontiguousarray(previous_prob[start:stop]).tobytes()).hexdigest()==block['probability_float32_sha256'],'every CPU native probability digest and mask unchanged')
            require(end==height,'all saved prediction rows present')
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
            if not fixture:require((standard['positive'],standard['positive']-removed,standard['negative'],len(eligible_ids))==(domain['remote_target_pixels'],domain['unprompted_target_pixels'],domain['negative_GT_pixels'],domain['unprompted_components']),'all accepted full30 task-domain denominators')
            selected=digest_index.get((row['item'],target,row['prompt_group'],budget,row['variant']),[])
            tile_ids={t['number'] for t in grid[sid] if not fixture or t['number'] in windows[row['item']]}
            require(len(selected)==len(tile_ids) and {d['tile'] for d in selected}==tile_ids,'every fixed query contribution and512 digest identity');digest_total+=len(selected)
            if contract['cpu_fixture']:
                for d in selected:
                    ref=next(z for z in reference_512 if z['method']==method and (z['window'],z['target'],z['budget'],z['variant'],z['tile'])==(row['item'],target,budget,row['variant'],d['tile']))
                    require(hashlib.sha256(np.load(ref['path']).tobytes()).hexdigest()==d['probability512_float32_sha256'],'every CPU decoder512 probability digest')
            evidence.append(dict(item=row['item'],target=target,budget=budget,variant=row['variant'],valid_pixels=standard['positive']+standard['negative']))
            dump('status.json',dict(status='independent_interactive_GT_score_recount',worker_pid=pid,method=method,completed_rows=row_number+1,total=len(rows)))
            del labels,cc,bits
        require(digest_total==len(digests),'no extra unassigned decoder probability digests')
        result=dict(status='accepted_interactive_score_fixture' if fixture else 'accepted_interactive_main30_GT_scores_qualified',worker_pid=pid,recipe=recipe,rows=len(rows),source_summary_sha256=sha(a.source/'summary.json'),seconds=time.perf_counter()-started,evidence=evidence,
                    scope='Every fullcell saved packed prediction/mask/count digest and original GT/component/distance statistic; explicit NoCell reference fixture additionally replays accepted probabilities. Fullcell model probabilities not exhaustively redecoded',remaining='Full GPU/30 execution, independent decoder probability replay qualification, BF1/statistics and full-cell main30')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);p.add_argument('--reference-fixture',action='store_true')
    run(p.parse_args())
