"""Full valid4x native closed12 raster, separate from strict-remote tables."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from .semantic_stripe import SemanticStripe
from .semantic_raster_gate import confusion


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def class_metrics(matrix):
    rows=[];valid=int(matrix.sum())
    for c in range(12):
        tp=int(matrix[c,c]);truth=int(matrix[c].sum());predicted=int(matrix[:,c].sum());fp=predicted-tp;fn=truth-tp
        rows.append(dict(target=c,tp=tp,fp=fp,fn=fn,positive=truth,valid=valid,dice=2*tp/(2*tp+fp+fn) if truth else None,
            iou=tp/(tp+fp+fn) if truth else None,precision=tp/predicted if predicted else 0.,recall=tp/truth if truth else None,false_positive_fraction=fp/valid))
    return rows


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();started=time.perf_counter();torch.set_num_threads(4)
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    core=read(a.semantic_gate/'summary.json');require(core['status']=='passed_native12_probability_stitch_and_GT_scoring_gate','actual probability/scoring core gate')
    for p,d in core['source_sha256'].items():require(sha(p)==d,'same accepted raster/scoring core')
    original=read(a.data/'summary.json');data=read(a.data/'slides_private.json');require(original['status']=='complete_main30_full_GT_and_geometry_audit_qualified' and sha(a.data/'slides_private.json')==original['slides_sha256'],'full30 accepted data identity')
    root=a.root;input_root=root/'full_valid_inputs_full_1028_034700'
    accepted_inputs=read(root/'full_valid_inputs_acceptance_1028_034700/summary.json')
    require(accepted_inputs['status']=='accepted_all28452_full_valid_inputs_and_native24_bridge' and accepted_inputs['source_summary_sha256']==sha(input_root/'summary.json'),'all full-valid original physical fields accepted')
    sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/semantic_stripe.py'),Path('benchmarks/v5/wsi_global/semantic_raster_gate.py')]}
    signature=dict(evaluation_scope='complete_valid4x_including_support',source_sha256=sources,data_summary_sha256=sha(a.data/'summary.json'),full_input_summary_sha256=sha(input_root/'summary.json'),semantic_gate_sha256=sha(a.semantic_gate/'summary.json'),weight_sha256='0aa11b35866262648d2b56340a0d4501165da0516cf21df7bdfebf8da05d06a4')
    cache_contract={'signature':{'weight_sha256':signature['weight_sha256']}}
    if a.gate_fixture:
        snapshot=read(a.gate_inputs/'summary.json')
        require(snapshot['status']=='accepted_full_valid_left_native_cache_fixture' and snapshot['weight_sha256']==signature['weight_sha256'],'real newly prepared left-native fixture')
        require(snapshot['inputs_sha256']==sha(a.gate_inputs/'inputs.json'),'all original gate cache identities')
        entries=read(a.gate_inputs/'inputs.json');r=data[0]
        work=[(g['window'],r,[next(e for e in entries if e['identity']['tile']==n) for n in g['tiles']]) for g in snapshot['windows']]
    else:
        require(a.verified_gate is not None,'actual full-valid raster driver gate required')
        g=read(a.verified_gate/'summary.json');require(g['status']=='passed_closed12_raster_production_fixture' and g['signature']==signature,'same-source full-valid native gate')
        index=read(a.cache/'summary.json');require(index['status']=='accepted_closed12_full_valid_cache_index_qualified' and index['recipe']['weight_sha256']==signature['weight_sha256'] and index['recipe']['input_summary_sha256']==signature['full_input_summary_sha256'],'accepted merged28452 index')
        for receipt in index['receipts']:
            require(sha(Path(receipt['source'])/'summary.json')==receipt['source_summary_sha256'] and sha(Path(receipt['acceptance'])/'summary.json')==receipt['acceptance_summary_sha256'],'each immutable independent cache receipt')
        require(sha(a.cache/'cache_manifest.jsonl')==index['cache_manifest_sha256'],'complete28452 cache index bytes')
        entries=[json.loads(line) for line in (a.cache/'cache_manifest.jsonl').open()];require(len(entries)==28452,'entire valid4x grid')
        work=[]
        for r in data:
            group=[e for e in entries if e['identity']['wsi_id']==r['wsi_id']];grid=[json.loads(line) for line in Path(r['grid']).open()]
            require([e['identity']['tile'] for e in group]==[t['number'] for t in grid],'complete original Cartesian row-major grid')
            work.append((r['wsi_id'],r,group))
    estimate=0
    for _,_,group in work:
        boxes=[e['identity']['native_box'] for e in group];estimate+=(max(b[0]+b[2] for b in boxes)-min(b[0] for b in boxes))*(max(b[1]+b[3] for b in boxes)-min(b[1] for b in boxes))*11
    require(shutil.disk_usage(a.output).free>1.5*estimate,'actual native categorical/count/two-probability disk headroom')
    dump('contract.json',dict(worker_pid=pid,gate_fixture=a.gate_fixture,signature=signature,cache_index_summary_sha256=sha(a.cache/'summary.json') if not a.gate_fixture else None,gate_inputs_summary_sha256=sha(a.gate_inputs/'summary.json') if a.gate_fixture else None,estimated_output_bytes=estimate,prediction_grid='native level0',score_grid='original GT cells fully inside prediction rectangle; nearest native categorical pixel at GT cell centre',GT_access='only after categorical/probability files finalized per work item',numeric='float32 softmax and PIL per-channel resize; float64 uniform averaging; argmax',selected_probability_classes=[0,10],failure_policy='stop and preserve, no retry'))
    results=[];all_rows=[]
    for index,(name,r,group) in enumerate(work):
        for p,d in sources.items():require(sha(p)==d,'frozen raster source')
        boxes=[e['identity']['native_box'] for e in group];bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
        folder=a.output/name;writer=SemanticStripe(folder,bounds,boxes[0][2:]);decode_start=time.perf_counter()
        for number,e in enumerate(group):
            require(sha(e['path'])==e['sha256'],'every actual cache byte');value=torch.load(e['path'],map_location='cpu',weights_only=False)
            require(value['identity']==e['identity'] and value['checkpoint_sha256']==cache_contract['signature']['weight_sha256'] and value['method']=='closed_reference','real12class cache/input/weight')
            p=value['logits'].softmax(1)[0].numpy();writer.push(p,e['identity']['native_box'])
            if number%20==0:dump('status.json',dict(status='native_probability_raster',worker_pid=pid,completed_items=index,total_items=len(work),wsi_id=r['wsi_id'],tile=number+1,tiles=len(group),seconds=time.perf_counter()-started))
        coverage=writer.finish();decode_seconds=time.perf_counter()-decode_start;dump(name+'/coverage.json',coverage)
        files={p.name:dict(path=str(p),sha256=sha(p)) for p in folder.glob('*.npy')}
        # Query labels are not opened until this WSI prediction has been finalized and hashed.
        gtpath=a.data/r['wsi_id']/'labels.npy';require(sha(gtpath)==r['files'][str(gtpath)]['sha256'],'original GT label bytes identity');labels=np.load(gtpath,mmap_mode='r');gw,gh=r['gt_wh'];w,h=r['native_wh']
        pred=np.load(folder/'class_prediction.npy',mmap_mode='r');matrix,domain=confusion(pred,labels,bounds,(w/gw,h/gh));np.save(folder/'confusion.npy',matrix,allow_pickle=False)
        rows=class_metrics(matrix)
        for row in rows:row.update(wsi_id=r['wsi_id'],item=name,method='closed12_supervised_reference',extra_supervision='12class CE',prompt_budget=None)
        all_rows+=rows
        if a.gate_fixture:
            if name!='full_left':
                expected=next(g for g in core['results'] if g['window']==name);require(matrix.tolist()==expected['confusion'],'same original remote fixture confusion')
        result=dict(item=name,wsi_id=r['wsi_id'],coverage=coverage,GT_domain=domain,confusion=matrix.tolist(),prediction_files=files,decode_and_write_seconds=decode_seconds)
        results.append(result);dump(name+'/result.json',result);dump('rows.json',all_rows);dump('items.json',results)
        print(json.dumps(dict(item=name,completed=index+1,total=len(work),GT_pixels=int(matrix.sum()),tiles=len(group))),flush=True)
        del writer,labels,pred
    summary=dict(status='passed_closed12_raster_production_fixture' if a.gate_fixture else 'complete_closed12_WSI_quality_execution_unaccepted',worker_pid=pid,signature=signature,items=len(results),tiles=sum(len(g) for _,_,g in work),rows=len(all_rows),items_sha256=sha(a.output/'items.json'),rows_sha256=sha(a.output/'rows.json'),seconds=time.perf_counter()-started,
        remaining='Independent full-valid native12 probability/GT metrics and crop-to-remote invariance; privileged whole-valid diagnostic, not disjoint remote or whole-tissue coverage')
    dump('summary.json',summary);dump('status.json',dict(status=summary['status'],worker_pid=pid));print(json.dumps(dict(status=summary['status'],seconds=summary['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','cache','data','semantic-gate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gate-inputs',type=Path);p.add_argument('--gate-fixture',action='store_true');p.add_argument('--verified-gate',type=Path);run(p.parse_args())
