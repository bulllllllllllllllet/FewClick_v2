"""Independently replay saved Dense checkpoints and recount paired pilot masks."""
import argparse
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import torch

from .diagnose_local_errors import read, write, sha
from .dense_hardnegative_repair import state_sha
from .metrics import score
from benchmarks.v5.phase_6_mask_decoder.dense_prompt_decoder import DensePromptDecoder


def aggregate(rows):
    return dict(n=len(rows),dice=float(np.mean([r['dice'] for r in rows])),
        precision=float(np.mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0 for r in rows])),
        recall=float(np.mean([r['tp']/(r['tp']+r['fn']) for r in rows])),
        bf1=float(np.mean([r['bf1_um2'] for r in rows if r['bf1_um2'] is not None])),
        FP_per_valid=float(np.mean([r['fp']/r['valid'] for r in rows])))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
    summary=read(a.source/'summary.json');contract=read(a.source/'contract.json')
    assert summary['status'] in ('passed_dense_hardnegative_training_gate','complete_dense_hardnegative_pilot_unaccepted')
    assert sha(Path(__file__).with_name('dense_hardnegative_repair.py'))==summary['signature']['source_sha256']
    plans=read(a.source/'plans.json');order=read(a.source/'training_order.json')
    assert len(order)==summary['steps_per_arm'] and order==[i%len(plans['train']) for i in range(len(order))]
    assert all(p['split']=='train' for p in plans['train']) and all(p['split']=='val' for p in plans['val'])
    assert not {p['query_id'] for p in plans['train']}&{p['query_id'] for p in plans['val']}
    arms=('original','continued_original_loss','hardnegative_repair');outputs={};bindings={}
    for arm in arms:
        rows=read(a.source/arm/'rows.json');assert len(rows)==len(plans['val'])
        recounted=[]
        for index,row in enumerate(rows):
            p=a.source/arm/f'case{index:02d}.npz';bindings[str(p)]=sha(p)
            with np.load(p) as z:
                probability=z['probability'];truth=z['truth'];valid=z['valid']
            c=torch.load(a.source/'val_cache'/f'case{index:02d}.pth',map_location='cpu',weights_only=False)
            assert np.array_equal(truth,c['truth'][0].numpy()) and np.array_equal(valid,c['valid'][0].numpy())
            metric=score(probability>=.5,truth,valid,c['mpp'])
            for key,v in metric.items():
                assert (row[key] is None and v is None) or row[key]==v,(arm,index,key)
            assert row['query_id']==plans['val'][index]['query_id'] and row['target']==plans['val'][index]['target']
            recounted.append(row)
        outputs[arm]=recounted
        if arm!='original':
            history=read(a.source/arm/'history.json');assert len(history)==len(order)
            assert [r['step'] for r in history]==list(range(1,len(order)+1))
            assert [r['train_index'] for r in history]==order
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) and r['gradient_norm']>0 for r in history)
            path=a.source/arm/'checkpoint.pth';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['signature']==summary['signature'] and checkpoint['steps']==len(order)
            assert checkpoint['initial_sha256']==summary['checks'][arm]['initial_sha256']
            assert sha(path)==summary['checks'][arm]['checkpoint_sha256']
            dense=DensePromptDecoder(high_resolution=True);dense.load_state_dict(checkpoint['dense'],strict=True);dense.eval()
            assert state_sha(dense)==summary['checks'][arm]['final_sha256']!=checkpoint['initial_sha256']
            # Independent, newly constructed decoder for both domains.
            indices=[next(i for i,p in enumerate(plans['val']) if p['dataset']==d) for d in ('crc','tiger')]
            for index in indices:
                c=torch.load(a.source/'val_cache'/f'case{index:02d}.pth',map_location='cpu',weights_only=False)
                assert set(c['features'])=={'deep','shallow','semantic','base','task','prompt_maps'}
                with torch.inference_mode():fresh=dense(c['features'])['pixel_probability'][0].numpy()
                with np.load(a.source/arm/f'case{index:02d}.npz') as z:assert np.array_equal(fresh,z['probability'])
    assert summary['checks']['continued_original_loss']['initial_sha256']==summary['checks']['hardnegative_repair']['initial_sha256']
    reports={}
    for arm,rows in outputs.items():
        reports[arm]=dict(overall=aggregate(rows),by_area={},by_domain={})
        for name,predicate in [('small_lt5pct',lambda r:r['GT_fraction']<.05),('large_gt50pct',lambda r:r['GT_fraction']>.5)]:
            selected=[r for r in rows if predicate(r)]
            if selected:reports[arm]['by_area'][name]=aggregate(selected)
        for domain in ('crc','tiger'):reports[arm]['by_domain'][domain]=aggregate([r for r in rows if r['dataset']==domain])
    result=dict(status='accepted_dense_hardnegative_gate' if contract['gate'] else 'accepted_dense_hardnegative_pilot_qualified',worker_pid=os.getpid(),
        source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),source=str(a.source),reports=reports,
        checkpoints_reloaded_independently=4,all_saved_masks_recounted=3*len(plans['val']),bindings=bindings,
        limitations='Posthoc single-seed small development pilot, fixed extra Dense-only training. Not formal original matched40k/WSI quality; no cherry-pick of best checkpoint.')
    write(a.output/'summary.json',result)
    lines=['# Dense困难负区域修复小试','',result['limitations'],'','| 方法 | n | Dice% | Precision% | Recall% | BF1% |','|---|---:|---:|---:|---:|---:|']
    for arm,report in reports.items():
        r=report['overall'];lines.append(f"|{arm}|{r['n']}|{100*r['dice']:.2f}|{100*r['precision']:.2f}|{100*r['recall']:.2f}|{100*r['bf1']:.2f}|")
    lines+=['','面积/数据域分层见summary.json；所有原结果保留。改善与退化均报告，接受仅说明证据核对通过，不代表修复一定有效。']
    (a.output/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
