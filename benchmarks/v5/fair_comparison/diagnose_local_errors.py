"""Read-only error decomposition and explicitly separate CPU diagnostic replays."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


METHODS = ('sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def aggregate(rows):
    pairs = defaultdict(list)
    for r in rows:
        tp, fp, fn = r['tp'], r['fp'], r['fn']
        assert tp+fn == r['positive'] and 0 <= tp+fp <= r['valid']
        dice = 2*tp/(2*tp+fp+fn)
        assert abs(dice-r['dice']) < 1e-12
        pairs[(r['query_id'], r['target'])].append(dict(
            dice=dice, recall=tp/(tp+fn),
            precision=tp/(tp+fp) if tp+fp else 0.,
            fp_per_valid=fp/r['valid'], fn_per_positive=fn/(tp+fn),
            bf1=r['bf1_um2'], empty=float(tp+fp == 0)))
    result = dict(rows=len(rows), pairs=len(pairs))
    for key in ('dice','recall','precision','fp_per_valid','fn_per_positive','bf1','empty'):
        vals = [np.mean([v[key] for v in group if v[key] is not None])
                for group in pairs.values() if any(v[key] is not None for v in group)]
        result[key] = float(np.mean(vals)) if vals else None
    return result


def stats(a):
    a.output.mkdir(parents=True, exist_ok=False)
    runs, outputs, binding = {}, {}, {}
    for method in METHODS:
        path = a.root/f'evaluate_{method}_1025_131451/rows.jsonl'
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == 6040
        runs[method] = rows
        binding[str(path)] = sha(path)
        primary = [r for r in rows if r['click_budget'] == '3+2']
        report = dict(primary=aggregate(primary), by_class={}, by_domain={}, by_area={}, by_budget={})
        for domain, target in sorted({(r['dataset'],r['target']) for r in primary}):
            report['by_class'][f'{domain}|{target}'] = aggregate([r for r in primary if (r['dataset'],r['target']) == (domain,target)])
        for domain in ('crc','tiger'):
            report['by_domain'][domain] = aggregate([r for r in primary if r['dataset'] == domain])
        for low, high in ((0,.05),(.05,.2),(.2,.5),(.5,1.000001)):
            selected = [r for r in primary if low <= r['positive']/r['valid'] < high]
            if selected:
                report['by_area'][f'{low}-{high}'] = aggregate(selected)
        for budget in ('1+1','3+1','3+2','5+1'):
            report['by_budget'][budget] = aggregate([r for r in rows if r['click_budget'] == budget])
        train = a.root/f'train_{method}_1025_131451'
        summary = read(train/'summary.json')
        assert summary['status'] == 'complete' and summary['step'] == 10000
        report['development_history'] = [dict(step=h['step'], dice=h['score']) for h in summary['history']]
        history = [json.loads(line) for line in (train/'progress.jsonl').open()]
        assert len(history) == 10000 and all(np.isfinite(r['loss']) for r in history)
        report['loss_windows'] = {str(step): float(np.mean([r['loss'] for r in history[step-1000:step]])) for step in (1000,5000,8000,10000)}
        report['joint_nonzero_backbone_steps'] = sum(r['gradient_norms']['backbone'] > 0 for r in history[1000:])
        best = read(train/'best_checkpoint.json')
        assert sha(best['path']) == best['checkpoint_sha256']
        binding[str(train/'summary.json')] = sha(train/'summary.json')
        binding[str(train/'best_checkpoint.json')] = sha(train/'best_checkpoint.json')
        outputs[method] = report
    def key(r):
        return r['query_id'],r['target'],r['prompt_group'],r['click_budget']
    indexed = {m:{key(r):r for r in rows} for m,rows in runs.items()}
    keys = set(indexed['sam_ft'])
    for m in METHODS:
        assert set(indexed[m]) == keys and len(indexed[m]) == 6040
        for k in keys:
            r, ref = indexed[m][k],indexed['sam_ft'][k]
            for name in ('positive','valid','true_boundary_pixels','positive_xy','negative_xy','split','support_id'):
                assert r[name] == ref[name], (m,k,name)
    base = [r for r in runs['fewclick_m3d_xcell'] if r['click_budget']=='3+2' and r['prompt_group']==0]
    first = [next(r for r in base if r['dataset']==d) for d in ('crc','tiger')]
    crc = [r for r in base if r['dataset']=='crc']
    gap = lambda r:indexed['sam_ft'][key(r)]['dice']-r['dice']
    ranked = sorted(crc,key=lambda r:(gap(r),r['query_id'],r['target']))
    chosen = first+[ranked[-1],ranked[len(ranked)//2]]
    selection = [dict(reason=reason,plan=r,saved={m:indexed[m][key(r)] for m in METHODS}) for reason,r in zip(
        ('first_CRC_fixed_order','first_TIGER_fixed_order','largest_CRC_SAM_minus_FewClick_gap_posthoc','median_CRC_gap_posthoc'),chosen)]
    write(a.output/'validation_selection.json',selection)
    write(a.output/'summary.json',dict(status='complete_saved_count_diagnosis_single_seed',worker_pid=os.getpid(),
        methods=outputs,bindings=binding,source_sha256=sha(__file__),all6040_task_GT_prompt_parity=True,
        limitations=['Single seed diagnostic; no new model superiority inference.',
                    'No saved train-set Dice: loss/validation trend alone cannot establish underfitting or overfitting.',
                    'Saved counts cannot locate pixelwise confusion or prove boundary-only error.']))
    print(json.dumps({m:v['primary'] for m,v in outputs.items()}),flush=True)


def replay(a):
    # Never touch physical GPUs or change the original running process.
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import pyvips
    import torch
    from torch import nn
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    from .adapted_models import build, SAMFineTune
    from .local_data import LocalData
    from .metrics import score
    from PIL import Image
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    data = LocalData(a.tasks)
    selected = read(a.diagnosis/'validation_selection.json')
    plans = [dict(role='val',reason=r['reason'],plan=r['plan'],saved=r['saved']) for r in selected]
    training_path = a.root.parent/'realized_train_1025_125928/training_plans.jsonl'
    training = [json.loads(line) for line in training_path.open()]
    for domain in ('crc','tiger'):
        seen=set()
        for p in training:
            if p['dataset']==domain and p['query_id'] not in seen:
                seen.add(p['query_id'])
                plans.append(dict(role='train',reason='first_two_distinct_actual_training_fields_per_domain',plan=p))
                if len(seen)==2:break
    write(a.output/'plans.json',plans)
    write(a.output/'contract.json',dict(worker_pid=os.getpid(),device='CPU float32 diagnostic only',threads=2,
        source_sha256=sha(__file__),selection_sha256=sha(a.diagnosis/'validation_selection.json'),
        training_sha256=sha(training_path),task_contract_sha256=sha(a.tasks/'contract.json'),
        threshold=.5,query_GT_use='Local simulated prompts/scoring/offline diagnosis only, never WSI inference or threshold/model selection'))
    results=[]
    for method in ('fewclick_m3d_xcell','sam_ft'):
        tick=time.perf_counter()
        best=read(a.root/f'train_{method}_1025_131451/best_checkpoint.json')
        assert sha(best['path'])==best['checkpoint_sha256']
        if method=='sam_ft':
            # Identical class/forward; avoid public adapter's hardcoded CUDA constructor.
            model=SAMFineTune.__new__(SAMFineTune);nn.Module.__init__(model)
            model.sam=sam_model_registry['vit_b'](checkpoint=None)
            model.transform=ResizeLongestSide(model.sam.image_encoder.img_size)
        else:
            model=build(method,torch.device('cpu'))
        saved=torch.load(best['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);del saved
        model.eval()
        for i,item in enumerate(plans):
            p=item['plan'];raw,prompts,supervision=data.materialize(p,torch.device('cpu'),model.contexts,model.cells)
            with torch.inference_mode():
                prediction=model(raw,prompts)
            probability=prediction['pixel_probability'][0].float().numpy()
            assert probability.shape==(512,512) and np.isfinite(probability).all()
            binary=probability>=.5;truth=supervision['truth'][0].numpy();valid=supervision['valid'][0].numpy()
            metrics=score(binary,truth,valid,supervision['mpp'])
            point_metrics={}
            for sign in ('positive','negative'):
                coords=np.floor(np.asarray(p[sign+'_xy'])*512).astype(int)
                hits=binary[coords[:,1],coords[:,0]]
                point_metrics[sign+'_correct_fraction']=float(np.mean(hits if sign=='positive' else ~hits))
            folder=a.output/f'case{i:02d}';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/(method+'.npz'),probability=probability,mask=binary,truth=truth,valid=valid)
            error=np.zeros((512,512,3),np.uint8);error[binary&truth&valid]=[0,180,0]
            error[binary&~truth&valid]=[255,60,60];error[~binary&truth&valid]=[70,100,255]
            Image.fromarray(error).save(folder/(method+'_errors.png'))
            Image.fromarray(raw['rgb'][0].permute(1,2,0).numpy()).save(folder/'RGB.png')
            Image.fromarray((truth*255).astype(np.uint8)).save(folder/'GT.png')
            prior=item.get('saved',{}).get(method)
            row=dict(method=method,case=i,role=item['role'],reason=item['reason'],query_id=p['query_id'],
                dataset=p['dataset'],target=p['target'],metrics=metrics,points=point_metrics,
                checkpoint_sha256=best['checkpoint_sha256'],saved_GPU_metrics=prior and {k:prior[k] for k in ('tp','fp','fn','dice','bf1_um2')},
                CPU_minus_saved_dice=metrics['dice']-prior['dice'] if prior else None)
            results.append(row);write(a.output/'results.json',results)
            write(a.output/'status.json',dict(status='running_CPU_diagnostic_replay',worker_pid=os.getpid(),method=method,case=i,completed=len(results),total=2*len(plans)))
            print(json.dumps(dict(method=method,case=i,dice=metrics['dice'],seconds=time.perf_counter()-tick)),flush=True)
        del model;gc.collect()
    write(a.output/'summary.json',dict(status='complete_CPU_float32_diagnostic_not_formal_GPU_rescore',worker_pid=os.getpid(),rows=len(results),cases=len(plans),
        results_sha256=sha(a.output/'results.json'),limitations='Eight selected cases only; no representative train/val gap or CPU/GPU exact parity claim.'))
    write(a.output/'status.json',dict(status='complete_CPU_diagnostic',worker_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true')
    p.add_argument('--tasks',type=Path)
    p.add_argument('--diagnosis',type=Path)
    a=p.parse_args()
    if a.replay:replay(a)
    else:stats(a)
