"""Replay fixed adipose tasks; inspect Dense/base probabilities without selecting models."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch

from .audit_heldout_local import read, sha, require, key
from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.metrics import score


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summarize_probability(p, truth, valid, prompts):
    prediction = p >= .5
    record = dict(tp=int((prediction & truth & valid).sum()), fp=int((prediction & ~truth & valid).sum()),
                  fn=int((~prediction & truth & valid).sum()))
    denominator = 2*record['tp']+record['fp']+record['fn']
    record['dice'] = 2*record['tp']/denominator if denominator else None
    for name, mask in [('GT', truth & valid), ('nonGT', ~truth & valid)]:
        record[name+'_probability_quantiles'] = np.quantile(p[mask], [0,.1,.5,.9,1]).tolist() if mask.any() else None
    for sign in ('positive', 'negative'):
        xy = prompts[sign+'_xy'][0].detach().cpu().numpy()
        pixel = np.floor(xy*np.array(p.shape[::-1])).astype(int)
        record[sign+'_point_probabilities'] = p[pixel[:,1],pixel[:,0]].tolist()
    return record


def infer(model, data, plan, device, use_amp=True):
    raw, prompts, supervision = data.materialize(plan, device, model.contexts, model.cells)
    capture = {}
    hook = model.core.dense.register_forward_pre_hook(lambda module, args: capture.update(base=args[0]['base'].detach()))
    try:
        with torch.inference_mode(), (torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()):
            result = model(raw, prompts)
    finally:
        hook.remove()
    probability = result['pixel_probability'][0].float().cpu().numpy()
    base = capture['base'][0].float().cpu().numpy()
    residual = result['residual'][0].float().cpu().numpy()
    require(np.isfinite(probability).all() and np.isfinite(base).all() and np.isfinite(residual).all(), 'finite real inference')
    require(torch.equal(result['pixel_probability'], result['pixel_logits'].sigmoid()), 'probability/logit identity')
    truth = supervision['truth'][0].cpu().numpy();valid = supervision['valid'][0].cpu().numpy()
    labels = supervision['labels'][0].cpu().numpy()
    record = summarize_probability(probability, truth, valid, prompts)
    record['base'] = summarize_probability(base, truth, valid, prompts)
    record['residual_GT_quantiles'] = np.quantile(residual[truth & valid], [0,.1,.5,.9,1]).tolist()
    record['predicted_original_label_counts'] = {str(int(v)): int(n) for v,n in zip(*np.unique(labels[(probability >= .5) & valid], return_counts=True))}
    record['positive'] = int((truth & valid).sum());record['valid'] = int(valid.sum())
    return record, dict(probability=probability, base=base, residual=residual, truth=truth, valid=valid,
        rgb=raw['rgb'][0].permute(1,2,0).cpu().numpy())


def preview(path, arrays, plan, label):
    panels = []
    for title, data in [('H&E + clicks', arrays['rgb']), ('Target GT', arrays['truth']*255),
                         ('Before Dense >=.5', (arrays['base'] >= .5)*255), ('Final >=.5', (arrays['probability'] >= .5)*255)]:
        im = Image.fromarray(np.asarray(data, np.uint8)).convert('RGB')
        if title == 'H&E + clicks':
            draw = ImageDraw.Draw(im)
            for sign, color in [('positive', '#00ff00'), ('negative', '#ff3030')]:
                for xy in plan[sign+'_xy']:
                    x,y = np.asarray(xy)*512
                    draw.ellipse((x-5,y-5,x+5,y+5), outline=color, width=3)
        panel = Image.new('RGB',(512,542),'white');panel.paste(im,(0,30));ImageDraw.Draw(panel).text((8,8),title,fill='black');panels.append(panel)
    canvas = Image.new('RGB',(2048,572),'white')
    for i,panel in enumerate(panels):canvas.paste(panel,(512*i,30))
    ImageDraw.Draw(canvas).text((8,8),label,fill='black');canvas.save(path)


def load_model(path, method, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved['contract']['method'] == method, 'saved method')
    model = build(method, device);model.load_state_dict(saved['model'], strict=True);model.eval()
    return model


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    pid = os.getpid();start = time.time();dump(a.output/'status.json', dict(status='preparing', worker_pid=pid))
    try:
        torch.set_num_threads(4)
        require(torch.cuda.is_available(), 'declared real GPU required')
        device = torch.device('cuda:0');method = 'fewclick_m3d_image'
        formal = a.heldout_root/'formal_queue_1026_012700'
        train = formal/'train_adipose_fewclick_m3d_image_1026_012700'
        tasks = a.heldout_root/'protocols_geometry_1026_010300/adipose/heldout_local'
        data = LocalData(tasks)
        plans = [nested(p,'3+2') for p in read(tasks/'validation_maximum.json')]
        require(len(plans) == 100 and all(p['target'] == 10 and p['dataset'] == 'crc' for p in plans), 'all fixed adipose tasks')
        if a.gate:plans = plans[:1]
        old_train = a.heldout_root.parent/'protocol_1025_124208/adaptation_1025_131451/train_fewclick_m3d_image_1025_131451'
        old_score = old_train.parent/'evaluate_fewclick_m3d_image_1025_131451'
        roles = {'all_label': old_train, 'adipose_heldout': train}
        score_paths = {'all_label': old_score, 'adipose_heldout':formal/'score_adipose_fewclick_m3d_image_1026_012700'}
        meta = dict(worker_pid=pid, gate=a.gate, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            tasks=str(tasks), task_sha256=sha(str(tasks/'validation_maximum.json')), source_sha256=sha(__file__),
            inference_source_sha256={str(p):sha(str(p)) for p in [Path('benchmarks/v5/fair_comparison/adapted_models.py'),
                Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/fair_comparison/local_data.py')]},
            scope='Fixed-task forensic replay; checkpoint trajectory is diagnostic only, never heldout-based selection')
        dump(a.output/'run.json',meta)
        full_records = {};weight_bindings = {};trajectory = []
        with (a.output/'rows.jsonl').open('x') as handle:
            for role, folder in roles.items():
                best = read(folder/'best_checkpoint.json');weight = Path(best['path'])
                require(sha(str(weight)) == best['checkpoint_sha256'], 'selected weight identity')
                saved_run = read(folder/'run.json')['contract']
                for p,value in saved_run['source_sha256'].items():require(sha(p) == value, 'training source changed')
                original = {key(r):r for r in [json.loads(line) for line in (score_paths[role]/'rows.jsonl').open()]}
                model = load_model(weight,method,device)
                weight_bindings[role] = dict(checkpoint=str(weight),sha256=best['checkpoint_sha256'],selected_step=best['step'])
                records = []
                for i,plan in enumerate(plans):
                    require(all(original[key(plan)][k] == v for k,v in plan.items()), 'same old/new points and task metadata')
                    record, arrays = infer(model,data,plan,device)
                    record.update(role=role,pair_id=plan['pair_id'],prompt_group=plan['prompt_group'],original_dice=original[key(plan)]['dice'])
                    record['original_counts_exact'] = all(record[k] == original[key(plan)][k] for k in ('tp','fp','fn'))
                    record['original_dice_difference'] = record['dice']-original[key(plan)]['dice']
                    np.savez_compressed(a.output/f'{role}_{i:03d}.npz',**arrays,
                        positive_xy=np.asarray(plan['positive_xy']),negative_xy=np.asarray(plan['negative_xy']))
                    if plan['prompt_group'] == 0:
                        preview(a.output/f'{role}_pair_{plan["pair_id"]}.png',arrays,plan,
                            f'{role} pair {plan["pair_id"]}, Dice={record["dice"]:.5f}, beforeDense={record["base"]["dice"]:.5f}')
                    if i == 0:
                        fp32,_ = infer(model,data,plan,device,False)
                        dump(a.output/f'{role}_first_fp32.json', fp32)
                    records.append(record);handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
                    dump(a.output/'status.json',dict(status='replaying',worker_pid=pid,role=role,completed=i+1,total=len(plans)))
                full_records[role] = records
                del model;gc.collect();torch.cuda.empty_cache()
            if not a.gate:
                group0 = [p for p in plans if p['prompt_group'] == 0]
                for step in range(1000,10001,1000):
                    weight = train/f'checkpoint_step_{step:05d}.pth'
                    model = load_model(weight,method,device);records=[]
                    for plan in group0:
                        record,_ = infer(model,data,plan,device)
                        record.update(stage='diagnostic_trajectory',step=step,pair_id=plan['pair_id'],prompt_group=0)
                        handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush();records.append(record)
                    trajectory.append(dict(step=step,checkpoint_sha256=sha(str(weight)),dice=float(np.mean([r['dice'] for r in records])),
                        base_dice=float(np.mean([r['base']['dice'] for r in records])),positive_click_probability=float(np.mean([np.mean(r['positive_point_probabilities']) for r in records]))))
                    dump(a.output/'status.json',dict(status='diagnostic_checkpoint_trajectory',worker_pid=pid,step=step))
                    print(json.dumps(trajectory[-1]),flush=True)
                    del model;gc.collect();torch.cuda.empty_cache()
        results = {}
        for role, records in full_records.items():
            results[role] = dict(tasks=len(records),dice=float(np.mean([r['dice'] for r in records])),base_dice=float(np.mean([r['base']['dice'] for r in records])),
                original_count_exact_rows=sum(r['original_counts_exact'] for r in records),
                maximum_original_dice_difference=max(abs(r['original_dice_difference']) for r in records),
                positive_click_probability=float(np.mean([np.mean(r['positive_point_probabilities']) for r in records])),
                negative_click_probability=float(np.mean([np.mean(r['negative_point_probabilities']) for r in records])))
        dump(a.output/'summary.json',dict(status='complete_diagnostic_gate' if a.gate else 'complete_diagnostic_execution',gate=a.gate,
            worker_pid=pid,results=results,trajectory=trajectory,weight_bindings=weight_bindings,seconds=time.time()-start,
            rows_sha256=sha(str(a.output/'rows.jsonl')),source_sha256=sha(__file__),
            limits=['Diagnostic only; no heldout threshold/model selection', 'Dense input base is co-trained, not standalone comparator',
                    'Different full-label/heldout task distribution and checkpoint selector; no isolated causal attribution', 'Full training-label/prompt exclusion audit already accepted separately']))
        dump(a.output/'status.json',dict(status='complete',worker_pid=pid))
    except Exception:
        dump(a.output/'status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--heldout-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');a=p.parse_args();run(a)
