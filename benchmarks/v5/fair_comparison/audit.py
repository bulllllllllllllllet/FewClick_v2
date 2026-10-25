"""Audit existing four-arm evidence without replacing any prior artifact."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .metrics import boundary
from benchmarks.v5.multidomain_training.scoring import split_target_tasks


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    catalog = root/'training_catalog_1025_040737'
    rows = pd.read_parquet(catalog/'tiles.parquet').to_dict('records')
    by_id = {r['tile_id']:r for r in rows}; edges = {}
    def gt_edges(row):
        key = row['query_id'], row['target']
        if key not in edges:
            meta = by_id[key[0]]; gt = np.asarray(Image.open(meta['labels']))
            if hashlib.sha256(gt.tobytes()).hexdigest() != meta['labels_array_sha256']:
                raise ValueError('label identity changed')
            edges[key] = int(boundary(gt == key[1], gt != 255).sum())
        return edges[key]
    arms = ['single_region','multi_region','single_dense','multi_dense']; audit = {}; corrected = []
    for arm in arms:
        directory = root/f'{arm}_1025_040737'
        contract = json.loads((directory/'run.json').read_text())
        summary = json.loads((directory/'summary.json').read_text())
        selected = json.loads((directory/'best_checkpoint.json').read_text())
        validation = directory/f'validation_{selected["step"]:05d}_1025'
        paths = [('3+2',validation/'rows.json')]
        paths.extend((f'{n}+1',root/f'prompt_budgets_{arm}_1025_040737'/f'budget_{n}p1n_1025'/'rows.json') for n in (1,3,5))
        for budget, path in paths:
            values = json.loads(path.read_text()); primary,_ = split_target_tasks(values)
            present = [r for r in primary if r['positive'] > 0]
            eligible = [r for r in present if gt_edges(r)>0]
            existing = [r['bf1_um2'] for r in present if r['bf1_um2'] is not None]
            filled = [float(r['bf1_um2']) if r['bf1_um2'] is not None else 0. for r in eligible]
            corrected.append(dict(arm=arm,budget=budget,selected_step=selected['step'],
                dice=float(np.mean([r['dice'] for r in present])),
                legacy_square_bf1=float(np.mean(existing)),legacy_denominator=len(existing),
                corrected_square_bf1=float(np.mean(filled)),fixed_gt_denominator=len(filled),
                missed_boundary_cases_added=sum(r['bf1_um2'] is None for r in eligible),
                source_rows=str(path),source_sha256=digest(path)))
        task_path = directory/'training_episodes.jsonl'
        task_count = 0; kinds = Counter()
        with task_path.open() as handle:
            for line in handle:
                task_count += 1; kinds[json.loads(line)['kind']] += 1
        audit[arm] = dict(status=summary['status'], steps=summary['step'], tasks=task_count,
            training_tasks_sha256=digest(task_path),validation_sha256=digest(directory/'validation_plans.json'),
            budget=contract['contract']['budget'],weights=contract['contract']['weights'],
            parameters=contract['parameters_total'],seconds=summary['seconds'],task_kinds=dict(kinds),
            checkpoint_reload_exact=summary['checkpoint_reload_exact'],selected=selected,
            checkpoint_sha256=digest(selected['path']))
    for name in ('training_tasks_sha256','validation_sha256','steps','tasks','budget'):
        if len({json.dumps(v[name],sort_keys=True) for v in audit.values()})!=1:
            raise ValueError('four-arm parity failed: '+name)
    if not all(v['status']=='complete' and v['checkpoint_reload_exact'] for v in audit.values()):
        raise ValueError('training incomplete or checkpoint gate failed')
    # Repeat the largest-footprint overlap audit, including physical source space.
    spaces = defaultdict(lambda:defaultdict(list))
    for r in rows:
        spaces[r['space_key']][r['split']].append(json.loads(r['multiscale_json'])['context_um'])
    overlaps = 0
    for splits in spaces.values():
        val = np.asarray(splits['val'], float)
        if not len(val): continue
        for a in splits['train']:
            overlaps += int(((np.minimum(a[2],val[:,2])>np.maximum(a[0],val[:,0])) &
                             (np.minimum(a[3],val[:,3])>np.maximum(a[1],val[:,1]))).sum())
    if overlaps: raise ValueError('cross-split input footprint overlap')
    report = dict(status='complete',arms=audit,counts=dict(Counter(r['dataset']+'|'+r['split'] for r in rows)),
        catalog_sha256=digest(catalog/'tiles.parquet'),largest_input_cross_split_overlap_pairs=overlaps,
        corrected_metrics=corrected,metric='legacy square tolerance; GT boundary denominator correction only',
        original_reports_preserved=True,selection_unchanged='Dice minus .25*absent FPA; BF1 not used',
        allowed_claims=['matched incremental tasks/updates among four retained-J5 arms','within-cohort spatial development validation'],
        prohibited_claims=['equal total target supervision versus external FT','capacity-matched hierarchy attribution',
                           'independent-patient generalization','unseen-concept evidence from retained J5'],
        unresolved=['historical J5 evaluation-region exposure','XCell pretraining overlap','single training seed'])
    write(output/'audit.json',report)
    text = ['# 四组实验公平性审计与 BF1 分母修正','',
            '新增训练任务、验证任务、更新预算一致；最大输入视野跨训练/验证重叠数为 0。',
            '历史 J5 监督、输入视野和模型容量差异仍然存在，因此不能宣称外部等总监督或等容量。',
            '以下仅修正旧版方形容差 BF1 的统计分母；不是新的欧氏距离 BF1。原始报告保留。',
            '选模依据 Dice−0.25×缺失目标误报面积，不含 BF1，因此本次分母修正不改变已选 checkpoint。','',
            '| 模型 | 预算 | Dice | 原 BF1 / n | 修正 BF1 / n |','|---|---|---|---|---|']
    for r in corrected:
        text.append(f"|{r['arm']}|{r['budget']}|{r['dice']:.4f}|{r['legacy_square_bf1']:.4f} / {r['legacy_denominator']}|{r['corrected_square_bf1']:.4f} / {r['fixed_gt_denominator']}|")
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='complete',output=str(output),primary=[r for r in corrected if r['budget']=='3+2']),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prior-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    run(args.prior_run,args.output)
