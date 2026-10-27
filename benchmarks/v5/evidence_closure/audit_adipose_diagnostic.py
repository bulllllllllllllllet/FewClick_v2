"""Verify saved adipose probabilities against original GT, clicks and score counts."""
import argparse
from collections import defaultdict
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit_heldout_local import read, sha, require, close
from benchmarks.v5.fair_comparison.protocol import nested


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = read(root/'summary.json'); meta = read(root/'run.json')
    require(summary['status'] == 'complete_diagnostic_execution' and not summary['gate'], 'full diagnostic terminal')
    require(root.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'diagnostic shell exit')
    require(sha(str(root/'rows.jsonl')) == summary['rows_sha256'], 'diagnostic row SHA')
    for p, value in meta['inference_source_sha256'].items():require(sha(p) == value, 'inference source identity')
    require(sha('benchmarks/v5/evidence_closure/diagnose_adipose.py') == meta['source_sha256'], 'diagnostic source identity')
    tasks = Path(meta['tasks']); contract = read(tasks/'contract.json')
    require(sha(str(tasks/'validation_maximum.json')) == meta['task_sha256'], 'fixed task SHA')
    catalog = Path(contract['catalog'])/'tiles.parquet'
    require(sha(str(catalog)) == contract['catalog_sha256'], 'original catalog SHA')
    fields = {r['tile_id']:r for r in pd.read_parquet(catalog).to_dict('records')}
    plans = [nested(p, '3+2') for p in read(tasks/'validation_maximum.json')]
    require(len(plans) == 100 and len({p['pair_id'] for p in plans}) == 20, 'twenty pairs/five prompt groups')
    expected = {(p['pair_id'],p['prompt_group']):p for p in plans}
    rows = [json.loads(line) for line in (root/'rows.jsonl').open()]
    primary = defaultdict(list); trajectory = defaultdict(list)
    for r in rows:
        (primary[r['role']] if 'role' in r else trajectory[r['step']]).append(r)
    checked = []; artifacts = {}; pair_means = defaultdict(dict)
    for role in ('all_label','adipose_heldout'):
        rr = primary[role]
        require(len(rr) == 100 and {(r['pair_id'],r['prompt_group']) for r in rr} == set(expected), 'complete fixed primary coverage')
        binding = summary['weight_bindings'][role]
        require(sha(binding['checkpoint']) == binding['sha256'], 'selected weight identity')
        for i, r in enumerate(rr):
            plan = expected[r['pair_id'],r['prompt_group']];field = fields[plan['query_id']]
            with np.load(root/f'{role}_{i:03d}.npz', allow_pickle=False) as saved:
                arrays = {k:saved[k] for k in saved.files}
            gt = np.asarray(Image.open(field['labels']))
            require(hashlib.sha256(gt.tobytes()).hexdigest() == field['labels_array_sha256'], 'actual GT array SHA')
            require(np.array_equal(arrays['truth'],gt == 10) and np.array_equal(arrays['valid'],gt != 255), 'saved mask uses original adipose class10')
            rgb_path = json.loads(field['variants_json'])['original']['image']
            require(np.array_equal(arrays['rgb'],np.asarray(Image.open(rgb_path).convert('RGB'))), 'same real RGB')
            for sign in ('positive','negative'):
                require(np.array_equal(arrays[sign+'_xy'],np.asarray(plan[sign+'_xy'])), 'saved click identity')
            t = arrays['truth'];v = arrays['valid']
            for name, p in [('final',arrays['probability']),('base',arrays['base'])]:
                require(p.shape == gt.shape and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'probability shape/range')
                pred = p >= .5
                counts = (int((pred&t&v).sum()),int((pred&~t&v).sum()),int((~pred&t&v).sum()))
                target = r if name == 'final' else r['base']
                require(counts == tuple(target[k] for k in ('tp','fp','fn')), 'independent saved probability pixel recount')
                close(2*counts[0]/(2*counts[0]+counts[1]+counts[2]),target['dice'],'Dice recount')
            b = arrays['base'].astype(np.float64).clip(1e-6,1-1e-6)
            logits = np.log(b/(1-b))+arrays['residual']
            reconstructed = 1/(1+np.exp(-logits))
            error = float(np.max(np.abs(reconstructed-arrays['probability'])))
            require(error < 2e-6, 'Dense probability equals sigmoid(logit(base)+residual)')
            require(r['original_counts_exact'] and r['original_dice_difference'] == 0, 'all original recorded counts exactly replayed')
            artifacts[f'{role}_{i:03d}.npz'] = sha(str(root/f'{role}_{i:03d}.npz'))
            checked.append(dict(role=role,pair_id=r['pair_id'],prompt_group=r['prompt_group'],dice=r['dice'],base_dice=r['base']['dice'],dense_reconstruction_error=error))
        for pair in {r['pair_id'] for r in rr}:
            pair_means[pair][role] = math.fsum(r['dice'] for r in rr if r['pair_id'] == pair)/5
        close(math.fsum(r['dice'] for r in rr)/100,summary['results'][role]['dice'],'reported primary mean')
    require(set(trajectory) == set(range(1000,10001,1000)), 'all diagnostic checkpoints, no best-checkpoint omission')
    selection = read(Path(summary['weight_bindings']['adipose_heldout']['checkpoint']).parent/'summary.json')['history']
    for point in summary['trajectory']:
        rr = trajectory[point['step']]
        require(len(rr) == 20 and len({r['pair_id'] for r in rr}) == 20, 'twenty fixed trajectory cases')
        close(math.fsum(r['dice'] for r in rr)/20,point['dice'],'checkpoint trajectory mean')
        point['seen_selection_dice'] = next(h['score'] for h in selection if h['step'] == point['step'])
    pd.DataFrame([dict(pair_id=p,**values) for p,values in sorted(pair_means.items())]).to_csv(output/'paired_cases.csv',index=False)
    result = dict(status='passed_200_saved_predictions_and_dense_algebra',checked=checked,source_sha256=sha(__file__),
        diagnostic_summary_sha256=sha(str(root/'summary.json')),saved_prediction_sha256=artifacts,
        results=summary['results'],trajectory=summary['trajectory'],
        limits=['Only two selected models and100fixed3+2tasks each pixel-recounted', 'Checkpoint trajectory is20group0cases only; not used for model selection',
                'Source/GT/click/weight and arithmetic checks exclude specified faults, not every possible implementation defect',
                'Different training distributions; no isolated causal attribution; Dense base co-trained'])
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr = summary['trajectory'];x = [r['step'] for r in tr]
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(x,[r['seen_selection_dice'] for r in tr],'-o',label='Seen development selection')
    axes[0].plot(x,[r['dice'] for r in tr],'-o',label='Adipose heldout: diagnostic only')
    axes[0].set(title='Seen selection vs heldout diagnostic',xlabel='Training update',ylabel='Dice',ylim=(0,1));axes[0].legend()
    axes[1].plot(x,[r['base_dice'] for r in tr],'-o',label='Before Dense (co-trained)')
    axes[1].plot(x,[r['dice'] for r in tr],'-o',label='Final')
    axes[1].set(title='Fixed 20 adipose cases, prompt group 0',xlabel='Training update',ylabel='Dice',ylim=(0,1));axes[1].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(output/'diagnostic_trajectory.png',dpi=180);fig.savefig(output/'diagnostic_trajectory.pdf');plt.close(fig)
    text = ['# 脂肪留出异常核查', '', '200份保存概率均与原始GT/点击/RGB一致，独立TP/FP/FN及Dense代数重算通过；原评分200/200逐任务精确重放。', '',
        '| 条件 | 最终Dice | Dense前Dice | 正点平均概率 | 负点平均概率 |', '|---|---:|---:|---:|---:|']
    for role,r in summary['results'].items():
        text.append(f'|{role}|{r["dice"]:.6f}|{r["base_dice"]:.6f}|{r["positive_click_probability"]:.6f}|{r["negative_click_probability"]:.6f}|')
    text += ['', '训练轨迹只用固定20图像目标对的group0；原seen选模不变，禁止以诊断分数重选checkpoint或阈值。',
        '当前定位为预测本身低召回、区域分支偏低及Dense进一步抑制；没有查到这批任务的标签/点击/权重/计数错配。',
        '这不是排除所有实现错误的证明；训练分布变化、公共先验、单seed等仍限制归因。Dense前结果是组合模型内部信号，不能替代独立训练的消融。']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=result['status'],saved_predictions=len(checked),results=result['results'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();run(a.root,a.output)
