"""Paired complete-task comparisons and source-cluster uncertainty, without oracle selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .audit import digest,write


def key(row):return row['pair_id'],row['prompt_group'],row['click_budget']


def report(tasks,runs,output,repetitions=2000):
    output.mkdir(parents=True,exist_ok=False);contract=json.loads((tasks/'contract.json').read_text())
    protocol_sha=digest(tasks/'contract.json');tables={};records={};summary={};reference=None
    maximum=json.loads((tasks/'validation_maximum.json').read_text())
    expected={(p['pair_id'],p['prompt_group'],b) for p in maximum for b in contract['report_budgets']}
    for run in runs:
        meta=json.loads((run/'run.json').read_text());result=json.loads((run/'summary.json').read_text())
        if result['status']!='complete' or meta.get('gate',False):raise ValueError('incomplete or gate run')
        if meta['protocol_sha256']!=protocol_sha:raise ValueError('different evaluation protocol')
        rows=[json.loads(line) for line in (run/'rows.jsonl').read_text().splitlines()]
        table={key(r):r for r in rows}
        if len(table)!=len(rows) or set(table)!=expected:raise ValueError('missing/duplicate tasks')
        if reference is not None:
            for identity,row in table.items():
                other=reference[identity]
                for field in ('dataset','target','query_id','positive_xy','negative_xy','valid','positive','true_boundary_pixels'):
                    if row[field]!=other[field]:raise ValueError('unmatched task/prompt/GT: '+field)
        else:reference=table
        method=meta['method']
        if method in tables:raise ValueError('duplicate method')
        stage=meta.get('stage','retained-J5' if method.startswith('retained_') else 'adapted')
        tables[method]=table;records[method]=dict(run=str(run),stage=stage,rows_sha256=digest(run/'rows.jsonl'),prior=meta.get('weights',meta.get('initialization')))
    clusters=sorted({row['space_key'] for row in reference.values()});cluster_ids={name:i for i,name in enumerate(clusters)}
    rng=np.random.default_rng(20261025);sample=rng.integers(len(clusters),size=(repetitions,len(clusters)))
    arrays={}
    for method,table in tables.items():
        summary[method]={};arrays[method]={}
        for budget in contract['report_budgets']:
            pairs=defaultdict(list)
            for row in table.values():
                if row['click_budget']==budget:pairs[row['pair_id']].append(row)
            pair_values={i:float(np.mean([r['dice'] for r in rows])) for i,rows in pairs.items()}
            by_class=defaultdict(list);sums=np.zeros(len(clusters));counts=np.zeros(len(clusters))
            for identity,rows in pairs.items():
                value=pair_values[identity];row=rows[0];by_class[row['dataset']+'|'+str(row['target'])].append(value)
                index=cluster_ids[row['space_key']];sums[index]+=value;counts[index]+=1
            bootstrap=sums[sample].sum(1)/counts[sample].sum(1);arrays[method][budget]=bootstrap
            bf=[r['bf1_um2'] for r in table.values() if r['click_budget']==budget and r['bf1_um2'] is not None]
            summary[method][budget]=dict(pairs=len(pairs),point_groups=contract['validation_point_groups'],
                macro_pair_dice=float(np.mean(list(pair_values.values()))),
                source_cluster_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),source_clusters=len(clusters),
                macro_class_dice=float(np.mean([np.mean(v) for v in by_class.values()])),
                bf1_um2=float(np.mean(bf)),bf1_denominator=len(bf),
                per_class_dice={k:float(np.mean(v)) for k,v in sorted(by_class.items())})
    paired=[];methods=list(tables)
    for i,a in enumerate(methods):
        for b in methods[i+1:]:
            for budget in contract['report_budgets']:
                difference=arrays[b][budget]-arrays[a][budget]
                paired.append(dict(a=a,b=b,budget=budget,b_minus_a=summary[b][budget]['macro_pair_dice']-summary[a][budget]['macro_pair_dice'],
                    paired_source_cluster_ci95=np.quantile(difference,[.025,.975]).tolist()))
    result=dict(status='complete',protocol_sha256=protocol_sha,rows_per_method=len(expected),records=records,
        metrics=summary,paired_differences=paired,bootstrap_repetitions=repetitions,bootstrap_seed=20261025,
        scope='fixed within-cohort spatial development; sources resampled with paired methods; point groups averaged within image-target pair',
        limits=['same-case dependence may remain across source spaces','single training seed; no training-seed uncertainty',
                'different pretraining and input fields; matching tasks alone does not match total supervision',
                'closed reference receives known target class ID and all-class training labels'])
    write(output/'summary.json',result)
    text=['# 固定本图任务的完整对比','',
        '所有方法使用相同任务、点击坐标、GT与指标版本。public、adapted及retained-J5行分别记账；任务相同不代表等总监督。',
        f"{contract['validation_pairs']}个图像目标对，每对{contract['validation_point_groups']}组模拟点击；同一来源空间重采样{repetitions}次，区间只反映当前空间开发任务的抽样变化。",'',
        '| 方法 | 条件 | 点击预算 | Dice | 来源聚类95%区间 | 类别宏平均Dice | BF1 / 分母 |',
        '|---|---|---|---|---|---|---|']
    for method,values in summary.items():
        for budget,m in values.items():
            lo,hi=m['source_cluster_bootstrap_ci95']
            text.append(f"|{method}|{records[method]['stage']}|{budget}|{m['macro_pair_dice']:.4f}|[{lo:.4f}, {hi:.4f}]|{m['macro_class_dice']:.4f}|{m['bf1_um2']:.4f} / {m['bf1_denominator']}|")
    text+=['','同一病例仍可能对应多个来源空间；该区间不是独立病例泛化、训练seed重复或盲测证据。历史FewClick混合协议120任务不得与本表直接排名。',
           '输入差异：SAM使用fine；WSI-SAM使用fine+真实2x；FewClick-M3使用fine+2x+4x，细胞条件依行区分。闭集参考提供类别ID及全类别监督，不能当未见概念证据。来源/权重/输入合同见各run.json。']
    (output/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(status='complete',output=str(output),methods=methods)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.tasks,a.runs,a.output)
