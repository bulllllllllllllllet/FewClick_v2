"""Complete external-task parity and paired ROI-cluster uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .prepare import digest,write
from benchmarks.v5.fair_comparison.metrics import VERSION
from benchmarks.v5.fair_comparison.protocol import nested


def identity(row):return row['pair_id'],row['prompt_group'],row['click_budget']


def report(tasks,runs,output,repetitions=2000):
    output.mkdir(parents=True,exist_ok=False)
    contract=json.loads((tasks/'contract.json').read_text());protocol_sha=digest(tasks/'contract.json')
    if digest(tasks/'local_maximum.json')!=contract['clicks_sha256']:raise ValueError('external clicks changed')
    maximum=json.loads((tasks/'local_maximum.json').read_text())
    expected={identity(p):p for plan in maximum for p in [nested(plan,b) for b in contract['report_budgets']]}
    tables={};records={};reference=None
    for run in runs:
        metadata=json.loads((run/'run.json').read_text());result=json.loads((run/'summary.json').read_text())
        if result['status']!='complete' or result['gate'] or metadata['metric_version']!=VERSION:
            raise ValueError('incomplete/gate/different-metric run')
        if metadata['protocol_sha256']!=protocol_sha or result['rows_sha256']!=digest(run/'rows.jsonl'):
            raise ValueError('external protocol/row identity changed')
        rows=[json.loads(line) for line in (run/'rows.jsonl').read_text().splitlines()]
        table={identity(r):r for r in rows}
        if len(table)!=len(rows) or set(table)!=set(expected):raise ValueError('missing/duplicate external tasks')
        for key,row in table.items():
            for field in ('target','dataset','query_id','positive_xy','negative_xy'):
                if row[field]!=expected[key][field]:raise ValueError('external task or point mismatch')
            if not np.isclose(row['dice'],2*row['tp']/(2*row['tp']+row['fp']+row['fn']),atol=1e-12,rtol=0):
                raise ValueError('external Dice/count inconsistency')
            if reference is not None:
                for field in ('valid','positive','true_boundary_pixels','source_id','space_key'):
                    if row[field]!=reference[key][field]:raise ValueError('external GT/source mismatch')
        if reference is None:reference=table
        name=metadata['method']
        if name in tables:raise ValueError('duplicate external method')
        tables[name]=table
        records[name]=dict(run=str(run),rows_sha256=digest(run/'rows.jsonl'),
            stage='public zero adaptation' if metadata.get('adaptation_steps')==0 else 'CRC/TIGER adapted',
            device=metadata.get('device'),precision=metadata.get('precision','float32 CPU'),
            initialization=metadata.get('prior',metadata.get('initialization')))
    clusters=sorted({r['source_id'] for r in reference.values()});index={s:i for i,s in enumerate(clusters)}
    rng=np.random.default_rng(20261025);sample=rng.integers(len(clusters),size=(repetitions,len(clusters)))
    metrics={};distributions={}
    for method,table in tables.items():
        metrics[method]={};distributions[method]={}
        for budget in contract['report_budgets']:
            pairs=defaultdict(list)
            for row in table.values():
                if row['click_budget']==budget:pairs[row['pair_id']].append(row)
            sums=np.zeros(len(clusters));counts=np.zeros(len(clusters));classes=defaultdict(list);values=[]
            for group in pairs.values():
                value=float(np.mean([r['dice'] for r in group]));row=group[0]
                if len(group)!=5:raise ValueError('incomplete external point groups')
                values.append(value);classes[row['target']].append(value)
                sums[index[row['source_id']]]+=value;counts[index[row['source_id']]]+=1
            bootstrap=sums[sample].sum(1)/counts[sample].sum(1);distributions[method][budget]=bootstrap
            bf=[r['bf1_um2'] for r in table.values() if r['click_budget']==budget and r['bf1_um2'] is not None]
            metrics[method][budget]=dict(macro_pair_dice=float(np.mean(values)),
                roi_cluster_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),pairs=len(pairs),point_groups=5,ROI_clusters=len(clusters),
                macro_class_dice=float(np.mean([np.mean(v) for v in classes.values()])),
                per_class_dice={str(c):float(np.mean(v)) for c,v in sorted(classes.items())},
                bf1_um2=float(np.mean(bf)) if bf else None,bf1_denominator=len(bf))
    differences=[];names=list(tables)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for budget in contract['report_budgets']:
                differences.append(dict(a=a,b=b,budget=budget,b_minus_a=metrics[b][budget]['macro_pair_dice']-metrics[a][budget]['macro_pair_dice'],
                    paired_roi_cluster_ci95=np.quantile(distributions[b][budget]-distributions[a][budget],[.025,.975]).tolist()))
    limits=['Only10 native-context ROI centres; no independent-patient claim',
        'Five click groups are averaged within each image-target pair, not independent patients or training seeds',
        'Current development-selected models and different public/internal priors/views; no matched-total-pretraining claim',
        'Public CPUfloat32 and adapted GPUbfloat16 conditions must be disclosed; this is not a compute/latency comparison',
        'No model or threshold selection on this external subset; nine classes, not complete IGNITE16-class coverage']
    write(output/'summary.json',dict(status='complete',scope='complete supplied runs only; not completion of allG7',
        protocol_sha256=protocol_sha,rows_per_method=len(expected),records=records,metrics=metrics,
        paired_differences=differences,bootstrap_repetitions=repetitions,bootstrap_seed=20261025,limits=limits,
        source_sha256=digest(__file__)))
    text=['# 原生完整视野外部ROI比较','',
        '10个IGNITE几何中心、34个图像目标对、每对5组模拟点击；先对点组平均，再对图像目标对平均。按来源ROI配对bootstrap2000次。',
        '仅汇总本次指定且全部通过680行身份核验的方法，不能代表FewClick等尚未完成的方法。','',
        '| 方法 | 条件 | 预算 | Dice | ROI聚类95%区间 | 类宏平均Dice | BF1@2µm |',
        '|---|---|---|---|---|---|---|']
    for method,values in metrics.items():
        for budget,m in values.items():
            lo,hi=m['roi_cluster_ci95']
            text.append(f"|{method}|{records[method]['stage']} / {records[method]['device']}|{budget}|{m['macro_pair_dice']:.4f}|[{lo:.4f}, {hi:.4f}]|{m['macro_class_dice']:.4f}|{m['bf1_um2']:.4f}|")
    text += ['','10个ROI不是已证实的10个独立病例；区间不含训练seed不确定性。SAM使用fine，WSI-SAM使用fine+真实2x，FewClick使用fine+2x+4x和对应细胞条件；各自先验、输入与数值模式见run.json。不得据此称等总训练/等计算。']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='complete',methods=names,ROI_clusters=len(clusters),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.tasks,a.runs,a.output)
