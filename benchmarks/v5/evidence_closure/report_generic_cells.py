"""Paired source-cluster statistics for all five fixed Dense cell conditions."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .audit_heldout_local import read, sha, require, close, key
from benchmarks.v5.fair_comparison.protocol import nested


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid()
    def dump(name,value):
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    try:
        while True:
            try:state=read(a.preparation/'status.json')
            except json.JSONDecodeError:time.sleep(1);continue
            if state['status']=='accepted_full_generic_preparation_qualified':
                log=a.preparation.with_suffix('.log').read_text().rstrip()
                if log.endswith('EXIT_CODE=0'):break
                require('EXIT_CODE=' not in log.splitlines()[-1],'preparation nonzero exit')
                time.sleep(1);continue
            require(state['status']!='failed','preparation audit failed')
            producer=state['worker_pid'];proc=Path(f'/proc/{producer}/cmdline')
            require(proc.exists() and b'audit_generic_preparation' in proc.read_bytes(),'specific preparation worker not live')
            dump('status.json',dict(status='waiting_live_preparation',worker_pid=pid,producer_pid=producer,observed_at=time.time(),producer_status=state))
            time.sleep(5)
        local=a.root/'protocol_1025_124208';generic=a.root/'generic_cells_1025_231000'
        proof_paths=[local/'G3_complete_audit_1025_155140.json',a.root/'mechanisms_1025_134753/acceptance_1025_225200/acceptance.json',
            generic/'local_acceptance_1026_183200/summary.json',a.preparation/'summary.json']
        proofs=[read(p) for p in proof_paths]
        require(proofs[0]['status']==proofs[1]['status']=='passed' and proofs[2]['score_rows']==12080 and proofs[3]['projections']==10422,'complete upstream acceptances')
        tasks=local/'tasks_1025_125400';contract=read(tasks/'contract.json')
        plans=read(tasks/'validation_maximum.json');expected={key(p):p for m in plans for p in [nested(m,b) for b in contract['report_budgets']]}
        require(len(expected)==6040,'fixed302pairs x5groups x4budgets')
        base=local/'adaptation_1025_131451';formal=generic/'formal_controls_1026_001500'
        runs={
            'fewclick_m3d_image':base/'evaluate_fewclick_m3d_image_1025_131451',
            'm3d_morphology':a.root/'mechanisms_1025_134753/mechanism_formal_1025_141300/evaluate_m3d_morphology_1025_141300',
            'generic_he_pca64':formal/'evaluate_generic_he_pca64_1026_001500',
            'generic_he_white64':formal/'evaluate_generic_he_white64_1026_001500',
            'fewclick_m3d_xcell':base/'evaluate_fewclick_m3d_xcell_1025_131451'}
        tables={};evidence={};reference=None
        for method,folder in runs.items():
            meta=read(folder/'run.json');done=read(folder/'summary.json')
            require(done['status']=='complete' and not meta['gate'] and meta['method']==method,'complete scorer scope')
            require(sha(str(folder/'rows.jsonl'))==done['rows_sha256'],'score SHA')
            trained=Path(meta['checkpoint']).parent;tr=read(trained/'summary.json');tm=read(trained/'run.json')['contract']
            require(tr['step']==10000 and tm['seed']==20261025 and tm['training_plans_sha256']==tr['training_consumed_sha256']=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','same40k/firstseed')
            require(sha(meta['checkpoint'])==meta['checkpoint_sha256'],'selected weight SHA')
            rows=[json.loads(line) for line in (folder/'rows.jsonl').open()];table={key(r):r for r in rows}
            require(len(rows)==len(table)==6040 and set(table)==set(expected),'complete score coverage')
            for k,r in table.items():
                require(all(r[f]==v for f,v in expected[k].items()),'actual task/point identity')
                close(r['dice'],2*r['tp']/(2*r['tp']+r['fp']+r['fn']),'Dice/count consistency')
                if reference is not None:
                    require(all(r[f]==reference[k][f] for f in ('valid','positive','true_boundary_pixels','boundary_evaluable','space_key','source_id','spatial_group_id')),'same GT eligibility and provenance')
            reference=table if reference is None else reference;tables[method]=table
            evidence[method]=dict(score_folder=str(folder),rows_sha256=done['rows_sha256'],checkpoint_sha256=meta['checkpoint_sha256'],
                selected_step=read(trained/'best_checkpoint.json')['step'],initialization=tm['initialization'])
        clusters=sorted({r['space_key'] for r in reference.values()});require(len(clusters)==75,'same75source spaces')
        indices=np.random.default_rng(20261025).integers(len(clusters),size=(2000,len(clusters)))
        np.save(a.output/'bootstrap_cluster_indices.npy',indices,allow_pickle=False)
        cluster_id={s:i for i,s in enumerate(clusters)};metrics={};bootstrap={};case_rows=[]
        for method,table in tables.items():
            metrics[method]={}
            for budget in contract['report_budgets']:
                pairs=defaultdict(list)
                for r in table.values():
                    if r['click_budget']==budget:pairs[r['pair_id']].append(r)
                require(len(pairs)==302 and all(len(v)==5 for v in pairs.values()),'pair/group coverage')
                sums={m:np.zeros(len(clusters)) for m in ('dice','bf1')};counts={m:np.zeros(len(clusters)) for m in sums}
                for pair,rr in pairs.items():
                    source=rr[0]['space_key'];c=cluster_id[source]
                    values=dict(dice=float(np.mean([r['dice'] for r in rr])),bf1=float(np.mean([r['bf1_um2'] for r in rr])) if rr[0]['boundary_evaluable'] else None)
                    case_rows.append(dict(method=method,budget=budget,pair_id=pair,space_key=source,dataset=rr[0]['dataset'],target=rr[0]['target'],**values))
                    for name,value in values.items():
                        if value is not None:sums[name][c]+=value;counts[name][c]+=1
                metrics[method][budget]={}
                for name in sums:
                    denominator=counts[name][indices].sum(1);require((denominator>0).all(),'defined bootstrap denominator')
                    samples=sums[name][indices].sum(1)/denominator;mean=float(sums[name].sum()/counts[name].sum())
                    bootstrap[method+'|'+budget+'|'+name]=samples
                    metrics[method][budget][name]=dict(mean=mean,ci95=np.quantile(samples,[.025,.975]).tolist(),pairs=int(counts[name].sum()))
                close(metrics[method][budget]['dice']['mean'],read(runs[method]/'summary.json')['metrics'][budget]['macro_pair_dice'],'reported pair Dice')
                close(metrics[method][budget]['bf1']['mean'],read(runs[method]/'summary.json')['metrics'][budget]['bf1_um2'],'reported BF1')
        contrasts=[];methods=list(runs)
        for i,left in enumerate(methods):
            for right in methods[i+1:]:
                for budget in contract['report_budgets']:
                    for name in ('dice','bf1'):
                        difference=bootstrap[right+'|'+budget+'|'+name]-bootstrap[left+'|'+budget+'|'+name]
                        contrasts.append(dict(a=left,b=right,budget=budget,metric=name,b_minus_a=metrics[right][budget][name]['mean']-metrics[left][budget][name]['mean'],ci95=np.quantile(difference,[.025,.975]).tolist()))
        pd.DataFrame(case_rows).to_csv(a.output/'case_means.csv',index=False)
        np.savez_compressed(a.output/'bootstrap_values.npz',**bootstrap)
        result=dict(status='complete_qualified_generic_cell_statistics',worker_pid=pid,metrics=metrics,contrasts=contrasts,methods=evidence,
            upstream_acceptance_sha256={str(p):sha(str(p)) for p in proof_paths},clusters=clusters,bootstrap_seed=20261025,bootstrap_repetitions=2000,
            source_sha256=sha(__file__),score_rows=30200,
            limits=['Single training seed, same development-selected images; not independent-patient/blind-test inference',
                    '75source-space intervals are conditional/exploratory; same-case residual dependence and multiple comparisons remain',
                    'NoCell branch capacity differs; all cell appearance tracks retain geometry/count and detector priors',
                    'XCell internal data excludes current CRC/TIGER per user; public CTransPath separate; generic PCA fits current training images',
                    'Both predeclared HE projections reported; no best-variant selection or isolated pretraining-objective causal claim'])
        dump('summary.json',result)
        text=['# 普通HE与XCell固定五条件对比','','主3正2负，302对先平均5组点；同firstseed，不混入三seed均值。75来源空间配对bootstrap2000，区间为探索性开发任务区间。','',
              '| 条件 | Dice | 来源95%区间 | BF1@2µm |','|---|---:|---:|---:|']
        for method in methods:
            m=metrics[method]['3+2'];lo,hi=m['dice']['ci95']
            text.append(f'|{method}|{m["dice"]["mean"]:.6f}|[{lo:.6f}, {hi:.6f}]|{m["bf1"]["mean"]:.6f}|')
        text+=['','完整10组方法配对×4预算×2指标均保存在summary.json；不挑有利条件。','',
               'XCell内部预训练与公开CTP+训练图像PCA不同；同任务/同74维接口不等于等总预训练或mIF目标隔离因果比较。无细胞分支的参数量也不同。',
               '已完成特征准备和评分独立验收，但仍有同网络预训练目标控制、完整统计重复/WSI/审稿证据闭环等整体任务。']
        (a.output/'report.md').write_text('\n'.join(text)+'\n')
        dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],primary={m:metrics[m]['3+2']['dice'] for m in methods})),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','preparation','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
