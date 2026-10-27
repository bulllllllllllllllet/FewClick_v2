"""Matched heldout/reference statistics with explicit remote support dependencies."""
import argparse
from collections import defaultdict,Counter
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .audit_heldout_local import read,sha,require,close,key
from benchmarks.v5.fair_comparison.protocol import nested


FOLDS=('necrosis','adipose','lymphoid_related')
METRICS=('dice','bf1','absent_FPA')


def dependency(plans):
    pairs={p['pair_id']:p for p in plans if p['prompt_group']==0};parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for p in pairs.values():parent[find(p['source_space'])]=find(p['query_space'])
    components=Counter(find(p['source_space']) for p in pairs.values())
    return dict(pairs=len(pairs),query_WSIs=len({p['query_space'] for p in pairs.values()}),
        source_WSIs=len({p['source_space'] for p in pairs.values()}),components=len(components),component_pair_counts=sorted(components.values()),
        interpretation='Query-WSI bootstrap conditional on fixed supports/weights; shared-support dependence not represented')


def statistics(values,clusters,sample):
    result=dict(pairs=len(values),present_pairs=sum(v['present'] for v in values),absent_pairs=sum(not v['present'] for v in values));boot={}
    for metric in METRICS:
        usable=[v for v in values if v[metric] is not None]
        if not usable:result[metric]=None;continue
        sums=np.zeros(len(clusters));counts=np.zeros(len(clusters))
        for v in usable:
            i=clusters.index(v['cluster']);sums[i]+=v[metric];counts[i]+=1
        denominator=counts[sample].sum(1);good=denominator>0
        dist=np.full(2000,np.nan);dist[good]=sums[sample].sum(1)[good]/denominator[good]
        result[metric]=dict(mean=float(np.mean([v[metric] for v in usable])),pairs=len(usable),
            conditional_cluster_ci95=np.quantile(dist[good],[.025,.975]).tolist(),valid_draws=int(good.sum()),contributing_clusters=int((counts>0).sum()))
        boot[metric]=dist
    return result,boot


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid()
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid))
    try:
        root=a.root;protocols=root/'protocols_geometry_1026_010300'
        local_accept=read(root/'local_acceptance_1026_130200/summary.json');remote_accept=read(root/'remote_acceptance_1026_131000/summary.json')
        extraction=read(root/'all_label_extract_1026_023000/summary.json');old_wsi=read(root/'all_label_wsi_reference_1026_023000/summary.json')
        require(local_accept['terminal_rows']==4800 and remote_accept['full_rows']==19360 and remote_accept['saved_masks']==240,'full heldout acceptances')
        proofs=[root/'local_acceptance_1026_130200/summary.json',root/'remote_acceptance_1026_131000/summary.json',
                root/'all_label_independent_1026_024500.json',root/'all_label_wsi_acceptance_1026_030600/summary.json']
        require(read(proofs[2])['extracted_rows']==19320 and read(proofs[3])['full_reference_rows']==4840,'full all-label references')
        local_complete=read(root/'formal_queue_1026_012700/summary.json')['complete'];remote_complete=read(root/'remote_queue_1026_014400/summary.json')['complete']
        folders={}
        for r in extraction['records']:folders[r['fold'],r['mode'],'all_label',r['method']]=Path(r['output'])
        for fold in FOLDS:
            for method in ('fewclick_m3d_image','sam_ft','wsi_sam_ft'):
                folders[fold,'local','heldout',method]=Path(local_complete[fold+'|'+method+'|score'])
                parent=Path(remote_complete[fold+'|'+method+'|score'])
                for variant,p in read(parent/'summary.json')['outputs'].items():folders[fold,'remote','heldout',variant]=Path(p)
            parent=Path(old_wsi['complete'][fold+'|score'])
            for method,p in read(parent/'summary.json')['outputs'].items():folders[fold,'remote','all_label',method]=Path(p)
        require(len(folders)==42,'all42 matched tables')
        outputs={};evidence={};contrasts=[];case_rows=[];draw_files={};bootstrap={};dependencies={};full_rows=0
        for fold in FOLDS:
            for mode in ('local','remote'):
                tasks=protocols/fold/('heldout_'+mode);contract=read(tasks/'contract.json');plans=read(tasks/'validation_maximum.json')
                points_sha=contract['validation_sha256'] if mode=='local' else contract['tasks_sha256']
                require(sha(str(tasks/'validation_maximum.json'))==points_sha,'frozen points SHA')
                expected={key(p):p for maximum in plans for p in [nested(maximum,b) for b in contract['report_budgets']]}
                tables={};reference=None
                for (f,m,role,method),folder in sorted(folders.items()):
                    if f!=fold or m!=mode:continue
                    meta=read(folder/'run.json');done=read(folder/'summary.json');rows=[json.loads(line) for line in (folder/'rows.jsonl').open()]
                    require(sha(str(folder/'rows.jsonl'))==done['rows_sha256'],'accepted score row SHA')
                    table={key(r):r for r in rows};require(len(table)==len(rows)==len(expected) and set(table)==set(expected),'full table coverage')
                    for k,r in table.items():
                        require(all(r.get(field)==value for field,value in expected[k].items()),'same fixed task and click')
                        if r['positive']:close(r['dice'],2*r['tp']/(2*r['tp']+r['fp']+r['fn']),'Dice count algebra')
                        else:require(r['dice'] is None,'absent Dice NA')
                        close(r['false_positive_fraction'],r['fp']/r['valid'],'valid annotation FPA')
                        if reference is not None:
                            require(all(r[field]==reference[k][field] for field in ('positive','valid','true_boundary_pixels','boundary_evaluable','space_key','source_id','spatial_group_id')),'same original GT and provenance')
                    reference=table if reference is None else reference
                    weight=meta.get('checkpoint_sha256',meta.get('selected_checkpoint_sha256'))
                    if role=='heldout':
                        base_method=meta.get('adapted_method',meta.get('method'))
                        if mode=='remote':base_method=read(folder.parent/'run.json')['adapted_method']
                        accepted=local_accept['conditions'][fold+'|'+base_method]
                        require(weight==accepted['checkpoint_sha256'] if mode=='local' else read(folder.parent/'run.json')['checkpoint_sha256']==accepted['checkpoint_sha256'],'selected seen weight binding')
                        if mode=='remote':
                            match=[x for x in remote_accept['checks'] if x['stage']=='score' and x['fold']==fold and x['variant']==method]
                            require(len(match)==1 and match[0]['rows_sha256']==done['rows_sha256'],'accepted remote row binding')
                        else:require(done['rows_sha256']==accepted['rows_sha256'],'accepted local row binding')
                    tables[role,method]=table;full_rows+=len(rows)
                    evidence['|'.join([fold,mode,role,method])]=dict(folder=str(folder),rows=len(rows),rows_sha256=done['rows_sha256'],
                        metadata_sha256=sha(str(folder/'run.json')),weight_sha256=weight if mode=='local' or role=='all_label' else read(folder.parent/'run.json')['checkpoint_sha256'])
                methods=sorted({method for role,method in tables})
                require(set(tables)=={(role,method) for role in ('all_label','heldout') for method in methods},'paired roles/methods')
                first=next(iter(reference.values()));cluster_field='space_key' if mode=='local' else 'query_space'
                cluster_domain={r[cluster_field]:r['dataset'] for r in reference.values()};clusters=sorted(cluster_domain)
                rng=np.random.default_rng(20261025);draws=[]
                for domain in sorted(set(cluster_domain.values())):
                    ids=np.array([i for i,c in enumerate(clusters) if cluster_domain[c]==domain]);draws.append(rng.choice(ids,size=(2000,len(ids)),replace=True))
                sample=np.concatenate(draws,axis=1);draw_path=a.output/f'draws_{fold}_{mode}.npy';np.save(draw_path,sample,allow_pickle=False)
                draw_files[fold+'|'+mode]=dict(path=str(draw_path),clusters=clusters,cluster_dataset=cluster_domain)
                pair_plans={p['pair_id']:p for p in plans if p['prompt_group']==0}
                if mode=='remote':dependencies[fold]=dependency(plans)
                else:dependencies.setdefault('local_sources',{})[fold]=dict(source_spaces=len(clusters),pairs=len(pair_plans))
                mode_stats={};overall_boot={}
                for (role,method),table in tables.items():
                    mode_stats[role+'|'+method]={}
                    for budget in contract['report_budgets']:
                        groups=defaultdict(list)
                        for r in table.values():
                            if r['click_budget']==budget:groups[r['pair_id']].append(r)
                        values=[]
                        for pair,rr in groups.items():
                            require(len(rr)==5 and {r['prompt_group'] for r in rr}==set(range(5)),'five independent point groups')
                            r=rr[0];present=r['positive']>0
                            bf=[x['bf1_um2'] for x in rr if x['bf1_um2'] is not None] if present else []
                            v=dict(fold=fold,mode=mode,role=role,method=method,budget=budget,pair_id=pair,dataset=r['dataset'],
                                cluster=r[cluster_field],present=present,query_id=r['query_id'],support_id=r['support_id'],
                                query_valid_pixels=r['valid'],query_total_pixels=262144,partially_annotated_query=r['valid']<262144,
                                dice=float(np.mean([x['dice'] for x in rr])) if present else None,bf1=float(np.mean(bf)) if bf else None,
                                absent_FPA=float(np.mean([x['false_positive_fraction'] for x in rr])) if not present else None)
                            if mode=='remote':v.update(kind=r['kind'],protocol=r['protocol'],source_space=r['source_space'],query_space=r['query_space'],distance_um=r['support_query_distance_um'])
                            values.append(v);case_rows.append(v)
                        strata=defaultdict(list)
                        for v in values:
                            names=['overall',v['dataset']]
                            if mode=='remote':
                                names += [v['protocol'],v['kind'],v['dataset']+'|'+v['kind']]
                                if v['distance_um'] is not None:names += ['same_wsi_distance_'+('lt2mm' if v['distance_um']<2000 else '2to5mm' if v['distance_um']<5000 else 'ge5mm')]
                            for name in names:strata[name].append(v)
                        mode_stats[role+'|'+method][budget]={}
                        for name,items in strata.items():
                            stats,dist=statistics(items,clusters,sample);stats['partially_annotated_pairs']=sum(v['partially_annotated_query'] for v in items)
                            mode_stats[role+'|'+method][budget][name]=stats
                            if name=='overall':
                                for metric,samples in dist.items():
                                    identity='|'.join([fold,mode,role,method,budget,metric]);bootstrap[identity]=samples;overall_boot[role,method,budget,metric]=samples
                comparisons=[]
                for role in ('all_label','heldout'):
                    for i,left in enumerate(methods):
                        for right in methods[i+1:]:comparisons.append((role,left,role,right,'within_role_method'))
                for method in methods:comparisons.append(('all_label',method,'heldout',method,'descriptive_exposure_difference'))
                for ar,am,br,bm,kind in comparisons:
                    for budget in contract['report_budgets']:
                        for metric in METRICS:
                            if (ar,am,budget,metric) not in overall_boot:continue
                            left=overall_boot[ar,am,budget,metric];right=overall_boot[br,bm,budget,metric]
                            good=np.isfinite(left)&np.isfinite(right);require(np.array_equal(np.isfinite(left),np.isfinite(right)),'paired metric eligibility')
                            delta=right[good]-left[good]
                            lm=mode_stats[ar+'|'+am][budget]['overall'][metric]['mean'];rm=mode_stats[br+'|'+bm][budget]['overall'][metric]['mean']
                            contrasts.append(dict(fold=fold,mode=mode,kind=kind,a_role=ar,a_method=am,b_role=br,b_method=bm,budget=budget,metric=metric,
                                b_minus_a=rm-lm,conditional_paired_ci95=np.quantile(delta,[.025,.975]).tolist(),valid_draws=int(good.sum())))
                outputs[fold+'|'+mode]=mode_stats
                dump('status.json',dict(status='summarizing',worker_pid=pid,fold=fold,mode=mode,tables=len(evidence),rows=full_rows))
                print(json.dumps(dict(fold=fold,mode=mode,tables=len(tables),clusters=len(clusters))),flush=True)
        require(full_rows==48320 and len(evidence)==42 and len(case_rows)==9664,'all matched accepted rows')
        pd.DataFrame(case_rows).to_csv(a.output/'case_means.csv',index=False);np.savez_compressed(a.output/'bootstrap_values.npz',**bootstrap)
        result=dict(status='complete_qualified_heldout_statistics',worker_pid=pid,metrics=outputs,contrasts=contrasts,dependencies=dependencies,
            evidence=evidence,input_acceptance_sha256={str(p):sha(str(p)) for p in proofs},draws=draw_files,
            score_rows=full_rows,case_rows=len(case_rows),bootstrap_seed=20261025,bootstrap_repetitions=2000,source_sha256=sha(__file__),
            limits=['Label supervision heldout, not all morphology/public prior unseen; all runs single seed',
                    'Local simulated query clicks and remote source-only prompts are distinct tasks/tables',
                    'All-label references expose target during adaptation AND development checkpoint selection; different recipe distribution; differences descriptive, not causal label deletion',
                    'Conditional query WSI bootstrap omits support dependence; remote components3/2/1 too few for reliable population inference',
                    'Absence only on valid annotation; unknown query pixels not negative',
                    'SAM/WSI remote tracks custom feature prototypes; graphDense fine maps differ from region-only prototype; no official native global decoder claim',
                    'Multiple unadjusted exploratory contrasts; no confirmatory superiority or independent-patient claim'])
        dump('summary.json',result)
        text=['# 留出类与全标签同任务参考统计','','local三方法和remote四方法分别报告。全部firstseed；先平均同pair五点组，区间是分域来源/查询WSI聚类的探索性区间。','',
              '| 概念 | 本图方法 | 全标签参考Dice | 留出Dice | 留出−参考（描述） |','|---|---|---:|---:|---:|']
        for fold in FOLDS:
            data=outputs[fold+'|local']
            for method in ('fewclick_m3d_image','sam_ft','wsi_sam_ft'):
                old=data['all_label|'+method]['3+2']['overall']['dice']['mean'];new=data['heldout|'+method]['3+2']['overall']['dice']['mean']
                text.append(f'|{fold}|{method}|{old:.6f}|{new:.6f}|{new-old:+.6f}|')
        text+=['','| 概念 | 迁移方法（留出权重） | present Dice | absent FPA |','|---|---|---:|---:|']
        for fold in FOLDS:
            for name,values in outputs[fold+'|remote'].items():
                if not name.startswith('heldout|'):continue
                m=values['3+2']['overall'];text.append(f'|{fold}|{name.split("|",1)[1]}|{m["dice"]["mean"]:.6f}|{m["absent_FPA"]["mean"]:.6f}|')
        text+=['','支持—查询组件数分别3/2/1，组件pair数分别[2,49,64]/[15,50]/[62]，不能把115/65/62个pair或查询WSI视为独立总体样本。',
               '两role训练标签、任务分布和选模信息不同；全标签不是未见类结果，差值不能叫纯监督删除效应。',
               '全部预算、来源区间、逐域/same-cross/距离分层、同方法描述配对与方法间对比见summary.json。保留所有结果，不按脂肪诊断轨迹重新选模。',
               '单seed/开发视野/未知标注区/先验与特征接口差异限制均保留；这不关闭TLS、整片效率与reviewer整体证据任务。']
        (a.output/'report.md').write_text('\n'.join(text)+'\n');dump('status.json',dict(status=result['status'],worker_pid=pid));print('Complete42tables/48320rows matched statistics',flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
