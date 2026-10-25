"""Audit semantic exclusions and feasible prompts without training or image edits."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.protocol import resolve_clicks
from benchmarks.v5.multidomain_data.prepare import LABELS
from benchmarks.v5.multidomain_training.scoring import BACKGROUND_TARGETS

FOLDS={
    'necrosis':dict(exclude={'crc':[3],'tiger':[3,5]},test={'crc':[3],'tiger':[5]},
        reason='TIGER in-situ label3 may include necrotic portions; label5 excludes in-situ necrosis'),
    'adipose':dict(exclude={'crc':[10],'tiger':[7]},test={'crc':[10],'tiger':[]},
        reason='TIGER rest7 may contain adipose but is not a pure adipose target'),
    'lymphoid_related':dict(exclude={'crc':[8],'tiger':[2,6,7]},test={'crc':[8],'tiger':[]},
        reason='Exclude explicit aggregate plus conservative stromal/composite neighbors; not all immune morphology'),
}


def read(path):return json.loads(Path(path).read_text())


def audit(tasks,realized,output):
    output.mkdir(parents=True,exist_ok=False)
    contract=read(tasks/'contract.json');catalog=Path(contract['catalog'])
    require(digest(catalog/'tiles.parquet')==contract['catalog_sha256'],'catalog changed')
    require(digest(tasks/'validation_maximum.json')==contract['validation_sha256'],'validation changed')
    require(digest(realized/'training_plans.jsonl')=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','original40k changed')
    rows=pd.read_parquet(catalog/'tiles.parquet').to_dict('records');fields={r['tile_id']:r for r in rows}
    plans=[json.loads(line) for line in (realized/'training_plans.jsonl').open()]
    by_field=defaultdict(list)
    for p in plans:by_field[p['support_id']].append(p)
    evidence={name:dict(coverage=defaultdict(Counter),old_task_audit=Counter(),probes=[],candidates=[]) for name in FOLDS}
    field_audit=[];probe_strata=defaultdict(set)
    for index,row in enumerate(sorted(rows,key=lambda r:r['tile_id'])):
        labels=np.asarray(Image.open(row['labels']));require(hashlib.sha256(labels.tobytes()).hexdigest()==row['labels_array_sha256'],'actual GT SHA')
        counts=np.bincount(labels.ravel(),minlength=256);domain=row['dataset'];split=row['split']
        eligible=sorted(int(t) for t in json.loads(row['eligible_json']) if int(t) not in BACKGROUND_TARGETS[domain])
        field_audit.append(dict(tile_id=row['tile_id'],dataset=domain,split=split,labels_sha256=row['labels_array_sha256'],
            counts={str(k):int(v) for k,v in enumerate(counts) if v}))
        for name,spec in FOLDS.items():
            entry=evidence[name];excluded=spec['exclude'][domain];removed=int(counts[excluded].sum());valid=int(labels.size-counts[255]-removed)
            target_candidates=[t for t in eligible if t not in excluded and counts[t]>=64 and valid-counts[t]>=64]
            cover=entry['coverage'][domain+'|'+split];cover.update(fields=1,valid_before=int(labels.size-counts[255]),
                removed_pixels=removed,valid_after=valid,candidate_pairs=len(target_candidates),
                fields_with_candidates=int(bool(target_candidates)),fine_clean_fields=int(removed==0),
                fully_annotated_fine_clean_fields=int(removed==0 and counts[255]==0))
            for target in spec['test'][domain]:
                if counts[target]>=64:cover['heldout_positive_fields_target'+str(target)]+=1
            for target in target_candidates:
                entry['candidates'].append(dict(tile_id=row['tile_id'],dataset=domain,split=split,target=target,
                    positive_pixels=int(counts[target]),remaining_valid_pixels=valid))
            for plan in by_field[row['tile_id']]:
                audit_counts=entry['old_task_audit'];audit_counts['tasks']+=1
                audit_counts['positive_targets_excluded']+=int(plan['target'] in excluded)
                any_bad=False
                for sign in ('positive','negative'):
                    xy=np.asarray(plan[sign+'_xy']);pixel=np.floor(xy*np.array(labels.shape[::-1])).astype(int)
                    values=labels[pixel[:,1],pixel[:,0]];bad=np.isin(values,excluded)
                    audit_counts[sign+'_points_in_excluded_labels']+=int(bad.sum());any_bad=any_bad or bool(bad.any())
                    if sign=='negative' and plan['target'] not in excluded:
                        audit_counts['seen_target_tasks_with_excluded_negative_points']+=int(bad.any())
                        audit_counts['seen_target_excluded_negative_points']+=int(bad.sum())
                audit_counts['tasks_with_any_excluded_point']+=int(any_bad)
                if plan['target'] not in excluded and removed>0:audit_counts['seen_target_tasks_with_excluded_loss_pixels']+=1
            stratum=domain,split
            if target_candidates and stratum not in probe_strata[name]:
                masked=labels.copy();masked[np.isin(masked,excluded)]=255
                path=output/f'probe_{name}_{domain}_{split}.png';Image.fromarray(masked.astype(np.uint8)).save(path)
                source=row|dict(labels=str(path),labels_array_sha256=hashlib.sha256(masked.tobytes()).hexdigest())
                base=dict(number=80000000+len(entry['probes']),pair_id=80000000+len(entry['probes']),
                    support_id=row['tile_id'],query_id=row['tile_id'],dataset=domain,target=target_candidates[0],split=split,
                    kind='local',protocol='local',prompt_group=0,click_seed=20261026+len(entry['probes']))
                resolved=resolve_clicks(base,source)
                for sign in ('positive','negative'):
                    xy=np.asarray(resolved[sign+'_xy']);pixel=np.floor(xy*np.array(masked.shape[::-1])).astype(int)
                    sampled=masked[pixel[:,1],pixel[:,0]]
                    require((sampled==resolved['target']).all() if sign=='positive' else
                        ((sampled!=resolved['target'])&(sampled!=255)&~np.isin(sampled,excluded)).all(),'heldout prompt leakage')
                entry['probes'].append(dict(plan=resolved,masked_labels=str(path),masked_sha256=source['labels_array_sha256'],
                    excluded_supervision_pixels=int(np.isin(labels,excluded).sum()),all_points_avoid_excluded=True))
                probe_strata[name].add(stratum)
        if index%100==0:
            write(output/'status.json',dict(status='running',worker_pid=os.getpid(),fields_audited=index+1,total_fields=len(rows)))
            print(json.dumps(dict(stage='labels',fields=index+1)),flush=True)
    result={}
    for name,spec in FOLDS.items():
        entry=evidence[name];require(entry['old_task_audit']['tasks']==40000,'all old task exposures audited')
        require(probe_strata[name]=={(d,s) for d in ('crc','tiger') for s in ('train','val')},'real two-domain/split prompt probes')
        write(output/f'{name}_candidates.json',entry['candidates']);write(output/f'{name}_probes.json',entry['probes'])
        result[name]=dict(definition=spec,excluded_label_names={d:{str(i):LABELS[d][i] for i in ids} for d,ids in spec['exclude'].items()},
            coverage={k:dict(v) for k,v in entry['coverage'].items()},old_task_exposure=dict(entry['old_task_audit']),
            candidates_sha256=digest(output/f'{name}_candidates.json'),probes_sha256=digest(output/f'{name}_probes.json'),
            real_prompt_probes=len(entry['probes']))
    write(output/'actual_field_counts.json',field_audit)
    write(output/'summary.json',dict(status='complete_feasibility_audit',worker_pid=os.getpid(),folds=result,
        catalog_sha256=contract['catalog_sha256'],source_sha256=digest(__file__),
        official_definition_url='https://tiger.grand-challenge.org/Data/',official_retrieval='primary-page search index; direct open403 on2026-10-26',
        stages='No heldout training executed; fold labels/tasks/model gates still required',
        strict_context_morphology_exclusion='not established; fine-clean counts are only an upper bound, not full4x absence proof',
        initializer_contract='Primary FewClick fresh generic COCO/noJ5/image-only; public model priors recorded, not all-pretraining unseen claim',
        selection_contract='Mask exclusions from training positive/negative/loss and seen-val selection; heldout query GT only terminal scoring',
        actual_field_counts_sha256=digest(output/'actual_field_counts.json')))
    text=['# 三概念留出监督覆盖审计','','基于预先定义本体的可行性统计；无任何heldout训练或精度结论。','',
        '| Fold | 域/划分 | 剩余候选图像目标对 | 有候选视野 | fine未出现排除类视野（非context严格池） |',
        '|---|---|---:|---:|---:|']
    for name,entry in result.items():
        for stratum,count in entry['coverage'].items():text.append(f'|{name}|{stratum}|{count["candidate_pairs"]}|{count["fields_with_candidates"]}|{count["fine_clean_fields"]}|')
    text += ['','TIGER3可能包含原位病灶坏死；rest7可含脂肪，TIGER2/6为相关stroma。详见[TIGER官方Data](https://tiger.grand-challenge.org/Data/)，直接网页抓取403，使用其主站搜索索引说明与本地标签映射。',
        '不能继续复用旧40k正负点：各fold原任务受影响数量见summary.json。seen验证选模也必须忽略排除类，不能以heldout像素的负类分数选模。',
        '原始图像可含排除形态，只能称无该类适配标签监督。完整4x无形态暴露尚未认证，禁止以fine无标签替代。',
        'L仍可能看到其他组织内的淋巴细胞，不证明没有免疫形态先验；所有公共模型预训练覆盖另记。']
    (output/'report.md').write_text('\n'.join(text)+'\n');write(output/'status.json',dict(status='complete_feasibility_audit',worker_pid=os.getpid(),fields_audited=len(rows)))
    print(json.dumps(dict(status='complete_feasibility_audit',folds=list(FOLDS),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--realized',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('output already exists')
    try:audit(a.tasks,a.realized,a.output)
    except Exception:
        if a.output.is_dir():write(a.output/'failure.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
