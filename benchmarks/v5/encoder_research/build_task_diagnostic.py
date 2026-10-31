"""Freeze train-only50/25/25 task factors without changing source prompts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_tissue_contract_1031_011514/pilot_training4000.jsonl')
    catalog=Path('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')
    original=[json.loads(x) for x in source.read_text().splitlines()];all_rows=pd.read_parquet(catalog).to_dict('records')
    rows={r['tile_id']:r for r in all_rows};pools={}
    for dataset in ['crc','tiger']:pools[dataset]=sorted([r for r in all_rows if r['dataset']==dataset and r['split']=='train'],key=lambda r:r['tile_id'])
    counts={r['tile_id']:json.loads(r['counts_json']) for r in all_rows}
    boxes={r['tile_id']:np.array(json.loads(r['multiscale_json'])['context_um'],float) for r in all_rows}
    mixed=[];audit=[]
    for i,plan in enumerate(original):
        assert plan['kind']=='local' and plan['split']=='train' and rows[plan['query_id']]['split']=='train'
        if i%2==0:mixed.append(plan);continue
        source_row=rows[plan['query_id']];present=i%4==1;target=str(plan['target']);candidates=[]
        for q in pools[plan['dataset']]:
            n=counts[q['tile_id']].get(target,0)
            valid=sum(v for k,v in counts[q['tile_id']].items() if k!='255')
            if q['tile_id']==source_row['tile_id'] or valid<64 or (n<64 if present else n!=0):continue
            if q['space_key']==source_row['space_key']:
                x,y=boxes[q['tile_id']],boxes[source_row['tile_id']]
                if min(x[2],y[2])>max(x[0],y[0]) and min(x[3],y[3])>max(x[1],y[1]):continue
            candidates.append(q)
        if not candidates:
            (a.output/'failure.json').write_text(json.dumps(dict(status='insufficient_disjoint_train_pairs',task=i,dataset=plan['dataset'],target=plan['target'],present=present))+'\n')
            raise ValueError('No eligible train query; do not use validation or overlapping fallback')
        seed=int(hashlib.sha256(f'20261031|{i}|{plan["query_id"]}'.encode()).hexdigest()[:16],16)
        q=candidates[seed%len(candidates)]
        kind=('same_' if q['space_key']==source_row['space_key'] else 'cross_')+('present' if present else 'absent')
        task=plan|dict(query_id=q['tile_id'],kind=kind,query_organ=q['organ'],query_presence='present' if present else 'absent',
            query_has_human_clicks=False,source_space=source_row['space_key'],query_space=q['space_key'],diagnostic_parent_task=i,
            source_GT_role='frozen original support clicks only',query_GT_role='train target for loss, not forward or query prompts')
        mixed.append(task);audit.append(dict(task=i,support=plan['support_id'],query=q['tile_id'],
            dataset=plan['dataset'],target=plan['target'],presence=task['query_presence'],eligible_queries=len(candidates),
            query_annotation_sha=q['labels_array_sha256'],source_annotation_sha=source_row['labels_array_sha256']))
    for name,plans in [('local_control',original),('mixed',mixed)]:
        with (a.output/f'{name}.jsonl').open('w') as f:
            for plan in plans:f.write(json.dumps(plan,sort_keys=True)+'\n')
    assert sha(a.output/'local_control.jsonl')==sha(source)
    report=dict(status='complete_frozen_training_task_factors_unaccepted',source_sha256=sha(__file__),original_SHA=sha(source),
        catalog_sha256=sha(catalog),tasks=4000,local=2000,remote_present=1000,remote_absent=1000,
        kinds=dict(Counter(p['kind'] for p in mixed)),mixed_sha256=sha(a.output/'mixed.jsonl'),
        control_sha256=sha(a.output/'local_control.jsonl'),pair_audit=audit,
        inference_GT_policy='No query prompts/GT/class/presence into model; training GT only episode stratification and terminal loss',
        fairness='same start model+optimizer, incremental updates/loss count and annotated pool; remote2-view and target-area consumption differ and must be disclosed')
    (a.output/'contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='pair_audit'}))


if __name__=='__main__':main()
