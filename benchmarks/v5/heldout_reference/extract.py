"""Extract accepted full-label rows with exact frozen task/point identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.public_baselines import summarize as local_summary
from benchmarks.v5.transfer_comparison.evaluate_sparse import summarize as remote_summary


def read(path):return json.loads(Path(path).read_text())


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    accepted=read(a.local_root/'G3_complete_audit_1025_155140.json');raccepted=read(a.remote_root/'acceptance_1025_224800/acceptance.json')
    if accepted['status']!='passed' or raccepted['status']!='passed':raise ValueError('accepted G3/G4 evidence required')
    original=a.local_root/'adaptation_1025_131451';manifest=read(a.protocols/'manifest.json');records=[];catalog=None;gt_cache={}
    def labels(identity):
        if identity not in gt_cache:
            row=catalog[identity];gt=np.asarray(Image.open(row['labels']))
            if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('original GT changed')
            gt_cache[identity]=gt
        return gt_cache[identity]
    for mode in ('local','remote'):
        parents=[]
        for method in ('sam_ft','wsi_sam_ft','fewclick_m3d_image'):
            training=original/f'train_{method}_1025_131451';done=read(training/'summary.json');best=read(training/'best_checkpoint.json')
            if (done['status']!='complete' or done['step']!=10000 or done['training_consumed_sha256']!='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe' or
                digest(training/'training_consumed.jsonl')!=done['training_consumed_sha256'] or best!=accepted['methods'][method]['selected_checkpoint'] or digest(best['path'])!=best['checkpoint_sha256']):
                raise ValueError('accepted all-label actual training/selected weight identity')
            if mode=='local':parents.append((method,original/f'evaluate_{method}_1025_131451',best))
            elif method!='wsi_sam_ft':
                if raccepted['selected_local_weights'][method]['checkpoint_sha256']!=best['checkpoint_sha256']:raise ValueError('accepted remote weights')
                root=a.remote_root/f'clean_{method}_1025_162450'
                for path in sorted(root.glob('*/rows.jsonl')):parents.append((path.parent.name,path.parent,best))
        for fold,item in manifest.items():
            tasks=Path(item['heldout_'+mode]);contract=read(tasks/'contract.json')
            if catalog is None:catalog={r['tile_id']:r for r in pd.read_parquet(Path(contract['catalog'])/'tiles.parquet').to_dict('records')}
            maximum=read(tasks/'validation_maximum.json');wanted={}
            for p in maximum:
                for budget in contract['report_budgets']:
                    plan=nested(p,budget);key=(plan['pair_id'],plan['prompt_group'],budget)
                    if key in wanted:raise ValueError('duplicate requested task')
                    wanted[key]=plan
            for method,parent,best in parents:
                source=parent/'rows.jsonl';meta=read(parent/'run.json');summary=read(parent/'summary.json')
                if summary['status']!='complete' or summary['gate'] or digest(source)!=summary['rows_sha256'] or meta['checkpoint_sha256']!=best['checkpoint_sha256']:
                    raise ValueError('accepted parent score identity')
                target=a.output/f'{mode}_{fold}_{method}';target.mkdir();selected=[];seen=set();parent_rows=0
                with (target/'rows.jsonl').open('x') as stream:
                    for line in source.open():
                        parent_rows+=1;row=json.loads(line);key=(row['pair_id'],row['prompt_group'],row['click_budget'])
                        if key not in wanted:continue
                        if key in seen:raise ValueError('duplicate matching parent row')
                        seen.add(key);plan=wanted[key]
                        if not all(row.get(k)==v for k,v in plan.items()):raise ValueError('original frozen task/point mismatch')
                        gt=labels(row['query_id']);positive=int((gt==row['target']).sum());valid=int((gt!=255).sum())
                        if row['positive']!=positive or row['valid']!=valid or row['tp']+row['fn']!=positive:raise ValueError('actual GT denominators')
                        if positive and abs(row['dice']-2*row['tp']/(2*row['tp']+row['fp']+row['fn']))>1e-12:raise ValueError('Dice algebra')
                        selected.append(row);stream.write(line)
                if parent_rows!=(6040 if mode=='local' else 18700) or seen!=set(wanted):raise ValueError('complete parent/subset row coverage')
                info=dict(mode=mode,fold=fold,method=method,exposure='all-label training AND all-target local development checkpoint selection',
                    parent_rows=str(source),parent_rows_sha256=digest(source),protocol_sha256=digest(tasks/'contract.json'),
                    selected_checkpoint_sha256=best['checkpoint_sha256'],selected_step=best['step'],trajectory_updates=10000,trajectory_tasks=40000,
                    scope='Identical test images/clicks/metrics; different target supervision, task distribution and checkpoint-selection labels; descriptive reference, not isolated causal effect')
                write(target/'run.json',info)
                result=dict(status='complete_extracted_reference',rows=len(selected),rows_sha256=digest(target/'rows.jsonl'),
                    metrics=(local_summary if mode=='local' else remote_summary)(selected))
                write(target/'summary.json',result);records.append(info|dict(output=str(target),rows=len(selected),metrics=result['metrics']))
                print(json.dumps(dict(mode=mode,fold=fold,method=method,rows=len(selected))),flush=True)
    write(a.output/'summary.json',dict(status='complete_all_label_reference_extraction',worker_pid=os.getpid(),records=records,
        source_sha256=digest(__file__),scope='Previously accepted G3/G4 rows, not new heldout training results',
        remaining='New all-label WSI-SAM custom two-view remote prototype; actual heldout trained counterparts and independent paired statistics'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('local-root','remote-root','protocols','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
