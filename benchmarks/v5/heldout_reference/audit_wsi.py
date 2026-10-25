"""Independent full-row/GT/macro and saved-mask audit of all-label WSI references."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.transfer_comparison.data import TransferData


def read(path):return json.loads(Path(path).read_text())


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);summary=read(a.root/'summary.json');manifest=read(a.protocols/'manifest.json')
    if summary['status']!='complete_all_label_WSI_remote_execution':raise ValueError('all3 references must finish execution')
    checks=[]
    for fold,item in manifest.items():
        tasks=Path(item['heldout_remote']);data=TransferData(tasks);expected={}
        for p in data.plans:
            for budget in data.contract['report_budgets']:
                row=nested(p,budget);expected[row['pair_id'],row['prompt_group'],budget]=row
        for stage in ('gate','score'):
            folder=Path(summary['complete'][fold+'|'+stage]);done=read(folder/'summary.json');meta=read(folder/'run.json')
            if not done['all_label_reference'] or done['checkpoint_gate'] or not meta['all_label_reference']:
                raise ValueError('reference exposure annotation missing')
            for path,sha in meta['source_sha256'].items():
                if digest(path)!=sha:raise ValueError('reference inference source changed')
            if digest(meta['checkpoint'])!=meta['checkpoint_sha256'] or meta['protocol_sha256']!=digest(tasks/'contract.json'):
                raise ValueError('weights or frozen test changed')
            coverage=read(folder/'query_annotation_coverage.json')
            for query,value in coverage.items():
                gt=data.labels(query)
                if value!={'valid_pixels':int((gt!=255).sum()),'total_pixels':int(gt.size)}:raise ValueError('query annotation coverage schema/value')
            for output in done['outputs'].values():
                output=Path(output);stats=read(output/'summary.json');rows=[json.loads(line) for line in (output/'rows.jsonl').open()]
                if digest(output/'rows.jsonl')!=stats['rows_sha256']:raise ValueError('output row SHA changed')
                keys=set()
                for index,row in enumerate(rows):
                    key=(row['pair_id'],row['prompt_group'],row['click_budget'])
                    if key in keys or key not in expected:raise ValueError('duplicate/outside frozen task')
                    keys.add(key)
                    if not all(row.get(k)==v for k,v in expected[key].items()):raise ValueError('point or task identity mismatch')
                    truth=data.supervision(row);positive=int(truth['truth'].sum());valid=int(truth['valid'].sum())
                    if row['positive']!=positive or row['valid']!=valid or row['tp']+row['fn']!=positive or row['tp']+row['fp']>valid:
                        raise ValueError('actual GT/count denominators')
                    if positive and abs(row['dice']-2*row['tp']/(2*row['tp']+row['fp']+row['fn']))>1e-12:raise ValueError('Dice count algebra')
                    if not positive and abs(row['false_positive_fraction']-row['fp']/valid)>1e-12:raise ValueError('absence valid-area FPA')
                    if stage=='gate':
                        p=np.load(output/f'prediction_{index:04d}.npy',allow_pickle=False)>=.5;t=truth['truth'];v=truth['valid']
                        if (int((p&t&v).sum()),int((p&~t&v).sum()),int((~p&t&v).sum()))!=(row['tp'],row['fp'],row['fn']):
                            raise ValueError('saved probability pixel recount')
                if stage=='score' and keys!=set(expected):raise ValueError('full fixed task coverage')
                if stage=='gate' and (any(done['native_cached_errors']) or not done['saved_count_replay']):raise ValueError('native/cache gate evidence')
                for budget in data.contract['report_budgets']:
                    selected=[r for r in rows if r['click_budget']==budget];pairs=defaultdict(list)
                    for row in selected:
                        if row['positive']:pairs[row['pair_id']].append(row['dice'])
                    mean=math.fsum(math.fsum(v)/len(v) for v in pairs.values())/len(pairs)
                    negative=[r['false_positive_fraction'] for r in selected if not r['positive']]
                    fpa=math.fsum(negative)/len(negative);reported=stats['metrics'][budget]['overall']
                    if abs(mean-reported['present_dice'])>1e-12 or abs(fpa-reported['absent_FPA'])>1e-12:raise ValueError('independent macro/FPA recount')
                checks.append(dict(fold=fold,stage=stage,rows=len(rows),full_fixed_task_match=stage=='score',saved_pixel_replay=stage=='gate',
                    checkpoint_sha256=meta['checkpoint_sha256'],selected_weight_role='All available-label G3 local development reference',
                    primary=stats['metrics']['3+2']['overall']))
    write(a.output/'summary.json',dict(status='passed_all_label_WSI_reference_audit',checks=checks,source_sha256=digest(__file__),
        full_reference_rows=sum(x['rows'] for x in checks if x['stage']=='score'),gate_saved_masks=sum(x['rows'] for x in checks if x['stage']=='gate'),
        scope='Full-label trained/selected references accepted; newly excluded-label trained counterparts still running; no isolated causal deletion claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','protocols','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
