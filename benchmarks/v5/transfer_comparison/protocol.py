"""Freeze remote queries while reusing exactly the local support click sequences."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.paper_multiscale.data import intersects


KINDS=('same_present','cross_present','same_absent','cross_absent')


def build(local_tasks,output):
    output.mkdir(parents=True,exist_ok=False)
    contract=json.loads((local_tasks/'contract.json').read_text());catalog=Path(contract['catalog'])
    if digest(catalog/'tiles.parquet')!=contract['catalog_sha256'] or digest(local_tasks/'validation_maximum.json')!=contract['validation_sha256']:
        raise ValueError('local catalog/task identity changed')
    rows=pd.read_parquet(catalog/'tiles.parquet').to_dict('records');by_id={r['tile_id']:r for r in rows}
    val=[r for r in rows if r['split']=='val'];original=json.loads((local_tasks/'validation_maximum.json').read_text())
    supports=defaultdict(list)
    for plan in original:supports[plan['pair_id']].append(plan)
    counts={r['tile_id']:{int(k):int(v) for k,v in json.loads(r['counts_json']).items()} for r in val}
    boxes={r['tile_id']:json.loads(r['multiscale_json'])['context_um'] for r in val}
    plans=[];excluded=[];pair_number=40000000
    for identity,point_groups in sorted(supports.items()):
        point_groups=sorted(point_groups,key=lambda p:p['prompt_group']);base=point_groups[0]
        source=by_id[base['support_id']];target=base['target']
        for kind in KINDS:
            relation,presence=kind.split('_');candidates=[]
            for query in val:
                if query['dataset']!=source['dataset'] or query['tile_id']==source['tile_id']:continue
                positive=counts[query['tile_id']].get(target,0)
                if (positive<64 if presence=='present' else positive!=0):continue
                same_space=source['space_key']==query['space_key']
                if not source['slide_verified'] or not query['slide_verified']:continue
                if relation=='same':
                    if not same_space or intersects(boxes[source['tile_id']],boxes[query['tile_id']]):continue
                elif same_space:continue
                candidates.append(query)
            if not candidates:
                excluded.append(dict(support_pair_id=identity,support_id=source['tile_id'],dataset=source['dataset'],target=target,kind=kind,
                    reason='no verified full-context-separated query with declared target presence'))
                continue
            selected=min(candidates,key=lambda r:hashlib.sha256(f'transfer-v1|{identity}|{kind}|{r["tile_id"]}'.encode()).hexdigest())
            for group in point_groups:
                plans.append(group|dict(number=pair_number,support_pair_id=identity,pair_id=pair_number,
                    query_id=selected['tile_id'],kind=kind,protocol='same_wsi' if relation=='same' else 'cross_wsi',
                    query_organ=selected['organ'],query_presence=presence,candidate_count=len(candidates),
                    query_has_human_clicks=False,source_clicks_reused_from_local=True,
                    source_space=source['space_key'],query_space=selected['space_key'],
                    support_query_distance_um=float(np.linalg.norm(
                        (np.array([source['x0_um'],source['y0_um']])+np.array([source['x1_um'],source['y1_um']]))/2-
                        (np.array([selected['x0_um'],selected['y0_um']])+np.array([selected['x1_um'],selected['y1_um']]))/2)) if relation=='same' else None))
            pair_number+=1
    if not plans:raise ValueError('no feasible remote tasks')
    write(output/'validation_maximum.json',plans);write(output/'excluded.json',excluded)
    queries=Counter(p['query_id'] for p in plans if p['prompt_group']==0)
    report=dict(status='complete',version='support_only_clicks_transfer_v1_1025',catalog=str(catalog),
        catalog_sha256=digest(catalog/'tiles.parquet'),local_tasks=str(local_tasks),local_protocol_sha256=digest(local_tasks/'contract.json'),
        local_points_sha256=contract['validation_sha256'],tasks_sha256=digest(output/'validation_maximum.json'),
        tasks=len(plans),pairs=pair_number-40000000,point_groups=contract['validation_point_groups'],
        primary_budget='3+2',report_budgets=contract['report_budgets'],threshold=.5,
        pair_coverage=dict(Counter(p['dataset']+'|'+p['kind'] for p in plans if p['prompt_group']==0)),
        excluded=dict(Counter(p['dataset']+'|'+p['kind'] for p in excluded)),
        unique_support_fields=len({p['support_id'] for p in plans}),unique_query_fields=len(queries),
        repeated_query_pairs=sum(n-1 for n in queries.values()),max_pairs_per_query=max(queries.values()),
        source_query_full_context='same space 4x footprints disjoint; cross uses distinct verified WSI spaces',
        no_query_class_ID_in_inference=True,no_query_human_clicks=True,
        GT_roles='source GT generated frozen clicks; query GT defines benchmark presence and terminal scoring only',
        source_hashes={str(p):digest(p) for p in (Path(__file__),Path('benchmarks/v5/paper_multiscale/data.py'))},
        annotation_tracks={'sparse':'source positive/negative points only; no query points',
            'predicted_context':'same source points -> public model predicted source mask; full prior/compute ledger',
            'full_context_reference':'complete source mask; additional annotated area; separate table'},
        scope='within-cohort spatial development; queries repeat; not independent-patient or blind-test evidence')
    write(output/'contract.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


def verify(tasks,output):
    contract=json.loads((tasks/'contract.json').read_text());catalog=Path(contract['catalog'])
    if digest(catalog/'tiles.parquet')!=contract['catalog_sha256'] or digest(tasks/'validation_maximum.json')!=contract['tasks_sha256']:
        raise ValueError('transfer identity mismatch')
    rows=pd.read_parquet(catalog/'tiles.parquet').to_dict('records');by_id={r['tile_id']:r for r in rows}
    local={ (p['pair_id'],p['prompt_group']):p for p in json.loads((Path(contract['local_tasks'])/'validation_maximum.json').read_text())}
    plans=json.loads((tasks/'validation_maximum.json').read_text());cached={};identities=set();present=absent=0
    def labels(identity):
        if identity not in cached:
            row=by_id[identity];gt=np.asarray(Image.open(row['labels']))
            if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('GT identity mismatch')
            cached[identity]=gt
        return cached[identity]
    for p in plans:
        identity=p['pair_id'],p['prompt_group']
        if identity in identities:raise ValueError('duplicate transfer task')
        identities.add(identity);s=by_id[p['support_id']];q=by_id[p['query_id']]
        if s['split']!='val' or q['split']!='val' or p['support_id']==p['query_id']:raise ValueError('transfer split/identity mismatch')
        if not s['slide_verified'] or not q['slide_verified'] or s['dataset']!=q['dataset']:raise ValueError('source query provenance')
        same=s['space_key']==q['space_key']
        if same!=(p['protocol']=='same_wsi'):raise ValueError('WSI relation mismatch')
        if same and intersects(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um']):raise ValueError('source-query context leakage')
        original=local[p['support_pair_id'],p['prompt_group']]
        for field in ('support_id','target','positive_xy','negative_xy','click_seed','click_audit'):
            if p[field]!=original[field]:raise ValueError('source annotation changed: '+field)
        source=labels(p['support_id']);query=labels(p['query_id']);target=p['target'];mass=int(((query==target)&(query!=255)).sum())
        if p['query_presence']=='absent':
            if mass!=0:raise ValueError('absent query has target')
            absent+=1
        else:
            if mass<64:raise ValueError('present query insufficient target')
            present+=1
        for sign in ('positive','negative'):
            xy=np.asarray(p[sign+'_xy']);pixel=np.floor(xy*np.array(source.shape[::-1])).astype(int)
            if not ((xy>0)&(xy<1)).all():raise ValueError('support coordinate out of bounds')
            value=source[pixel[:,1],pixel[:,0]]
            if not (value==target if sign=='positive' else (value!=target)&(value!=255)).all():raise ValueError('support click sign')
    result=dict(status='passed',tasks=len(plans),present=present,absent=absent,contract_sha256=digest(tasks/'contract.json'),
        checks=['all support signs and source click equality','actual GT presence/absence','full physical input separation','distinct verified WSI relation','no duplicate task IDs'])
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local-tasks',type=Path);p.add_argument('--tasks',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if bool(a.local_tasks)==bool(a.tasks):raise ValueError('choose build or verify')
    build(a.local_tasks,a.output) if a.local_tasks else verify(a.tasks,a.output)
