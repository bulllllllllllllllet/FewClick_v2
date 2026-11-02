"""Realize train-only local clicks with the established representative policy."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.multidomain_training.click_policy import representative_clicks,POLICY


def initialize():
    global ROWS
    _,ROWS=load_cell_catalog(TASKS)


@lru_cache(maxsize=32)
def labels(identity):
    row=ROWS[identity];assert row['split']=='train'
    with Image.open(row['labels']) as im:gt=np.asarray(im).copy()
    assert hashlib.sha256(gt.tobytes()).hexdigest()==row['labels_array_sha256'];return gt


def realize(recipe):
    row=ROWS[recipe['support_id']];assert recipe['query_id']==recipe['support_id'] and row['split']=='train'
    clicks,audit=representative_clicks(labels(recipe['support_id']),recipe['target'],json.loads(row['input_mpp_xy_json']),
        recipe['positive_count'],recipe['negative_count'],np.random.default_rng(recipe['click_seed']),dataset=recipe['dataset'])
    return recipe|clicks|dict(click_audit=audit,prompt_policy=POLICY,needs_click_realization=False,gt_simulated_clicks=True,
        query_has_human_clicks=True,training_recipe='P3_C_balanced_local20261105',support_quality='original',query_quality='original',support_organ=row['organ'],query_organ=row['organ'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--gate-receipt',type=Path);a=p.parse_args()
    proof=json.loads((a.prepared/'report.json').read_text());assert proof['local_recipe_sha256']==sha(a.prepared/'local_recipe.jsonl')
    recipes=list(map(json.loads,(a.prepared/'local_recipe.jsonl').read_text().splitlines()));assert len(recipes)==4000
    if a.gate:recipes=[next(r for r in recipes if r['dataset']==d and r['target']==t) for d,t in [('crc',0),('crc',10),('tiger',1),('tiger',3)]]
    identity=dict(source_sha256=sha(__file__),recipe_sha256=proof['local_recipe_sha256'],prepared_report_sha256=sha(a.prepared/'report.json'))
    if not a.gate:
        gate=json.loads(a.gate_receipt.read_text());assert gate['status']=='accepted_local_realization_gate' and gate['identity']==identity
    a.output.mkdir(exist_ok=False);started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool,(a.output/'training.jsonl').open('w') as stream:
        for value in pool.map(realize,recipes,chunksize=8):
            stream.write(json.dumps(value,sort_keys=True)+'\n');rows.append(value)
            if len(rows)%25==0:stream.flush();(a.output/'status.json').write_text(json.dumps(dict(status='running',completed=len(rows),total=len(recipes),worker_pid=os.getpid(),seconds=time.monotonic()-started)))
    initialize()
    for row in rows:
        gt=labels(row['support_id']);assert row['positive_count']==len(row['positive_xy']) and row['negative_count']==len(row['negative_xy'])
        for sign in ['positive','negative']:
            for x,y in row[sign+'_xy']:
                assert 0<x<1 and 0<y<1;label=gt[int(y*gt.shape[0]),int(x*gt.shape[1])]
                assert label!=255 and ((label==row['target'])==(sign=='positive'))
    # Replay the independent public protocol resolver on fixed endpoints/middle.
    from benchmarks.v5.fair_comparison.protocol import resolve_clicks
    for i in sorted({0,len(rows)//2,len(rows)-1}):
        direct=resolve_clicks(recipes[i],ROWS[recipes[i]['support_id']],(recipes[i]['positive_count'],recipes[i]['negative_count']))
        for key in ['positive_xy','negative_xy','click_audit','prompt_policy']:assert rows[i][key]==direct[key]
    assert Counter((r['dataset'],r['target'],r['positive_count'],r['negative_count']) for r in rows)==Counter((r['dataset'],r['target'],r['positive_count'],r['negative_count']) for r in recipes)
    report=dict(status='accepted_local_realization_gate' if a.gate else 'accepted_local4000_clicks_preparation',identity=identity,tasks=len(rows),
                training_sha256=sha(a.output/'training.jsonl'),all_click_signs_checked=True,independent_resolver_replay=True,val_GT_read=False,
                training_allowed=False,remaining='final architecture frozen; P3-C execution/supervision scope contract and actual model learning gate required')
    (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'status.json').write_text(json.dumps(dict(status='complete',tasks=len(rows))))


if __name__=='__main__':main()
