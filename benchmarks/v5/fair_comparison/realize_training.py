"""Resolve shared training recipes once, before any method consumes them."""
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
import pandas as pd
from PIL import Image

from .audit import digest, write
from benchmarks.v5.multidomain_training.click_policy import representative_clicks, POLICY


def initialize(catalog):
    global ROWS
    ROWS={r['tile_id']:r for r in pd.read_parquet(Path(catalog)/'tiles.parquet').to_dict('records')}


@lru_cache(maxsize=64)
def labels(identity):
    row=ROWS[identity];gt=np.asarray(Image.open(row['labels']))
    if hashlib.sha256(gt.tobytes()).hexdigest()!=row['labels_array_sha256']:raise ValueError('GT identity changed')
    return gt


def realize(recipe):
    row=ROWS[recipe['support_id']]
    if row['split']!='train' or recipe['support_id']!=recipe['query_id']:raise ValueError('training protocol mismatch')
    clicks,audit=representative_clicks(labels(recipe['support_id']),recipe['target'],json.loads(row['input_mpp_xy_json']),
        recipe['positive_count'],recipe['negative_count'],np.random.default_rng(recipe['click_seed']),dataset=recipe['dataset'])
    return recipe | clicks | dict(click_audit=audit,prompt_policy=POLICY)


def run(tasks,output,workers=8,limit=None):
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    contract=json.loads((tasks/'contract.json').read_text());catalog=Path(contract['catalog'])
    if digest(catalog/'tiles.parquet')!=contract['catalog_sha256'] or digest(tasks/'training_recipe.jsonl')!=contract['training_recipe_sha256']:
        raise ValueError('manifest identity mismatch')
    recipes=[json.loads(line) for line in (tasks/'training_recipe.jsonl').read_text().splitlines()]
    if limit:recipes=recipes[:limit]
    write(output/'run.json',dict(worker_pid=os.getpid(),workers=workers,protocol=str(tasks),limit=limit,
        protocol_sha256=digest(tasks/'contract.json'),sources={str(p):digest(p) for p in (
            Path(__file__),Path('benchmarks/v5/multidomain_training/click_policy.py'),Path('benchmarks/v5/multidomain_training/catalog.py'))}))
    counts=Counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(catalog),)) as pool:
        with (output/'training_plans.jsonl').open('x') as handle:
            for i,plan in enumerate(pool.map(realize,recipes,chunksize=8),1):
                handle.write(json.dumps(plan,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');counts[plan['dataset']]+=1
                if i%100==0:
                    handle.flush();record=dict(status='running',worker_pid=os.getpid(),completed=i,total=len(recipes),seconds=time.time()-started)
                    write(output/'status.json',record);print(json.dumps(record),flush=True)
    # Independent direct resolver replay, rather than trusting only pool completion.
    from .protocol import resolve_clicks
    initialize(str(catalog));actual=[json.loads(line) for line in (output/'training_plans.jsonl').read_text().splitlines()]
    for index in sorted(set([0,len(actual)//2,len(actual)-1])):
        recipe=recipes[index];expected=resolve_clicks(recipe,ROWS[recipe['support_id']],(recipe['positive_count'],recipe['negative_count']))
        if actual[index]!=expected:raise ValueError('independent click replay mismatch')
    report=dict(status='complete',worker_pid=os.getpid(),tasks=len(actual),counts=dict(counts),
        complete_protocol=not limit or limit>=contract['training_tasks'],protocol=str(tasks),
        protocol_sha256=digest(tasks/'contract.json'),training_plans_sha256=digest(output/'training_plans.jsonl'),
        recipe_sha256=contract['training_recipe_sha256'],direct_replay_passed=True,seconds=time.time()-started)
    write(output/'summary.json',report);write(output/'status.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);a=p.parse_args()
    run(a.tasks,a.output,a.workers,a.limit)
