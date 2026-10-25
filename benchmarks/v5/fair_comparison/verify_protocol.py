"""Exercise frozen task geometry and deterministic training click resolution."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .audit import digest, write
from .protocol import nested, resolve_clicks


def verify(tasks, output):
    contract=json.loads((tasks/'contract.json').read_text());catalog=Path(contract['catalog'])
    for path,expected in [(catalog/'tiles.parquet',contract['catalog_sha256']),
            (tasks/'validation_maximum.json',contract['validation_sha256']),
            (tasks/'training_recipe.jsonl',contract['training_recipe_sha256'])]:
        if digest(path)!=expected:raise ValueError('manifest SHA mismatch')
    by_id={r['tile_id']:r for r in pd.read_parquet(catalog/'tiles.parquet').to_dict('records')}
    plans=json.loads((tasks/'validation_maximum.json').read_text()); identities=set()
    for p in plans:
        if (p['support_id']!=p['query_id'] or p['split']!='val' or p['kind']!='local'
                or by_id[p['query_id']]['split']!='val'):raise ValueError('local/split mismatch')
        identity=p['pair_id'],p['prompt_group']
        if identity in identities:raise ValueError('duplicate task')
        identities.add(identity);gt=np.asarray(Image.open(by_id[p['query_id']]['labels']))
        maximum=np.array(gt.shape[::-1])
        for sign in ['positive','negative']:
            xy=np.asarray(p[sign+'_xy']);pixel=np.floor(xy*maximum).astype(int)
            if not ((xy>0)&(xy<1)).all():raise ValueError('point bounds')
            code=gt[pixel[:,1],pixel[:,0]]
            good=code==p['target'] if sign=='positive' else (code!=p['target'])&(code!=255)
            if not good.all():raise ValueError('click sign mismatch')
        for budget in contract['report_budgets']:
            cut=nested(p,budget);npositive,nnegative=map(int,budget.split('+'))
            if cut['positive_xy']!=p['positive_xy'][:npositive] or cut['negative_xy']!=p['negative_xy'][:nnegative]:raise ValueError('prefix mismatch')
    recipes=[json.loads(line) for line in (tasks/'training_recipe.jsonl').read_text().splitlines()]
    if len(recipes)!=contract['training_tasks']:raise ValueError('training recipe count')
    examples=[]
    for dataset in sorted({p['dataset'] for p in recipes}):
        p=next(p for p in recipes if p['dataset']==dataset);row=by_id[p['query_id']]
        if row['split']!='train' or p['support_id']!=p['query_id']:raise ValueError('training split')
        maximum=p['positive_count'],p['negative_count']
        a=resolve_clicks(p,row,maximum);b=resolve_clicks(p,row,maximum)
        if a!=b:raise ValueError('training click replay mismatch')
        examples.append(a)
    report=dict(status='passed',validation_tasks=len(plans),training_recipes=len(recipes),
        training_replay_examples=examples,protocol_sha256=digest(tasks/'contract.json'),
        limits='representative training click replay only; complete trainer-coordinate SHA remains a G3 gate')
    write(output,report);print(json.dumps({k:v for k,v in report.items() if k!='training_replay_examples'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.tasks,a.output)
