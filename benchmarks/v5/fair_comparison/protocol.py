"""Freeze local-only task identities and model-independent representative clicks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit import digest, write
from .metrics import VERSION
from benchmarks.v5.multidomain_training.click_policy import representative_clicks, POLICY
from benchmarks.v5.multidomain_training.scoring import BACKGROUND_TARGETS


SEED = 20261025


def resolve_clicks(plan, row, maximum=(5,2)):
    gt = np.asarray(Image.open(row['labels']))
    import hashlib
    if hashlib.sha256(gt.tobytes()).hexdigest() != row['labels_array_sha256']:
        raise ValueError('GT identity changed')
    clicks, audit = representative_clicks(gt,plan['target'],json.loads(row['input_mpp_xy_json']),
                        *maximum,np.random.default_rng(plan['click_seed']),dataset=plan['dataset'])
    return plan | clicks | dict(click_audit=audit,prompt_policy=POLICY)


def nested(plan, budget):
    positive,negative=map(int,budget.split('+'))
    if positive>len(plan['positive_xy']) or negative>len(plan['negative_xy']):
        raise ValueError('budget exceeds frozen sequence')
    return plan | dict(positive_xy=plan['positive_xy'][:positive],negative_xy=plan['negative_xy'][:negative],click_budget=budget)


def build(catalog, output, groups=5, per_class=20, training_tasks=40000):
    output.mkdir(parents=True,exist_ok=False)
    rows=pd.read_parquet(catalog/'tiles.parquet').to_dict('records')
    pool=defaultdict(list); by_id={r['tile_id']:r for r in rows}
    for row in rows:
        for target in sorted(json.loads(row['eligible_json'])):
            if int(target) not in BACKGROUND_TARGETS[row['dataset']]:
                pool[row['split'],row['dataset'],int(target)].append(row)
    # Sample without inspecting predictions. Each (dataset,target) has a declared cap;
    # keep all rare pairs and use a stable hash order for more common pairs.
    val=[]; pair_number=30000000; coverage=[]
    for (split,dataset,target), candidates in sorted(pool.items()):
        if split!='val': continue
        ordered=sorted(candidates,key=lambda r:__import__('hashlib').sha256(
            f'local-fair-v1|{dataset}|{target}|{r["tile_id"]}'.encode()).hexdigest())
        selected=ordered[:per_class]
        coverage.append(dict(dataset=dataset,target=target,available=len(candidates),selected=len(selected)))
        for row in selected:
            base=dict(number=pair_number,support_id=row['tile_id'],query_id=row['tile_id'],dataset=dataset,target=target,
                kind='local',protocol='local',split='val',support_quality='original',query_quality='original',
                quality_condition='original',support_organ=row['organ'],query_organ=row['organ'],gt_simulated_clicks=True)
            for group in range(groups):
                plan=base | dict(pair_id=pair_number,prompt_group=group,click_seed=SEED+pair_number+group*1000000000)
                val.append(resolve_clicks(plan,row))
            pair_number+=1
            if (pair_number-30000000)%20==0:
                print(json.dumps(dict(stage='validation',pairs=pair_number-30000000,tasks=len(val))),flush=True)
    write(output/'validation_maximum.json',val)
    datasets=sorted({k[1] for k in pool if k[0]=='train'})
    policy=json.loads((catalog/'training_policy.json').read_text())
    probability=dict(zip(policy['datasets'],policy['probabilities']))
    rng=np.random.default_rng(SEED)
    # Freeze identity/seed recipe. Coordinates are resolved from audited support GT
    # by the shared resolver, then logged by every trainer. No model predictions enter.
    with (output/'training_recipe.jsonl').open('x') as handle:
        for number in range(training_tasks):
            dataset=str(rng.choice(datasets,p=[probability[d] for d in datasets]))
            classes=sorted(k[2] for k in pool if k[:2]==('train',dataset))
            target=int(rng.choice(classes)); candidates=sorted(pool['train',dataset,target],key=lambda r:r['tile_id'])
            row=candidates[int(rng.integers(len(candidates)))]
            plan=dict(number=number,support_id=row['tile_id'],query_id=row['tile_id'],dataset=dataset,target=target,
                split='train',kind='local',protocol='local',support_quality='original',query_quality='original',
                support_organ=row['organ'],query_organ=row['organ'],gt_simulated_clicks=True,
                positive_count=int(rng.integers(1,6)),negative_count=int(rng.integers(1,6)),
                click_seed=SEED+500000000+number)
            handle.write(json.dumps(plan,sort_keys=True)+'\n')
    contract=dict(version='local_fair_v1_1025',status='complete',catalog=str(catalog),catalog_sha256=digest(catalog/'tiles.parquet'),
        validation_sha256=digest(output/'validation_maximum.json'),training_recipe_sha256=digest(output/'training_recipe.jsonl'),
        source_hashes={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('metrics.py'),
            Path('benchmarks/v5/multidomain_training/click_policy.py'),Path('benchmarks/v5/multidomain_training/catalog.py')]},
        seed=SEED,metric_version=VERSION,point_policy=POLICY,point_geometry='normalized pixel centers; (x+.5)/width',
        selection_budget='3+2',selection_prompt_group=0,selection_metric='macro_pair_dice',
        report_budgets=['3+2','1+1','3+1','5+1'],validation_point_groups=groups,per_class_cap=per_class,
        coverage=coverage,validation_tasks=len(val),validation_pairs=pair_number-30000000,
        validation_unique_fields=len({p['query_id'] for p in val}),validation_fields_available=sum(r['split']=='val' for r in rows),
        training_tasks=training_tasks,training_steps=10000,accumulate=4,eval_every=1000,
        training_scope='one locally annotated image per task; same binary target supervision; no remote support auxiliary image',
        training_recipe_coordinates='resolve_clicks(recipe,row,(positive_count,negative_count)); log actual executed coordinates and SHA',
        threshold=.5,view_ledger={'SAM-public':['fine'],'WSI-SAM-public':['fine','2x'],'FewClick-M3':['fine','2x','4x']},
        validation_scope='within-cohort spatial development, used for checkpoint choice; not blind test',
        absent_target_scope='local positive-click tasks contain target; absent query belongs to separate transfer protocol',
        public_pretraining='different public priors; additional adaptation matched separately, no equal total pretraining claim',
        candidate_selection='SAM predicted IoU argmax; WSI-SAM single fused mask; no GT oracle',
        strict_external_claim_gate=['audited clean target initialization','same training recipe and realized coordinates',
            'same optimizer budget and validation schedule; method learning rates documented','full input and prior ledger'])
    write(output/'contract.json',contract)
    print(json.dumps(dict(status='complete',validation_tasks=len(val),pairs=contract['validation_pairs'],output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',type=int,default=5);p.add_argument('--per-class',type=int,default=20)
    p.add_argument('--training-tasks',type=int,default=40000)
    a=p.parse_args()
    if min(a.groups,a.per_class,a.training_tasks)<1:raise ValueError('invalid task counts')
    build(a.catalog,a.output,a.groups,a.per_class,a.training_tasks)
