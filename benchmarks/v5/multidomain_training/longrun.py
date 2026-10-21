"""Explicit long-run learning rates, frozen validation expansion and coverage."""
import math
from collections import defaultdict

from .episodes import DATASETS,KINDS


def learning_rates(step,total,joint_start,ramp=200,minimum=.1,schedule='constant'):
    if not 0<=step<total or not 0<=joint_start<=total or ramp<1 or not 0<minimum<=1:raise ValueError('invalid learning-rate schedule')
    if schedule=='constant':return dict(heads=1e-4,backbone=1e-5)
    if schedule!='warmup_cosine':raise ValueError('unknown learning-rate schedule')
    progress=max(0.,(step-joint_start)/max(1,total-joint_start-1))
    decay=minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))
    head=min(1.,(step+1)/ramp)*decay
    backbone=0. if step<joint_start else min(1.,(step-joint_start+1)/ramp)*decay
    return dict(heads=1e-4*head,backbone=1e-5*backbone)


def extend_validation(corpus,anchors,extra_per_stratum=20):
    if not 1<=extra_per_stratum<100:raise ValueError('extra repetitions must be in [1,99]')
    plans=[dict(p) for p in anchors];variants=[('original','0.8'),('0.8','original'),('1.0','0.8'),('0.8','1.0')]
    for d,dataset in enumerate(DATASETS):
        for k,kind in enumerate(KINDS):
            for repeat in range(extra_per_stratum):
                number=20000000+d*10000+k*100+repeat
                p=corpus.plan(number,split='val',dataset=dataset,kind=kind,quality_aug=False,fixed_clicks=True)
                pair=f'extended-{dataset}-{kind}-{repeat}'
                plans.append(p|dict(quality_condition='original',pair_id=pair))
                sq,qq=variants[repeat%len(variants)]
                plans.append(p|dict(support_quality=sq,query_quality=qq,quality_condition='degraded',pair_id=pair))
    return plans


class Coverage:
    def __init__(self,corpus,datasets=None):
        self.pool=defaultdict(set);self.support_pool=defaultdict(set)
        self.seen={role:defaultdict(set) for role in ['support','query','query_variant']}
        for row in corpus.rows:
            if row['split']!='train':continue
            if datasets is not None and row['dataset'] not in datasets:continue
            self.pool[row['dataset']].add(row['tile_id'])
            if corpus.eligible[corpus.index[row['tile_id']]]:self.support_pool[row['dataset']].add(row['tile_id'])

    def observe(self,plan):
        if plan['split']!='train':raise ValueError('validation cannot count as training coverage')
        dataset=plan['dataset']
        if plan['support_id'] not in self.support_pool[dataset] or plan['query_id'] not in self.pool[dataset]:raise ValueError('episode outside training pool')
        self.seen['support'][dataset].add(plan['support_id']);self.seen['query'][dataset].add(plan['query_id'])
        self.seen['query_variant'][dataset].add((plan['query_id'],plan['query_quality']))

    def summary(self):
        rows={d:dict(query_pool_tiles=len(pool),support_eligible_tiles=len(self.support_pool[d]),
            seen_query_tiles=len(self.seen['query'][d]),seen_support_tiles=len(self.seen['support'][d]),
            seen_query_variants=len(self.seen['query_variant'][d]),query_tile_fraction=len(self.seen['query'][d])/len(pool)) for d,pool in self.pool.items()}
        return dict(sampling='balanced sampling with replacement; no complete-epoch guarantee',by_dataset=rows,
            query_pool_tiles=sum(len(v) for v in self.pool.values()),seen_query_tiles=sum(len(v) for v in self.seen['query'].values()),
            seen_support_tiles=sum(len(v) for v in self.seen['support'].values()))
