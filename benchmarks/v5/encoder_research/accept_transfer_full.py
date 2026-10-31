"""All2805 mask replay, target coverage and conditional source-space intervals."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.transfer_comparison.data import TransferData


def summarize(group):
    present=[r for r in group if r['positive']>0];absent=[r for r in group if not r['positive']]
    mean=lambda v:float(np.mean(v)) if v else None
    return dict(pairs=len(group),present=len(present),absent=len(absent),
        unique_query=len({r['query_id'] for r in group}),unique_support=len({r['support_id'] for r in group}),
        dice=mean([r['dice'] for r in present]),BF1=mean([r['bf1_um2'] for r in present if r['bf1_um2'] is not None]),
        absent_FPA=mean([r['false_positive_fraction'] for r in absent]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.evaluation/'report.json').read_text());assert report['status']=='complete_frozen935_transfer_unaccepted'
    data=TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    plans=[p for p in data.plans if p['prompt_group']==0];rows=json.loads((a.evaluation/'rows.json').read_text())
    expected=[(b,p['pair_id'],p['support_id'],p['query_id'],p['target']) for b in ['3+2','1+1','5+1'] for p in plans]
    assert [(r['budget'],r['pair_id'],r['support_id'],r['query_id'],r['target']) for r in rows]==expected and len(rows)==2805
    @lru_cache(maxsize=64)
    def labels(identity):return data.labels(identity)
    with np.load(a.evaluation/'packed_masks.npz',allow_pickle=False) as f:masks=f['masks'].copy()
    assert masks.shape==(2805,32768)
    for i,row in enumerate(rows):
        assert row['mask_index']==i
        gt=labels(row['query_id']);mpp=json.loads(data.rows[row['query_id']]['input_mpp_xy_json'])
        pred=np.unpackbits(masks[i],count=512*512).reshape(512,512).astype(bool)
        values=score(pred,gt==row['target'],gt!=255,mpp)
        for key,value in values.items():
            if value is None:assert row[key] is None
            elif isinstance(value,(int,bool)):assert row[key]==value
            else:assert abs(row[key]-value)<1e-12
        if (i+1)%200==0:(a.evaluation/'replay_status.json').write_text(json.dumps(dict(status='running',replayed=i+1,total=2805))+'\n')
    by_budget={};by_target={};uncertainty={}
    for budget in ['3+2','1+1','5+1']:
        group=[r for r in rows if r['budget']==budget];by_budget[budget]=summarize(group)
        targets=defaultdict(list)
        for r in group:targets[(r['dataset'],r['target'])].append(r)
        by_target[budget]={d+'|'+str(t):summarize(g) for (d,t),g in sorted(targets.items())}
        means=defaultdict(list)
        for r in group:means[(r['query_id'],r['target'],r['positive']>0)].append(r)
        qdice=[np.mean([r['dice'] for r in g]) for (q,t,present),g in means.items() if present]
        qfpa=[np.mean([r['false_positive_fraction'] for r in g]) for (q,t,present),g in means.items() if not present]
        by_budget[budget]['query_target_averaged_Dice']=float(np.mean(qdice))
        by_budget[budget]['query_target_averaged_absent_FPA']=float(np.mean(qfpa))
        by_budget[budget]['class_macro_Dice']=float(np.mean([v['dice'] for v in by_target[budget].values() if v['dice'] is not None]))
        by_budget[budget]['class_macro_absent_FPA']=float(np.mean([v['absent_FPA'] for v in by_target[budget].values() if v['absent_FPA'] is not None]))
    for metric,present in [('dice',True),('false_positive_fraction',False)]:
        groups=defaultdict(list)
        for r in rows:
            if r['budget']=='3+2' and (r['positive']>0)==present:groups[data.rows[r['query_id']]['space_key']].append(r[metric])
        sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
        interval=None
        if len(groups)>=2:
            sampled=np.random.default_rng(20261031).integers(0,len(groups),(2000,len(groups)))
            estimates=sums[sampled].sum(1)/counts[sampled].sum(1);interval=np.quantile(estimates,[.025,.975]).tolist()
        uncertainty[metric]={'query_space_clusters':len(groups),'CI95':interval,'scope':'conditional on fixed support bank; exploratory single-seed development interval, support reuse/same-patient dependence not fully resolved'}
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for plan in plans:
        s,q=(data.rows[plan[k]]['space_key'] for k in ['support_id','query_id']);parent[find(s)]=find(q)
    components=len({find(x) for x in list(parent)})
    result=dict(status='accepted_full935_transfer',rows_replayed=2805,pairs=935,by_budget=by_budget,by_target=by_target,
        primary_conditional_uncertainty=uncertainty,support_query_space_graph_components=components,
        reference240_masks_exact=report['exact_reference_mask_equality'] and report['reference_subset_rows_reproduced']==240,
        checkpoint_sha256=report['checkpoint_sha256'],frozen_tasks_sha256=report['frozen_tasks_sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_steps=0,
        limitations=['previously exposed development set, not blind/external validation','935pairs share330queries and212supports, not independent patients','rare-class present/absent denominators reported separately; NA does not mean zero error','old80 selection/guards retained, full-cohort means have different target mixture'])
    (a.evaluation/'independent_acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.evaluation/'replay_status.json').write_text(json.dumps(dict(status='complete',replayed=2805))+'\n');print(json.dumps(dict(status=result['status'],primary=by_budget['3+2'],graph_components=components)),flush=True)


if __name__=='__main__':main()
