"""Freeze exact P3 consumption and evaluation plans without reading query labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .repaired_model import RepairedData
from .spatial_controls import ARMS
from .data import BASE
from benchmarks.v5.encoder_research.continuous_data import ContinuousData,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=RepairedData(a.cache);source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl'
    raw=source.read_bytes();lines=raw.decode().splitlines();assert len(lines)==4000
    training=list(map(json.loads,lines))
    for task in training:
        assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
        for sign in ['positive_xy','negative_xy']:
            assert len(task[sign]) and all(0<x<1 and 0<y<1 for x,y in task[sign])
    evaluation=[dict(scope=s,budget=b,plan=t) for s,b,t in ContinuousData.validation(data)]
    counts=Counter((x['scope'],x['budget']) for x in evaluation)
    assert counts==Counter({(s,b):n for s,n in [('local',302),('remote',935)] for b in ['3+2','1+1','5+1']})
    for item in evaluation:
        task=item['plan'];assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='val'
        assert (task['support_id']==task['query_id'])==(item['scope']=='local')
        positive,negative=map(int,item['budget'].split('+'))
        assert len(task['positive_xy'])==positive and len(task['negative_xy'])==negative
    a.output.mkdir(exist_ok=False)
    (a.output/'training_consumption.jsonl').write_bytes(raw*2)
    (a.output/'evaluation.jsonl').write_text(''.join(json.dumps(t,sort_keys=True)+'\n' for t in evaluation))
    assert len((a.output/'training_consumption.jsonl').read_text().splitlines())==8000
    assert list(map(json.loads,(a.output/'evaluation.jsonl').read_text().splitlines()))==evaluation
    ledger=dict(source_tasks=4000,consumed=8000,task_kind_counts_source=dict(Counter(t['dataset']+'|'+t['kind'] for t in training)),
                unique_support_images=len({t['support_id'] for t in training}),unique_query_images=len({t['query_id'] for t in training}),
                query_mask_occurrences=8000,extra_remote_support_mask_occurrences=2*sum(t['query_id']!=t['support_id'] for t in training),
                support_click_occurrences=2*sum(len(t['positive_xy'])+len(t['negative_xy']) for t in training),
                same_for_all_spatial_arms=True,qualification='occurrences not independent annotations; local-vs-mixed future tasks require a new ledger')
    contract=dict(status='accepted_spatial_task_protocol_only',source_sha256=sha(__file__),arms=list(ARMS),input_identity=data.identity,
                  source_tasks_sha256=sha(source),consumption_sha256=sha(a.output/'training_consumption.jsonl'),
                  evaluation_sha256=sha(a.output/'evaluation.jsonl'),evaluation_rows=3711,updates=2000,batch=4,seed=20261103,
                  fixed_final_checkpoint=True,threshold=.5,local_remote_separate=True,query_GT_read=False,
                  supervision_ledger=ledger,formal_training_allowed=False,remaining='trainer integration, all learning/inventory acceptances, smoke and comparison acceptance')
    (a.output/'acceptance.json').write_text(json.dumps(contract,indent=2)+'\n')


if __name__=='__main__':main()
