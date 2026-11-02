"""Independent local task identity/sign audit and actual shared-loader checks."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.architecture_bridge.repaired_model import RepairedData
from benchmarks.v5.fair_comparison.protocol import resolve_clicks


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    proof=json.loads((a.run/'acceptance.json').read_text());assert proof['status']=='accepted_local4000_clicks_preparation' and proof['training_sha256']==sha(a.run/'training.jsonl')
    prepared=json.loads((a.prepared/'report.json').read_text());assert proof['identity']['recipe_sha256']==sha(a.prepared/'local_recipe.jsonl')==prepared['local_recipe_sha256']
    recipes=list(map(json.loads,(a.prepared/'local_recipe.jsonl').read_text().splitlines()));tasks=list(map(json.loads,(a.run/'training.jsonl').read_text().splitlines()));assert len(tasks)==len(recipes)==4000
    data=RepairedData(a.cache);labels={}
    for task,recipe in zip(tasks,recipes):
        for key,value in recipe.items():
            if key!='needs_click_realization':assert task[key]==value,(task['number'],key)
        assert not task['needs_click_realization'] and task['gt_simulated_clicks'] and task['query_id']==task['support_id']
        row=data.rows[task['support_id']];assert row['split']=='train' and row['dataset']==task['dataset']
        if task['support_id'] not in labels:
            with Image.open(row['labels']) as im:gt=np.asarray(im).copy()
            assert hashlib.sha256(gt.tobytes()).hexdigest()==row['labels_array_sha256'];labels[task['support_id']]=gt
        gt=labels[task['support_id']]
        for sign in ['positive','negative']:
            assert len(task[sign+'_xy'])==recipe[sign+'_count']
            for x,y in task[sign+'_xy']:
                assert 0<x<1 and 0<y<1;label=gt[int(np.floor(y*gt.shape[0])),int(np.floor(x*gt.shape[1]))]
                assert label!=255 and ((label==task['target'])==(sign=='positive'))
    assert Counter((t['dataset'],t['target'],len(t['positive_xy']),len(t['negative_xy'])) for t in tasks)==Counter((r['dataset'],r['target'],r['positive_count'],r['negative_count']) for r in recipes)
    # Choose numerical probes by immutable index hash, never by model output.
    selected=sorted(range(4000),key=lambda i:hashlib.sha256(f'P3localaccept|{i}'.encode()).digest())[:16]
    for i in selected:
        direct=resolve_clicks(recipes[i],data.rows[tasks[i]['support_id']],(recipes[i]['positive_count'],recipes[i]['negative_count']))
        for key in ['positive_xy','negative_xy','click_audit','prompt_policy']:assert direct[key]==tasks[i][key]
        inputs=data.model_inputs(tasks[i],'xcell','cpu');assert inputs['local'] and inputs['shared'] and inputs['support'] is inputs['query']
        for scale in ['', '_1','_2']:
            assert inputs['support']['cells'+scale].shape[-1]==74 and torch.isfinite(inputs['support']['cells'+scale]).all()
            assert inputs['support']['image'+scale].shape==(1,3,512,512)
        assert inputs['prompts']['positive_xy'].shape[1]==recipes[i]['positive_count']
        assert inputs['prompts']['negative_xy'].shape[1]==recipes[i]['negative_count']
    a.output.mkdir(exist_ok=False)
    receipt=dict(status='accepted_independent_local4000_clicks_and_loader',tasks=4000,all_signs_and_quotas_checked=True,independent_policy_replays=16,real_loader_tasks=16,
                 unique_train_GT_read=len(labels),val_GT_read=0,source_sha256=sha(__file__),resolver_source_sha256=sha(inspect.getfile(resolve_clicks)),
                 training_sha256=sha(a.run/'training.jsonl'),preparation_acceptance_sha256=sha(a.run/'acceptance.json'),inputs=data.identity,
                 training_allowed=False,remaining='final architecture/recipe supervision scope and actual model learning acceptance required')
    (a.output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
