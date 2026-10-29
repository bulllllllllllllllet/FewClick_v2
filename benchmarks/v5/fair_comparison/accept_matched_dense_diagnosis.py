"""Independent selection and count recount for the same-class area-matched diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from .diagnose_local_errors import read,write,sha


def run(root):
    s=read(root/'summary.json');c=read(root/'contract.json');sel=read(root/'selection.json');rows=read(root/'rows.json')
    assert s['status']=='complete_matched_diagnostic_pending_independent_acceptance'
    assert sha(root/'rows.json')==s['rows_sha256'] and sha(root/'selection.json')==c['selection_sha256']
    assert sel==read(root/'selection_preflight.json')
    for p,h in c['source_sha256'].items():assert sha(p)==h
    for p,h in s['reused_bindings'].items():assert sha(p)==h
    assert all(x['base_exact'] and x['dense_exact'] for x in s['gates'].values())
    task=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
    tc=read(task/'tasks_1025_125400/contract.json');catalog=Path(tc['catalog'])/'tiles.parquet';assert sha(catalog)==tc['catalog_sha256']
    meta=pd.read_parquet(catalog).set_index('tile_id').to_dict('index')
    trainpath=task/'realized_train_1025_125928/training_plans.jsonl';assert sha(trainpath)==c['training_sha256']
    pool={}
    for line in trainpath.open():
        p=json.loads(line);k=p['dataset'],p['query_id'],p['target']
        if k in pool or len(p['positive_xy'])<3 or len(p['negative_xy'])<2:continue
        pool[k]=p
    def area(p):
        hist=json.loads(meta[p['query_id']]['counts_json']);return hist[str(p['target'])]/sum(v for k,v in hist.items() if k!='255')
    pilot=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/dense_hardnegative_repair_1029_160100/pilot_1029_160900')
    anchors=read(pilot/'plans.json')['val'];used=set();expected=[];missing=[]
    for i,v in enumerate(anchors):
        options=[]
        for k,t in pool.items():
            if k in used or k[0]!=v['dataset'] or k[2]!=v['target']:continue
            ratio=area(t)/area(v);diff=abs(area(t)-area(v))
            if ratio<.8 or ratio>1.25 or diff>.02:continue
            tie=hashlib.sha256(f"matched-dense-v1|{t['dataset']}|{t['query_id']}|{t['target']}".encode()).hexdigest()
            options.append((diff,tie,k))
        if options:
            chosen=sorted(options)[0][2];used.add(chosen);expected.append((i,chosen))
        else:missing.append(i)
    assert expected==[(p['anchor'],(p['train']['dataset'],p['train']['query_id'],p['train']['target'])) for p in sel['pairs']]
    assert missing==[p['anchor'] for p in sel['missing']]
    assert len(rows)==2*len(expected) and len(rows)==2*s['pairs']
    bypair={};files={}
    for p in sel['pairs']:
        chosen=pool[p['train']['dataset'],p['train']['query_id'],p['train']['target']]
        for sign,count in [('positive',3),('negative',2)]:
            assert p['train'][sign+'_xy']==chosen[sign+'_xy'][:count]
            assert p['val'][sign+'_xy']==anchors[p['anchor']][sign+'_xy']
        for role in ('train','val'):
            i=p['pair'];r=next(r for r in rows if r['pair']==i and r['role']==role);plan=p[role]
            assert (r['query_id'],r['dataset'],r['target'])==(plan['query_id'],plan['dataset'],plan['target'])
            assert meta[r['query_id']]['split']==role
            path=root/f'pair{i:02d}_{role}.npz';assert sha(path)==r['artifact_sha256'];files[path.name]=sha(path)
            with np.load(path,allow_pickle=False) as z:
                truth=z['truth'];valid=z['valid'];pos=int(np.count_nonzero(truth&valid));n=int(np.count_nonzero(valid))
                assert abs(pos/n-area(plan))<1e-12 and abs(pos/n-r['GT_fraction'])<1e-12
                for stage in ('base','dense'):
                    mask=z[stage]>=.5;tp=int(np.count_nonzero(mask&truth&valid));pred=int(np.count_nonzero(mask&valid))
                    numbers=dict(tp=tp,fp=pred-tp,fn=pos-tp,dice=2*tp/(pos+pred))
                    for k,value in numbers.items():assert abs(r['metrics'][stage][k]-value)<1e-12
            bypair[i,role]=r
    pairchanges=[]
    for i in range(len(expected)):
        tr=bypair[i,'train'];va=bypair[i,'val']
        pairchanges.append(dict(pair=i,dataset=tr['dataset'],target=tr['target'],val_fraction=va['GT_fraction'],
            train_gain=tr['dense_gain'],val_gain=va['dense_gain'],gain_gap=tr['dense_gain']-va['dense_gain'],
            base_gap=tr['metrics']['base']['dice']-va['metrics']['base']['dice'],dense_gap=tr['metrics']['dense']['dice']-va['metrics']['dense']['dice']))
    for label,g in s['groups'].items():
        ids={p['pair'] for p in sel['pairs'] if label=='all' or (p['val_fraction']<.05)==(label=='small')}
        for role,values in g.items():
            rr=[bypair[i,role] for i in ids];assert len(rr)==values['n']
            for stage in ('base','dense'):assert abs(statistics.mean(r['metrics'][stage]['dice'] for r in rr)-values[stage+'_dice'])<1e-12
            assert abs(statistics.mean(r['metrics']['dense']['dice']-r['metrics']['base']['dice'] for r in rr)-values['dense_gain'])<1e-12
    write(root/'independent_acceptance.json',dict(status='accepted_matched_dense_diagnostic_qualified',worker_pid=os.getpid(),pairs=len(expected),
        missing=missing,stage_masks_recounted=4*len(expected),selection_independently_reconstructed=True,artifacts=files,pair_changes=pairchanges,
        source_sha256=sha(__file__),summary_sha256=sha(root/'summary.json'),limitations='Descriptive fixed-anchor development matching; morphology, click placement and training exposure still differ.'))
    print('ACCEPTED',len(expected),'pairs',4*len(expected),'stage masks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
