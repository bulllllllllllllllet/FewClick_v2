"""Independent NumPy bilinear and label-pooling recount of frozen diagnostic outputs."""
import argparse
import os
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from .diagnose_local_errors import read, write, sha


def run(root):
    summary=read(root/'summary.json');contract=read(root/'contract.json');rows=read(root/'rows.json')
    assert summary['status']=='complete_diagnostic_pending_independent_acceptance'
    assert sha(root/'rows.json')==summary['rows_sha256'] and sha(root/'plans.json')==contract['plans_sha256']
    assert sha('benchmarks/v5/fair_comparison/prompt_region_diagnosis.py')==contract['source_sha256']
    assert all(g['probability_exact'] for g in summary['native_gates'].values()) and set(summary['native_gates'])=={'crc','tiger'}
    plans=read(root/'plans.json');plans=plans['train']+plans['val'];assert len(plans)==len(rows)==summary['n']
    taskroot=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
    tc=read(taskroot/'tasks_1025_125400/contract.json');cat=Path(tc['catalog'])/'tiles.parquet'
    assert sha(cat)==tc['catalog_sha256']
    meta=pd.read_parquet(cat).set_index('tile_id').to_dict('index')
    trainpath=taskroot/'realized_train_1025_125928/training_plans.jsonl'
    assert sha(trainpath)==contract['training_sha256']
    tr=[json.loads(line) for line in trainpath.open()]
    va=[p for p in read(taskroot/'tasks_1025_125400/validation_maximum.json') if p['prompt_group']==0]
    expected=[]
    for role,pool in [('train',tr),('val',va)]:
        seen=set();groups=defaultdict(list)
        for p in pool:
            key=p['dataset'],p['query_id'],p['target']
            if key in seen or len(p['positive_xy'])<3 or len(p['negative_xy'])<2:continue
            seen.add(key);c=json.loads(meta[p['query_id']]['counts_json']);f=c[str(p['target'])]/sum(v for k,v in c.items() if k!='255')
            area='small' if f<.05 else 'medium_small' if f<.2 else 'medium_large' if f<.5 else 'large'
            groups[p['dataset'],area].append(p)
        for k,group in sorted(groups.items()):
            ordered=sorted(group,key=lambda p:hashlib.sha256(f"prompt-region-v1|{p['dataset']}|{p['query_id']}|{p['target']}".encode()).hexdigest())
            expected.extend((role,p['query_id'],p['target']) for p in ordered[:4])
    assert expected==[(p['role'],p['query_id'],p['target']) for p in plans]
    max_error=0.;bindings={};stages=0
    for r,p in zip(rows,plans):
        assert all(r[k]==p[k] for k in ('role','dataset','query_id','target','area_band'))
        file=root/f"case{r['index']:03d}.npz";assert sha(file)==r['artifact_sha256'];bindings[file.name]=sha(file)
        with np.load(file,allow_pickle=False) as z:
            a=z['assignment'].astype(np.float64);truth=z['truth'];valid=z['valid']
            assert a.shape==(64,64,64) and np.max(np.abs(a.sum(0)-1))<2e-6
            t=truth.reshape(64,8,64,8).mean((1,3));v=valid.reshape(64,8,64,8).mean((1,3))
            mass=np.einsum('khw,hw->k',a,v);purity=np.einsum('khw,hw->k',a,t)/np.maximum(mass,1e-8)
            error=float(np.max(np.abs(purity-z['region_purity'])));max_error=max(max_error,error);assert error<3e-6
            for sign in ('positive','negative'):
                xy=z[sign+'_xy'];assert np.array_equal(xy,p[sign+'_xy'])
                w=[]
                for x,y in xy:
                    xx=np.clip(x*64-.5,0,63);yy=np.clip(y*64-.5,0,63);ix=int(xx);iy=int(yy);dx=xx-ix;dy=yy-iy
                    weights=(a[:,iy,ix]*(1-dx)*(1-dy)+a[:,iy,min(ix+1,63)]*dx*(1-dy)+
                        a[:,min(iy+1,63),ix]*(1-dx)*dy+a[:,min(iy+1,63),min(ix+1,63)]*dx*dy)
                    w.append(weights/weights.sum())
                w=np.array(w).T
                error=max(float(np.max(np.abs(w-z[sign+'_weights']))),float(np.max(np.abs(w.T@purity-r[sign+'_purity']))))
                max_error=max(max_error,error);assert error<3e-6
                pixels=np.floor(xy*512).astype(int);codes=truth[pixels[:,1],pixels[:,0]]
                assert np.all(codes if sign=='positive' else ~codes)
            for stage in ('base','dense'):
                mask=z[stage]>=.5;tp=int(np.count_nonzero(mask[truth&valid]));pred=int(np.count_nonzero(mask[valid]));pos=int(truth[valid].sum())
                actual=dict(tp=tp,fp=pred-tp,fn=pos-tp,dice=2*tp/(pos+pred))
                for k,value in actual.items():assert abs(r['metrics'][stage][k]-value)<1e-12
                stages+=1
    for key,s in summary['groups'].items():
        role,band=key.split('|');rr=[r for r in rows if r['role']==role and (band=='all' or r['area_band']==band)]
        assert s['n']==len(rr)
        for stage in ('base','dense'):assert abs(s[stage+'_dice']-np.mean([r['metrics'][stage]['dice'] for r in rr]))<1e-12
        for sign in ('positive','negative'):assert abs(s[sign+'_purity']-np.mean([np.mean(r[sign+'_purity']) for r in rr]))<1e-12
    write(root/'independent_acceptance.json',dict(status='accepted_prompt_region_diagnostic_qualified',worker_pid=os.getpid(),
        rows=len(rows),stage_masks_recounted=stages,max_purity_or_sampling_error=max_error,native_gates=summary['native_gates'],
        frozen_selection_independently_reconstructed=True,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),artifacts=bindings,
        scope='Pixel counts/GT signs/NumPy bilinear and pooled purity accepted. Single-seed posthoc development diagnostic, no causal or population-generalization claim.'))
    print('ACCEPTED',len(rows),'cases',stages,'masks','max error',max_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
