"""Bind existing spatial development inputs to the new CVPR experiment contract."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.transfer_comparison.data import TransferData


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def overlap(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    base=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison')
    lp=base/'protocol_1025_124208/tasks_1025_125400';rp=base/'transfer_1025_131856/tasks_1025_132300'
    local=LocalData(lp);remote=TransferData(rp);rows=list(local.rows.values())
    spaces=defaultdict(list)
    for row in rows:spaces[row['space_key']].append(row)
    split_overlaps=0
    for group in spaces.values():
        train=[json.loads(x['multiscale_json'])['context_um'] for x in group if x['split']=='train']
        dev=[json.loads(x['multiscale_json'])['context_um'] for x in group if x['split']=='val']
        split_overlaps+=sum(overlap(x,y) for x in train for y in dev)
    assert split_overlaps==0
    remote_primary=[nested(x,'3+2') for x in remote.plans if x['prompt_group']==0]
    assert len(remote_primary)==935
    for plan in remote_primary:
        assert not plan['query_has_human_clicks'] and plan['support_id']!=plan['query_id']
        s,q=(local.rows[plan[k]] for k in ['support_id','query_id'])
        assert s['split']==q['split']=='val'
        if s['space_key']==q['space_key']:
            assert not overlap(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um'])
    # Same RGB hashes across a split would reveal duplicated input representations.
    hashes=defaultdict(set)
    for row in rows:hashes[json.loads(row['variants_json'])['original']['image_sha256']].add(row['split'])
    assert not any(len(x)>1 for x in hashes.values())
    # Input gates: mixed labels, ignore coverage, sparse cell population are all selected without scores.
    selected={}
    for dataset in ['crc','tiger']:
        for split in ['train','val']:
            group=sorted([x for x in rows if x['dataset']==dataset and x['split']==split],key=lambda x:x['tile_id'])
            selected[(dataset,split,'identity')]=group[0]
            selected[(dataset,split,'mixed')]=max(group,key=lambda x:len([v for k,v in json.loads(x['counts_json']).items() if k!='255' and v>0]))
            selected[(dataset,split,'ignore')]=max(group,key=lambda x:json.loads(x['counts_json']).get('255',0))
            def total_cells(row):
                with np.load(json.loads(row['variants_json'])['original']['cells'],allow_pickle=False) as cache:
                    return int(cache['total'])
            selected[(dataset,split,'sparse_first32_identity_candidates')]=min(group[:32],key=total_cells)
            if dataset=='crc':
                selected[(dataset,split,'fat')]=max(group,key=lambda x:json.loads(x['counts_json']).get('10',0))
    gates=[]
    for reason,row in selected.items():
        raw,gt=local.raw(row['tile_id'],contexts=True,cells=True)
        assert tuple(raw['rgb'].shape)==(1,3,512,512)
        assert all(tuple(raw[f'rgb_{i}'].shape)==(1,3,512,512) for i in (1,2))
        gates.append(dict(reason=list(reason),tile_id=row['tile_id'],labels=np.unique(gt).tolist(),
            valid_pixels=int((gt!=255).sum()),cells=raw['cells'].shape[1],hash_loader_checks=True))
    remote_gates=[]
    for kind in sorted({x['kind'] for x in remote_primary}):
        plan=next(x for x in remote_primary if x['kind']==kind)
        inputs=remote.materialize_inputs(plan,'cpu',contexts=True,cells=True)
        assert set(inputs['query'])==set(inputs['support'])
        assert not any(k in inputs['query'] for k in ['GT','labels','target','positive_xy','negative_xy'])
        remote_gates.append(dict(kind=kind,support_id=plan['support_id'],query_id=plan['query_id'],query_has_points=False))
    manifest=[]
    for row in rows:
        manifest.append({k:row[k] for k in ['tile_id','dataset','split','spatial_group_id','space_key','patient_id','slide_verified','labels','labels_array_sha256','input_mpp_xy_json']}|
                        dict(max_context_um=json.loads(row['multiscale_json'])['context_um'],image_variants=json.loads(row['variants_json'])))
    training_path=base/'protocol_1025_124208/realized_train_1025_125928/training_plans.jsonl'
    training_summary=json.loads((training_path.parent/'summary.json').read_text())
    assert sha(training_path)==training_summary['training_plans_sha256']
    training=[]
    with training_path.open() as f:
        for line in f:
            plan=json.loads(line)
            assert plan['split']=='train' and plan['kind']=='local' and plan['support_id']==plan['query_id']
            assert local.rows[plan['query_id']]['split']=='train'
            training.append(plan)
            if len(training)==4000:break
    assert len(training)==4000
    with (a.output/'pilot_training4000.jsonl').open('w') as f:
        for plan in training:f.write(json.dumps(plan,sort_keys=True)+'\n')
    with (a.output/'tissue_inputs.jsonl').open('w') as f:
        for row in manifest:f.write(json.dumps(row)+'\n')
    report=dict(status='complete_tissue_development_contract_gate',counts=dict(Counter(x['dataset']+'|'+x['split'] for x in rows)),
        inputs=len(rows),local_primary_plans=len(local.validation),remote_primary_pairs=len(remote_primary),
        remote_kinds=dict(Counter(x['kind'] for x in remote_primary)),max_context_cross_split_overlaps=0,
        duplicate_RGB_cross_split=0,gates=gates,remote_gates=remote_gates,elapsed_seconds=time.time()-start,
        identities={str(p):sha(p) for p in [local.catalog/'tiles.parquet',lp/'contract.json',lp/'validation_maximum.json',rp/'contract.json',rp/'validation_maximum.json']},
        source_sha256=sha(__file__),manifest_sha256=sha(a.output/'tissue_inputs.jsonl'),
        frozen_training=dict(path=str(a.output/'pilot_training4000.jsonl'),sha256=sha(a.output/'pilot_training4000.jsonl'),tasks=4000,
                             original_path=str(training_path),original_sha256=sha(training_path)),
        primary_budget='3+2',secondary_budgets=['1+1','5+1'],threshold=.5,BF1_tolerance_um=2,
        GT_policy='source GT simulates support points; query GT only benchmark task definition and scoring; inference uses no query labels or clicks',
        absent_policy='FPA=positive predicted pixels/valid query pixels; Dice on present targets; partial annotations evaluate only known valid domain; report NA where no eligible cases',
        exposure=dict(fit='existing spatial train blocks; no J5 target-supervised initialization in controlled arms',
            development='existing local and935remote pairs repeatedly exposed; cannot call blind or independent-patient',
            confirmation='No demonstrably unexposed confirmation domain yet. Must establish before P5; never relabel old dev as blind test.',
            cells='OCELOT train/dev expert points only; XCell internal training per user; upstream TCGA/pathology pretraining separately disclosed',
            graph_and_crop='No cross-tile graph or wider context than audited footprint permitted under this contract; extensions require new audit'),
        pilot=dict(max_updates=1000,tasks_per_update=4,training_recipe='first4000 of frozen local realized train sequence; same identities and points in every image arm',
                   selection='fixed final update; dev only; no new arbitrary stopping/threshold selection'),
        constraints=['local and remote separate tables','SAM feature encoder versus native SAM system separate','partial annotation absence means absent in valid pixels only','no imposed patient split; preserve current spatial blocks','all classes including fat supervised, no adversarial heldout fat','cell probe scores cannot substitute downstream tissue evidence'],
        remaining='Independent confirmation and WSI full-cohort contracts remain future phase gates, not claims of completed evidence')
    (a.output/'contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],inputs=len(rows),gates=len(gates),remote_gates=len(remote_gates),seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
