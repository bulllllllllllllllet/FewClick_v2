"""Build independent masked-adaptation and original-label heldout test contracts."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .audit import FOLDS
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.protocol import resolve_clicks


def read(path):return json.loads(Path(path).read_text())


def build(base_tasks,audit,remote,output):
    output.mkdir(parents=True,exist_ok=False)
    base=read(base_tasks/'contract.json');original_catalog=Path(base['catalog'])
    require(digest(original_catalog/'tiles.parquet')==base['catalog_sha256'],'original catalog')
    prior=read(audit/'summary.json');require(prior['status']=='complete_feasibility_audit','actual feasibility audit')
    rows=pd.read_parquet(original_catalog/'tiles.parquet').to_dict('records');fields={r['tile_id']:r for r in rows}
    old_local=read(base_tasks/'validation_maximum.json');old_remote=read(remote/'validation_maximum.json')
    require(digest(base_tasks/'validation_maximum.json')==base['validation_sha256'],'old local points')
    require(digest(remote/'validation_maximum.json')==read(remote/'contract.json')['tasks_sha256'],'old remote points')
    policy=read(original_catalog/'training_policy.json');probability=dict(zip(policy['datasets'],policy['probabilities']))
    manifest={}
    for fold_index,(fold,spec) in enumerate(FOLDS.items()):
        folder=output/fold;folder.mkdir();catalog=folder/'catalog';catalog.mkdir();labels_dir=catalog/'labels';labels_dir.mkdir()
        candidates=read(audit/f'{fold}_candidates.json');require(digest(audit/f'{fold}_candidates.json')==prior['folds'][fold]['candidates_sha256'],'candidate identity')
        by_tile=defaultdict(list);pool=defaultdict(list)
        for entry in candidates:
            by_tile[entry['tile_id']].append(entry['target'])
            pool[entry['split'],entry['dataset'],entry['target']].append(entry['tile_id'])
        changed=[]
        for row in rows:
            labels=np.asarray(Image.open(row['labels']));require(hashlib.sha256(labels.tobytes()).hexdigest()==row['labels_array_sha256'],'original GT SHA')
            masked=labels.copy();masked[np.isin(masked,spec['exclude'][row['dataset']])]=255
            path=labels_dir/f'{row["tile_id"]}.png';Image.fromarray(masked.astype(np.uint8)).save(path)
            reloaded=np.asarray(Image.open(path));require(np.array_equal(masked,reloaded),'masked label disk replay')
            values,counts=np.unique(masked,return_counts=True)
            count_map={str(int(v)):int(n) for v,n in zip(values,counts) if v!=255}
            interior=json.loads(row['interior_counts_json']);interior={str(k):v for k,v in interior.items() if int(k) not in spec['exclude'][row['dataset']]}
            changed.append(row|dict(labels=str(path),labels_array_sha256=hashlib.sha256(masked.tobytes()).hexdigest(),
                counts_json=json.dumps(count_map),eligible_json=json.dumps(sorted(by_tile[row['tile_id']])),interior_counts_json=json.dumps(interior),
                heldout_original_labels=row['labels'],heldout_original_labels_sha256=row['labels_array_sha256']))
        pd.DataFrame(changed).to_parquet(catalog/'tiles.parquet',index=False)
        shutil.copyfile(original_catalog/'training_policy.json',catalog/'training_policy.json')
        changed_by_id={r['tile_id']:r for r in changed};tasks=folder/'tasks';tasks.mkdir()
        validation=[];pair=60000000+fold_index*1000000;coverage=[]
        for (split,domain,target),tiles in sorted(pool.items()):
            if split!='val':continue
            ordered=sorted(tiles,key=lambda tile:hashlib.sha256(f'heldout-v1|{fold}|{domain}|{target}|{tile}'.encode()).hexdigest())
            chosen=ordered[:20];coverage.append(dict(dataset=domain,target=target,available=len(ordered),selected=len(chosen)))
            for tile in chosen:
                row=changed_by_id[tile]
                for group in range(5):
                    maximum=dict(number=pair,pair_id=pair,prompt_group=group,support_id=tile,query_id=tile,
                        dataset=domain,target=target,kind='local',protocol='local',split='val',support_quality='original',query_quality='original',
                        quality_condition='original',support_organ=row['organ'],query_organ=row['organ'],gt_simulated_clicks=True,
                        click_seed=20261026+pair+group*1000000000)
                    validation.append(resolve_clicks(maximum,row))
                pair+=1
        write(tasks/'validation_maximum.json',validation)
        rng=np.random.default_rng(20261026+fold_index);domains=sorted({d for split,d,t in pool if split=='train'})
        require(set(domains)=={'crc','tiger'},'both training domains required')
        weights=np.array([probability[d] for d in domains]);weights=weights/weights.sum()
        classes={d:sorted(t for split,domain,t in pool if split=='train' and domain==d) for d in domains}
        with (tasks/'training_recipe.jsonl').open('x') as stream:
            for number in range(40000):
                domain=str(rng.choice(domains,p=weights));target=int(rng.choice(classes[domain]))
                tiles=sorted(pool['train',domain,target]);tile=tiles[int(rng.integers(len(tiles)))];row=fields[tile]
                plan=dict(number=number,support_id=tile,query_id=tile,dataset=domain,target=target,kind='local',protocol='local',split='train',
                    support_quality='original',query_quality='original',support_organ=row['organ'],query_organ=row['organ'],
                    gt_simulated_clicks=True,positive_count=int(rng.integers(1,6)),negative_count=int(rng.integers(1,6)),
                    click_seed=20261026+fold_index*100000000+500000000+number)
                require(target not in spec['exclude'][domain],'excluded training target');stream.write(json.dumps(plan,sort_keys=True)+'\n')
        contract=base|dict(version='heldout_supervision_v1_1026',catalog=str(catalog),catalog_sha256=digest(catalog/'tiles.parquet'),
            validation_sha256=digest(tasks/'validation_maximum.json'),training_recipe_sha256=digest(tasks/'training_recipe.jsonl'),
            parent_protocol_sha256=digest(base_tasks/'contract.json'),heldout_fold=fold,heldout_excluded_labels=spec['exclude'],
            heldout_test_targets=spec['test'],coverage=coverage,validation_tasks=len(validation),validation_pairs=pair-(60000000+fold_index*1000000),
            validation_unique_fields=len({p['query_id'] for p in validation}),seed=20261026+fold_index,
            validation_scope='seen-target masked-label spatial development selection; no heldout-target scores used',
            claim_scope='no excluded class labels in current adaptation or seen selection; images/public priors may expose morphology',
            source_sha256=digest(__file__),strict_context_morphology_exclusion=False,
            initialization_contract='noJ5, no previously12-class tissue supervision; primary FewClick image-only; public priors qualified')
        write(tasks/'contract.json',contract)
        tests=folder/'heldout_local';tests.mkdir()
        selected=[p for p in old_local if p['target'] in spec['test'][p['dataset']]];require(selected,'heldout local tasks missing')
        write(tests/'validation_maximum.json',selected)
        write(tests/'contract.json',base|dict(version='heldout_score_only_v1_1026',heldout_fold=fold,
            heldout_test_targets=spec['test'],validation_sha256=digest(tests/'validation_maximum.json'),
            validation_tasks=len(selected),validation_pairs=len({p['pair_id'] for p in selected}),
            training_protocol=str(tasks),training_protocol_sha256=digest(tasks/'contract.json'),
            checkpoint_selection_allowed=False,scope='OriginalGT/old frozen heldout source points, terminal scoring only'))
        rt=folder/'heldout_remote';rt.mkdir()
        selected_remote=[p for p in old_remote if p['target'] in spec['test'][p['dataset']]];require(selected_remote,'heldout remote missing')
        write(rt/'validation_maximum.json',selected_remote)
        write(rt/'contract.json',read(remote/'contract.json')|dict(version='heldout_remote_score_only_v1_1026',heldout_fold=fold,
            tasks_sha256=digest(rt/'validation_maximum.json'),tasks=len(selected_remote),pairs=len({p['pair_id'] for p in selected_remote}),
            training_protocol=str(tasks),training_protocol_sha256=digest(tasks/'contract.json'),checkpoint_selection_allowed=False,
            scope='OriginalGT/original separated query/source points reused; query human points0; terminal scoring only'))
        manifest[fold]=dict(tasks=str(tasks),catalog=str(catalog),heldout_local=str(tests),heldout_remote=str(rt),
            protocol_sha256=digest(tasks/'contract.json'),seen_pairs=contract['validation_pairs'],seen_point_groups=len(validation),
            heldout_local_pairs=len({p['pair_id'] for p in selected}),heldout_remote_pairs=len({p['pair_id'] for p in selected_remote}),
            training_tasks=40000,actual_coordinates_pending=True)
        write(output/'manifest_partial.json',manifest);print(json.dumps(dict(fold=fold,stage='protocol_prepared',**manifest[fold])),flush=True)
    write(output/'manifest.json',manifest)
    write(output/'summary.json',dict(status='complete_protocols',worker_pid=os.getpid(),manifest_sha256=digest(output/'manifest.json'),
        source_sha256=digest(__file__),folds=list(manifest),actual_training_coordinate_generation='pending',
        scope='Three masked-label adaptation/seen-selection and separate original-label heldout test protocols; no model training/accuracy'))
    write(output/'status.json',dict(status='complete_protocols',worker_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-tasks','audit','remote','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('output already exists')
    build(a.base_tasks,a.audit,a.remote,a.output)
