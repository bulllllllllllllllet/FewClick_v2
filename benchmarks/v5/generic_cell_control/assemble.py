"""Assemble actual PCA64 cell caches while preserving RGB, labels and tasks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .fit import project
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.protocol import resolve_clicks
from benchmarks.v5.multidomain_training.scoring import BACKGROUND_TARGETS

ARMS = ('generic_he_pca64', 'generic_he_white64')


def read(path): return json.loads(Path(path).read_text())


def assemble(base_tasks, base_realized, raw_shards, pca, output, gate=False):
    output.mkdir(parents=True, exist_ok=False)
    base = read(base_tasks/'contract.json'); original_catalog = Path(base['catalog'])
    require(digest(original_catalog/'tiles.parquet') == base['catalog_sha256'], 'original catalog changed')
    rows = pd.read_parquet(original_catalog/'tiles.parquet').to_dict('records'); fields = {r['tile_id']:r for r in rows}
    fit = read(pca/'summary.json')
    require(fit['status'] == ('complete_gate' if gate else 'complete_train_fit') and fit['gate'] == gate, 'PCA stage mismatch')
    require(fit['pca_sha256'] == digest(pca/'pca64.npz') and fit['validation_raw_files_opened_for_fit'] == 0, 'PCA identity/leakage')
    require(read(pca/'run.json')['catalog_sha256'] == base['catalog_sha256'], 'PCA catalog')
    with np.load(pca/'pca64.npz',allow_pickle=False) as saved: model = {k:saved[k].copy() for k in saved.files}
    records = {}
    for shard in raw_shards:
        result = read(shard/'summary.json'); meta = read(shard/'run.json')
        require(result['status'] == 'complete' and result['rows_sha256'] == digest(shard/'rows.jsonl'), 'raw shard incomplete')
        require(meta['recipe']['catalog_sha256'] == base['catalog_sha256'], 'raw shard catalog')
        for line in (shard/'rows.jsonl').open():
            row = json.loads(line); key = row['tile_id'],row['scale']
            require(key not in records and row['split'] == fields[row['tile_id']]['split'], 'raw duplicate/split')
            records[key] = row
    identities = sorted({tile for tile,scale in records})
    require(set(records) == {(tile,s) for tile in identities for s in range(3)}, 'missing physical scale')
    require(len(identities) == (4 if gate else 1737), 'incomplete source coverage')
    actual_lines = (base_realized/'training_plans.jsonl').read_text().splitlines(keepends=True)
    require(digest(base_realized/'training_plans.jsonl') == 'fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', 'original actual40k changed')
    actual = [json.loads(line) for line in actual_lines]
    if gate:
        chosen = {}
        for domain in ('crc','tiger'):
            chosen[domain] = [i for i,p in enumerate(actual) if p['dataset'] == domain and p['support_id'] in identities][:4]
            require(len(chosen[domain]) == 4, 'gate training source coverage')
        indices = [chosen[d][i] for i in range(4) for d in ('crc','tiger')]
        selected_lines = [actual_lines[i] for i in indices]
        validation = []
        for number,row in enumerate(sorted((fields[t] for t in identities if fields[t]['split']=='val'),key=lambda r:r['dataset'])):
            targets = sorted(int(t) for t in json.loads(row['eligible_json']) if int(t) not in BACKGROUND_TARGETS[row['dataset']])
            require(bool(targets), 'gate val target coverage')
            plan = dict(number=90000000+number,pair_id=90000000+number,prompt_group=0,
                support_id=row['tile_id'],query_id=row['tile_id'],dataset=row['dataset'],target=targets[0],
                kind='local',protocol='local',split='val',support_quality='original',query_quality='original',
                quality_condition='original',support_organ=row['organ'],query_organ=row['organ'],
                gt_simulated_clicks=True,click_seed=20261025+number)
            validation.append(resolve_clicks(plan,row))
        require(len(validation) == 2, 'gate val two domains')
    else:
        selected_lines = actual_lines; validation = read(base_tasks/'validation_maximum.json')
    manifest = {}
    for arm in ARMS:
        folder = output/arm; folder.mkdir(); catalog = folder/'catalog';catalog.mkdir()
        cells_dir = folder/'cells';cells_dir.mkdir();contexts_dir = folder/'contexts';contexts_dir.mkdir()
        changed_rows = []; audits = []
        for tile in identities:
            row = fields[tile]; variant = json.loads(row['variants_json'])['original']
            require(digest(variant['cells']) == variant['cells_sha256'] and digest(row['multiscale_cache']) == row['multiscale_sha256'], 'original cell cache changed')
            features = {}; totals = {}
            with np.load(variant['cells'],allow_pickle=False) as old:
                original_features = {0:old['features'].copy()};original_totals = {0:int(old['total'])}
            with np.load(row['multiscale_cache'],allow_pickle=False) as old: contexts = {k:old[k].copy() for k in old.files}
            for scale in (1,2):original_features[scale] = contexts[f'cells_{scale}'];original_totals[scale] = int(contexts[f'total_{scale}'])
            for scale in range(3):
                source = records[tile,scale]; require(digest(source['output']) == source['output_sha256'], 'raw cache changed')
                with np.load(source['output'],allow_pickle=False) as cache:
                    raw = cache['raw768'];geometry = cache['geometry10'];total = int(cache['total'])
                require(raw.shape == (len(geometry),768) and geometry.shape == (len(raw),10) and total == original_totals[scale], 'raw shape/count')
                require(np.array_equal(geometry,original_features[scale][:,:10]), 'geometry changed')
                appearance = project(raw,model,arm == 'generic_he_white64')
                features[scale] = np.concatenate((geometry,appearance),axis=1).astype(np.float32)
                totals[scale] = total
                require(features[scale].shape == original_features[scale].shape and np.isfinite(features[scale]).all(), 'new74 features')
            fine = cells_dir/f'{tile}.npz';np.savez_compressed(fine,features=features[0],total=np.asarray(totals[0]))
            for scale in (1,2):contexts[f'cells_{scale}'] = features[scale]
            context = contexts_dir/f'{tile}.npz';np.savez_compressed(context,**contexts)
            with np.load(context,allow_pickle=False) as saved:
                require(all(np.array_equal(saved[k],v) for k,v in contexts.items()), 'context disk replay')
            with np.load(fine,allow_pickle=False) as saved:require(np.array_equal(saved['features'],features[0]), 'fine disk replay')
            variants = json.loads(row['variants_json']);require(set(variants)=={'original'}, 'unhandled image quality variants')
            variants['original'] = variant|dict(cells=str(fine),cells_sha256=digest(fine))
            changed = row|dict(variants_json=json.dumps(variants),multiscale_cache=str(context),multiscale_sha256=digest(context),
                cell_recipe_id=arm+'_'+fit['pca_sha256'],generic_HE_provenance_json=json.dumps(dict(pca=str(pca),pca_sha256=fit['pca_sha256'],whitened=arm=='generic_he_white64',source='CTransPath public pre-alignment raw768')))
            for key in ('labels','labels_array_sha256','space_key','patient_id','spatial_group_id','input_mpp_xy_json','multiscale_json'):
                require(changed[key] == row[key], 'source/label/physical geometry changed')
            changed_rows.append(changed);audits.append(dict(tile_id=tile,split=row['split'],fine_sha256=digest(fine),context_sha256=digest(context)))
        pd.DataFrame(changed_rows).to_parquet(catalog/'tiles.parquet',index=False)
        if (original_catalog/'training_policy.json').exists():shutil.copyfile(original_catalog/'training_policy.json',catalog/'training_policy.json')
        tasks = folder/'tasks';tasks.mkdir();realized = folder/'realized';realized.mkdir()
        if gate:write(tasks/'validation_maximum.json',validation)
        else:shutil.copyfile(base_tasks/'validation_maximum.json',tasks/'validation_maximum.json')
        (realized/'training_plans.jsonl').write_text(''.join(selected_lines))
        if gate:
            recipe = {p['number']:line for line in (base_tasks/'training_recipe.jsonl').read_text().splitlines(keepends=True) for p in [json.loads(line)]}
            (tasks/'training_recipe.jsonl').write_text(''.join(recipe[actual[i]['number']] for i in indices))
        else:shutil.copyfile(base_tasks/'training_recipe.jsonl',tasks/'training_recipe.jsonl')
        contract = base|dict(version='generic_HE_pca_control_v1_1025',catalog=str(catalog),catalog_sha256=digest(catalog/'tiles.parquet'),
            validation_sha256=digest(tasks/'validation_maximum.json'),training_recipe_sha256=digest(tasks/'training_recipe.jsonl'),
            parent_protocol_sha256=digest(base_tasks/'contract.json'),generic_HE_arm=arm,
            generic_HE_pca=str(pca),generic_HE_pca_sha256=fit['pca_sha256'],gate=gate,
            source_sha256=digest(__file__),feature_scope='same74 interface and geometry/count; generic HE64 replaces XCell64')
        if gate:contract.update(training_tasks=8,training_steps=4,accumulate=2,eval_every=2,validation_tasks=2,validation_pairs=2,validation_point_groups=1)
        write(tasks/'contract.json',contract)
        original_realized = read(base_realized/'summary.json')
        write(realized/'summary.json',original_realized|dict(status='complete',protocol=str(tasks),protocol_sha256=digest(tasks/'contract.json'),
            training_plans_sha256=digest(realized/'training_plans.jsonl'),recipe_sha256=digest(tasks/'training_recipe.jsonl'),tasks=len(selected_lines),gate=gate))
        write(folder/'assembly_audit.json',dict(status='passed',gate=gate,fields=len(changed_rows),records=audits,
            original_catalog_sha256=base['catalog_sha256'],catalog_sha256=digest(catalog/'tiles.parquet'),
            pca_sha256=fit['pca_sha256'],feature_assembly_tissue_GT_used=False,GT_used_for_gate_prompts=gate))
        manifest[arm] = dict(catalog=str(catalog),tasks=str(tasks),realized=str(realized),
            protocol_sha256=digest(tasks/'contract.json'),coordinate_sha256=digest(realized/'training_plans.jsonl'),
            validation_sha256=digest(tasks/'validation_maximum.json'),gate=gate,pca_sha256=fit['pca_sha256'])
    require(len({v['coordinate_sha256'] for v in manifest.values()}) == 1 and len({v['validation_sha256'] for v in manifest.values()}) == 1, 'arm task parity')
    write(output/'manifest.json',manifest)
    write(output/'summary.json',dict(status='complete_gate_assembly' if gate else 'complete',gate=gate,fields=len(identities),
        manifest_sha256=digest(output/'manifest.json'),pca_sha256=fit['pca_sha256'],source_sha256=digest(__file__),
        scope='Cache/task assembly; actual model gate and formal training remain separate'))
    print(json.dumps(dict(status='complete_gate_assembly' if gate else 'complete',fields=len(identities),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base-tasks','base-realized','pca','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--raw-shards',type=Path,nargs='+',required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    if a.output.exists():raise SystemExit('output already exists')
    assemble(a.base_tasks,a.base_realized,a.raw_shards,a.pca,a.output,a.gate)
