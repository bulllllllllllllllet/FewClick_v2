"""Version seeded task permutations without changing any actual click or sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import LocalData

SEEDS=(20261026,20261027)


def multiset_sha(lines):
    return hashlib.sha256(''.join(sorted(lines)).encode()).hexdigest()


def prepare(tasks,realized,output):
    output.mkdir(parents=True,exist_ok=False)
    base=json.loads((tasks/'contract.json').read_text());actual_meta=json.loads((realized/'summary.json').read_text())
    recipes=(tasks/'training_recipe.jsonl').read_text().splitlines(keepends=True)
    actual=(realized/'training_plans.jsonl').read_text().splitlines(keepends=True)
    if len(recipes)!=40000 or len(actual)!=40000 or actual_meta['status']!='complete':raise ValueError('base task pool incomplete')
    if digest(tasks/'training_recipe.jsonl')!=base['training_recipe_sha256'] or digest(realized/'training_plans.jsonl')!=actual_meta['training_plans_sha256']:
        raise ValueError('base recipe/actual file changed')
    if actual_meta['protocol_sha256']!=digest(tasks/'contract.json'):raise ValueError('base actual protocol mismatch')
    recipe_objects=[json.loads(line) for line in recipes];actual_objects=[json.loads(line) for line in actual]
    numbers=[p['number'] for p in actual_objects]
    if len(set(numbers))!=40000:raise ValueError('duplicate base training identities')
    for recipe,plan in zip(recipe_objects,actual_objects):
        for key in recipe:
            if recipe[key]!=plan[key]:raise ValueError('recipe/actual identity mismatch')
    base_multiset=multiset_sha(actual);manifest={}
    for seed in SEEDS:
        folder=output/f'seed_{seed}';folder.mkdir();protocol=folder/'tasks';ordered=folder/'realized';gate=folder/'gate_realized'
        for path in (protocol,ordered,gate):path.mkdir()
        permutation=np.random.default_rng(seed).permutation(len(actual))
        if np.array_equal(permutation,np.arange(len(actual))):raise ValueError('replication order identical')
        recipe_lines=[recipes[i] for i in permutation];actual_lines=[actual[i] for i in permutation]
        (protocol/'training_recipe.jsonl').write_text(''.join(recipe_lines));(ordered/'training_plans.jsonl').write_text(''.join(actual_lines))
        (protocol/'validation_maximum.json').write_bytes((tasks/'validation_maximum.json').read_bytes())
        if multiset_sha(actual_lines)!=base_multiset:raise ValueError('actual click multiset changed')
        write(folder/'permutation.json',list(map(int,permutation)))
        contract=base|dict(version='local_fair_seed_order_v1_1025',replication_seed=seed,
            base_protocol_sha256=digest(tasks/'contract.json'),base_coordinate_sha256=actual_meta['training_plans_sha256'],
            training_recipe_sha256=digest(protocol/'training_recipe.jsonl'),
            inherited_validation_sha256=digest(tasks/'validation_maximum.json'),
            training_coordinate_multiset_sha256=base_multiset,order_permutation_sha256=digest(folder/'permutation.json'),
            replication_scope='same complete task multiset/actual coordinates, new order and train RNG; not new data or new evaluation',
            initialization_scope='shared pretrained weights; fresh heads/train RNG; Dense internal initialization fixed by frozen implementation')
        write(protocol/'contract.json',contract)
        summary=actual_meta|dict(protocol=str(protocol),protocol_sha256=digest(protocol/'contract.json'),
            training_plans_sha256=digest(ordered/'training_plans.jsonl'),recipe_sha256=digest(protocol/'training_recipe.jsonl'),
            replication_seed=seed,base_protocol=str(tasks),base_actual=str(realized),multiset_sha256=base_multiset,
            direct_replay_passed=True,replay_scope='exact original actual lines under a verified bijective permutation')
        write(ordered/'summary.json',summary)
        by_domain={domain:[] for domain in ('crc','tiger')}
        for index in permutation:
            domain=actual_objects[index]['dataset']
            if len(by_domain[domain])<4:by_domain[domain].append(actual[index])
            if all(len(v)==4 for v in by_domain.values()):break
        gate_lines=[line for domain in sorted(by_domain) for line in by_domain[domain]]
        if len(gate_lines)!=8:raise ValueError('two-domain native gate incomplete')
        (gate/'training_plans.jsonl').write_text(''.join(gate_lines))
        write(gate/'summary.json',summary|dict(tasks=8,complete_protocol=False,
            training_plans_sha256=digest(gate/'training_plans.jsonl'),counts={'crc':4,'tiger':4},
            replay_scope='per-domain first four tasks in seeded order; representative native gate only'))
        data=LocalData(protocol,ordered)
        if len(data.training)!=40000 or digest(protocol/'validation_maximum.json')!=base['validation_sha256']:
            raise ValueError('new protocol lost tasks/validation identity')
        # Exercise each new protocol through the actual unchanged loader.
        checks=[]
        for domain in ('crc','tiger'):
            plan=next(p for p in data.training if p['dataset']==domain)
            raw,prompts,supervision=data.materialize(plan,'cpu',True,True)
            checks.append(dict(domain=domain,number=plan['number'],input_keys=len(raw),
                valid_pixels=int(supervision['valid'].sum()),positive_points=len(plan['positive_xy'])))
        manifest[str(seed)]=dict(tasks=str(protocol),realized=str(ordered),gate_realized=str(gate),
            protocol_sha256=digest(protocol/'contract.json'),coordinate_sha256=digest(ordered/'training_plans.jsonl'),
            gate_coordinate_sha256=digest(gate/'training_plans.jsonl'),actual_loader=checks)
    if len({v['coordinate_sha256'] for v in manifest.values()})!=2:raise ValueError('seed orders not independent')
    write(output/'manifest.json',manifest)
    write(output/'summary.json',dict(status='complete_preparation',additional_seeds=list(SEEDS),
        base_coordinate_sha256=actual_meta['training_plans_sha256'],base_coordinate_multiset_sha256=base_multiset,
        identical_validation=True,exact_task_and_click_multiset=True,real_loader_passed=True,
        native_GPU_gate_pending=True,source_sha256=digest(__file__)))
    print(json.dumps(dict(status='complete_preparation',output=str(output),seeds=list(SEEDS))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--realized',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.tasks,a.realized,a.output)
