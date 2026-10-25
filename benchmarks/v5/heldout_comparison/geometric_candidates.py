"""Audit full fixed candidate pools against the unchanged maximum click budget."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit import FOLDS
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.multidomain_training.catalog import interior_labels
from benchmarks.v5.multidomain_training.click_policy import _regions, _pick, POLICY
from benchmarks.v5.multidomain_training.scoring import BACKGROUND_TARGETS


def inspect(job):
    row, candidates = job
    labels = np.asarray(Image.open(row['labels']));mpp = np.array(json.loads(row['input_mpp_xy_json']))
    inside = interior_labels(labels,mpp)
    codes = [int(c) for c in np.unique(labels) if c != 255]
    groups = _regions(labels,inside,codes,mpp)
    result = []
    for candidate in candidates:
        target = candidate['target'];positive = [g for g in groups if g['code'] == target]
        tissue = [g for g in groups if g['code'] != target and g['code'] not in BACKGROUND_TARGETS[row['dataset']]]
        background_required = sum(len(g['pixels']) for g in tissue) < 5
        negative = tissue + ([g for g in groups if g['code'] != target and g['code'] in BACKGROUND_TARGETS[row['dataset']]] if background_required else [])
        reason = None
        # Exercise the exact picker at the maximum training budget. Excluding
        # infeasible pairs is declared before generating any replacement recipes.
        try:
            _pick(positive,5,np.random.default_rng(20261026),mpp,labels.shape)
            _pick(negative,5,np.random.default_rng(20261026),mpp,labels.shape,diverse_labels=True)
        except ValueError as error:reason = str(error)
        result.append(dict(candidate=candidate,accepted=reason is None,rejection=reason,
            positive_core_pixels=sum(len(g['pixels']) for g in positive),negative_core_pixels=sum(len(g['pixels']) for g in negative)))
    return result


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);folds = {}
    for fold in FOLDS:
        old = json.loads((a.audit/f'{fold}_candidates.json').read_text())
        catalog = a.protocols/fold/'catalog'
        rows = pd.read_parquet(catalog/'tiles.parquet').to_dict('records')
        by_tile = {}
        for candidate in old:by_tile.setdefault(candidate['tile_id'],[]).append(candidate)
        jobs = [(r,by_tile[r['tile_id']]) for r in rows if r['tile_id'] in by_tile]
        records = []
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for index, result in enumerate(pool.map(inspect,jobs,chunksize=4),1):
                records.extend(result)
                if index%100 == 0:write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),fold=fold,fields=index,total_fields=len(jobs)))
        accepted = [r['candidate'] for r in records if r['accepted']]
        write(a.output/f'{fold}_candidates.json',accepted);write(a.output/f'{fold}_geometry.json',records)
        counts = Counter((c['dataset'],c['split']) for c in accepted)
        if set(counts) != {('crc','train'),('crc','val'),('tiger','train'),('tiger','val')}:raise ValueError('domain/split lost after geometry audit')
        folds[fold] = dict(candidates_sha256=digest(a.output/f'{fold}_candidates.json'),original_candidates=len(old),accepted_candidates=len(accepted),
            rejected_candidates=len(old)-len(accepted),rejection_reasons=dict(Counter(r['rejection'] for r in records if not r['accepted'])),
            accepted_counts={d+'|'+s:n for (d,s),n in counts.items()},masked_catalog_sha256=digest(catalog/'tiles.parquet'))
        print(json.dumps(dict(fold=fold,**folds[fold])),flush=True)
    write(a.output/'summary.json',dict(status='complete_feasibility_audit',worker_pid=os.getpid(),folds=folds,
        source_sha256=digest(__file__),parent_ontology_audit=str(a.audit),parent_audit_sha256=digest(a.audit/'summary.json'),
        point_policy=POLICY,maximum_training_budget='5+5',scope='Full candidate geometric feasibility before any new recipes; unchanged picker/margins; not accuracy-driven selection'))
    write(a.output/'status.json',dict(status='complete_feasibility_audit',worker_pid=os.getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('audit','protocols','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);run(p.parse_args())
