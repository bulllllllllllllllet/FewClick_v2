"""Freeze prediction-independent sampling and unchanged training inputs."""
import argparse
from collections import defaultdict, deque
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
OLD = BASE/'hg_paper_ablation_envcomplete_1113_231852'
WSI = BASE/'whole_wsi_current_1112_171010'


def order(value):
    return hashlib.sha256(('reviewer_sample_v1|'+str(value)).encode()).hexdigest()


def choose(tasks, count):
    groups = defaultdict(lambda: defaultdict(list))
    for t in tasks: groups[t['query_space']][t['support_id']].append(t)
    per_slide = {}
    for slide, supports in groups.items():
        queues = [deque(sorted(v, key=lambda t: order(t['index']))) for k, v in sorted(supports.items(), key=lambda kv: order(kv[0]))]
        rolled = []
        while any(queues):
            for q in queues:
                if q: rolled.append(q.popleft())
        per_slide[slide] = deque(rolled)
    result = []
    queues = [q for k, q in sorted(per_slide.items(), key=lambda kv: order(kv[0]))]
    while len(result) < count and any(queues):
        for q in queues:
            if q and len(result) < count: result.append(q.popleft())
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    r = a.root; r.mkdir(exist_ok=False)
    c = json.loads((OLD/'contract.json').read_text())
    for k in ['training_consumption', 'functional_source', 'monitor_source']:
        assert sha(c[k]) == c[k+'_sha256']
    for f, h in c['weights'].items(): assert sha(f) == h
    source = Path(c['evaluation_source']); original = json.loads((source/'tasks.json').read_text())
    ec = json.loads((source/'contract.json').read_text()); assert sha(source/'tasks.json') == ec['tasks_sha256']
    rows = json.loads((source/'query_rows.json').read_text()); assert sha(source/'query_rows.json') == ec['query_rows_sha256']
    strata = defaultdict(list); lookup = {}; label_counts = {}; label_hashes = {}
    for t in original:
        identity = t['query_id']
        if identity not in label_counts:
            label = np.asarray(Image.open(rows[identity]['labels']))
            digest = hashlib.sha256(label.tobytes()).hexdigest()
            assert digest == rows[identity]['labels_array_sha256']
            values, counts = np.unique(label, return_counts=True)
            label_counts[identity] = {str(int(k)): int(v) for k, v in zip(values, counts)}
            label_hashes[identity] = digest
        counts = label_counts[identity]
        valid = sum(v for k, v in counts.items() if int(k) != 255)
        assert counts.get(str(t['target']), 0) == t['query_positive_pixels'] and valid > 0
        kind = 'absent' if t['query_positive_pixels'] == 0 else ('small' if t['query_positive_pixels']/valid < .2 else 'large')
        strata[(t['target'], kind)].append(t); lookup[str(t['index'])] = dict(target=t['target'], stratum=kind, valid=valid)
    selected = []; coverage = {}
    for target in sorted({t['target'] for t in original}):
        coverage[target] = {}
        for kind, limit in [('small', 50), ('large', 25), ('absent', 25)]:
            picked = choose(strata[target, kind], limit); selected.extend(picked)
            coverage[target][kind] = dict(population=len(strata[target, kind]), selected=len(picked),
                                          slides=len({t['query_space'] for t in picked}))
    selected.sort(key=lambda t: t['index']); assert len({t['index'] for t in selected}) == len(selected)
    assert len(coverage) == 11 and len(selected) <= 1100
    sample = r/'sample'; sample.mkdir(); dump(sample/'tasks.json', selected)
    shutil.copy2(source/'query_rows.json', sample/'query_rows.json')
    dump(sample/'sampling.json', dict(status='frozen_before_new_predictions', salt='reviewer_sample_v1',
         rule='stratified deterministic hash, round robin slides and supports; no model scores',
         parent_tasks_sha256=ec['tasks_sha256'], strata=coverage,
         task_strata={str(t['index']): lookup[str(t['index'])] for t in selected},
         note='Class-equal summary; small/large reweighted by original within-class counts. Spatial-balanced diagnostic subset, not a new blind test.'))
    dump(sample/'label_audit.json', dict(source='actual PNG arrays, not legacy template counts_json', array_sha256=label_hashes))
    references = {'HG_mix': BASE/'hg_combination_1112_115539/HG_mix/evaluation',
                  'sansa': BASE/'same_slide_retrain_1110_204945/sansa/evaluation',
                  'M': OLD/'M/evaluation', 'no_cross_scale': OLD/'no_cross_scale/evaluation'}
    ref_ck = {}
    for method, folder in references.items():
        conf = json.loads((folder/'contract.json').read_text()); acc = json.loads((folder/'acceptance.json').read_text())
        assert acc['status'] == 'complete_accepted' and conf['tasks_sha256'] == ec['tasks_sha256']
        ref_ck[method] = conf['checkpoints'][method]
    cases = json.loads((WSI/'cases.json').read_text())[:3]
    assert len(cases) == 3 and all(t['eligible'] and not any(t['membership'].values()) for t in cases)
    dump(r/'cases.json', cases)
    files = set(c['sources']) | {str(f) for f in Path('benchmarks/v5/reviewer_sample').glob('*.py')}
    c.update(arms=['proto_ms', 'fixed_grid'], gpus={'proto_ms': 3, 'fixed_grid': 5},
             sources={f: sha(f) for f in sorted(files)}, status='preregistered_authorized',
             created_at=datetime.datetime.now().astimezone().isoformat(),
             architecture='Proto-MS seven-call signed prototypes + light CNN; fixed_grid freezes zero content-assignment logits only',
             scope='Matched training; frozen stratified patch sample and first3 full original WSI; no other new arms/seeds',
             sample=str(sample), sample_tasks_sha256=sha(sample/'tasks.json'), sample_rows_sha256=sha(sample/'query_rows.json'),
             sample_manifest_sha256=sha(sample/'sampling.json'), sample_count=len(selected), cases_sha256=sha(r/'cases.json'),
             reference_evaluations={k: str(v) for k, v in references.items()}, reference_checkpoints=ref_ck,
             original_wsi_root=str(WSI), functional_gate=c['functional_gate'],
             extra_functional_gate='fixed_grid: active dense transfer residual; frozen assignments. proto_ms: true prompt response',
             resource_override='Authorized2026-11-16 two new training workers; max6 total GPU workers, all6 share>=12288MiB',
             recipe_diff='No training budget/loss change except Proto-MS has no assignment regularizer; fixed_grid regularizer constant',
             repository_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    dump(r/'contract.json', c)
    for arm in c['arms']: (r/arm).mkdir()
    (r/'tmp').mkdir()
    for f in c['sources']:
        dest = r/'source_snapshot'/f; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dest)
        assert sha(dest) == c['sources'][f]
    dump(r/'preparation.json', dict(status='accepted', tasks=len(selected), coverage=coverage,
         cases=[t['wsi_id'] for t in cases], training_sha256=c['training_consumption_sha256']))
    print(json.dumps(dict(root=str(r), tasks=len(selected), strata=coverage, cases=[t['wsi_id'] for t in cases])), flush=True)


if __name__ == '__main__': main()
