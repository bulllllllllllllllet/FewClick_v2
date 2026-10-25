"""Audit complete remote evaluations, preserving their annotation budgets."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.evidence_closure.audit_external import require, close
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.metrics import VERSION, boundary, score
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.paper_multiscale.data import intersects


def read(path):
    return json.loads(Path(path).read_text())


def identity(row):
    return row['pair_id'], row['prompt_group'], row['click_budget']


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'status.json', dict(status='running', worker_pid=os.getpid(), stage='inputs'))
    tasks = root/'tasks_1025_132300'; contract = read(tasks/'contract.json')
    protocol_sha = digest(tasks/'contract.json'); catalog = Path(contract['catalog'])
    require(digest(catalog/'tiles.parquet') == contract['catalog_sha256'], 'catalog identity')
    require(digest(tasks/'validation_maximum.json') == contract['tasks_sha256'], 'task identity')
    fields = {r['tile_id']: r for r in pd.read_parquet(catalog/'tiles.parquet').to_dict('records')}
    plans = read(tasks/'validation_maximum.json')
    pairs = {p['pair_id']: p for p in plans if p['prompt_group'] == 0}
    expected = {identity(p): p for maximum in plans for p in
                [nested(maximum, b) for b in contract['report_budgets']]}
    require(len(expected) == len(plans)*len(contract['report_budgets']), 'planned duplicate')
    local_tasks = Path(contract['local_tasks']); local_contract = read(local_tasks/'contract.json')
    require(digest(local_tasks/'contract.json') == contract['local_protocol_sha256'] and
            digest(local_tasks/'validation_maximum.json') == contract['local_points_sha256'], 'source local protocol')
    local = {(p['pair_id'], p['prompt_group']): p for p in read(local_tasks/'validation_maximum.json')}
    labels = {}; gt_stats = {}; eligible = set(); partial = defaultdict(int)
    used = {p[k] for p in plans for k in ('support_id', 'query_id')}
    for tile in sorted(used):
        row = fields[tile]; gt = np.asarray(Image.open(row['labels']))
        require(hashlib.sha256(gt.tobytes()).hexdigest() == row['labels_array_sha256'], 'actual GT SHA')
        labels[tile] = gt
    for plan in plans:
        s = fields[plan['support_id']]; q = fields[plan['query_id']]
        require(s['split'] == q['split'] == 'val' and s['tile_id'] != q['tile_id'] and
                s['slide_verified'] and q['slide_verified'] and s['dataset'] == q['dataset'] == plan['dataset'], 'remote provenance')
        same = s['space_key'] == q['space_key']
        require(same == (plan['protocol'] == 'same_wsi'), 'WSI relation')
        if same:
            require(not intersects(json.loads(s['multiscale_json'])['context_um'],
                                   json.loads(q['multiscale_json'])['context_um']), 'source query context overlap')
        require(plan['source_space'] == s['space_key'] and plan['query_space'] == q['space_key'], 'WSI IDs')
        original = local[plan['support_pair_id'], plan['prompt_group']]
        for key in ('support_id', 'target', 'positive_xy', 'negative_xy', 'click_seed', 'click_audit'):
            require(plan[key] == original[key], 'source local click equality')
        gt = labels[plan['support_id']]
        for sign in ('positive', 'negative'):
            xy = np.asarray(plan[sign+'_xy']); require(((xy > 0) & (xy < 1)).all(), 'point bounds')
            pixel = np.floor(xy*np.array(gt.shape[::-1])).astype(int); value = gt[pixel[:, 1], pixel[:, 0]]
            require((value == plan['target']).all() if sign == 'positive' else
                    ((value != plan['target']) & (value != 255)).all(), 'actual source point sign')
        if (gt != 255).all(): eligible.add(plan['pair_id'])
        key = plan['query_id'], plan['target']
        if key not in gt_stats:
            query = labels[key[0]]; valid = query != 255; truth = query == key[1]
            gt_stats[key] = dict(valid=int(valid.sum()), positive=int((truth & valid).sum()),
                                 true_boundary_pixels=int(boundary(truth, valid).sum()))
        positive = gt_stats[key]['positive']
        require(positive == 0 if plan['query_presence'] == 'absent' else positive >= 64, 'presence definition')
        if plan['prompt_group'] == 0 and gt_stats[key]['valid'] < labels[key[0]].size:
            partial[plan['query_presence']] += 1
    context = root/'context_full_1025_152401'
    coverage = read(context/'support_coverage.json')
    require(set(coverage['full_GT_eligible_pair_ids']) == eligible and
            {p['pair_id'] for p in coverage['excluded']} == set(pairs)-eligible, 'dense reference subset')
    # Identify connected WSI components: shared supports can correlate different queries.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    for p in pairs.values():
        a, b = find(p['source_space']), find(p['query_space']); parent[a] = b
    components = defaultdict(int)
    for p in pairs.values(): components[find(p['source_space'])] += 1
    context_index = read(context/'source_context_index.json'); checked_contexts = set()
    outputs = []
    for name in ('sparse_retained_m3_1025_133833', 'sparse_sam_public_features_1025_133833',
                 'clean_sam_ft_1025_162450', 'clean_fewclick_m3d_image_1025_162450',
                 'clean_fewclick_m3d_xcell_1025_162450', 'context_full_1025_152401'):
        completed = read(root/name/'summary.json')
        require(completed['status'] == 'complete' and not completed['gate'], 'incomplete parent')
        log = (root/(name+'.log')).read_text()
        require(log.rstrip().endswith('EXIT_CODE=0'), 'parent exit evidence')
        for folder in completed['outputs'].values(): outputs.append(Path(folder))
    tables = {}; records = {}; gates_replayed = 0
    for folder in outputs:
        meta = read(folder/'run.json'); result = read(folder/'summary.json'); method = result['method']
        require(method not in tables and meta.get('method', method) == method, 'duplicate/wrong method')
        require(result['status'] == 'complete' and not result['gate'] and meta['metric_version'] == VERSION and
                meta['protocol_sha256'] == result['protocol_sha256'] == protocol_sha, 'run protocol/metric')
        require(result['rows_sha256'] == digest(folder/'rows.jsonl'), 'rows SHA')
        dense = method == 'multiverseg_full_GT_reference'; seen = set(); aggregate = defaultdict(list)
        for line in (folder/'rows.jsonl').open():
            row = json.loads(line); key = identity(row)
            require(key not in seen, 'duplicate saved row'); seen.add(key)
            if dense:
                require(row['pair_id'] in eligible and row['prompt_group'] == -1 and row['click_budget'] == 'full_mask', 'dense row scope')
                plan = {k: v for k, v in pairs[row['pair_id']].items() if k not in ('positive_xy', 'negative_xy', 'click_audit')}
                plan.update(prompt_group=-1, click_budget='full_mask')
                require(row['support_human_points'] == 0 and row['support_dense_annotated_pixels'] == 262144, 'dense annotation ledger')
                mask = (labels[row['support_id']] == row['target']).astype(np.uint8)
                require(hashlib.sha256(mask.tobytes()).hexdigest() == row['source_mask_sha256'], 'dense source mask identity')
            else:
                require(key in expected, 'unplanned sparse task'); plan = expected[key]
            for k, value in plan.items(): require(row[k] == value, 'plan mismatch: '+k)
            q = fields[row['query_id']]
            for k in ('space_key', 'source_id', 'spatial_group_id'): require(row[k] == q[k], 'query provenance')
            require(row['method'] == method and row.get('query_human_clicks', 0) == 0, 'method or query points')
            stats = gt_stats[row['query_id'], row['target']]
            for k, value in stats.items(): require(row[k] == value, 'actual GT denominator: '+k)
            tp, fp, fn = [row[k] for k in ('tp', 'fp', 'fn')]
            require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)) and
                    tp+fn == stats['positive'] and fp <= stats['valid']-stats['positive'], 'impossible counts')
            close(row['dice'], 2*tp/(2*tp+fp+fn) if stats['positive'] else None, 'Dice recount')
            close(row['false_positive_fraction'], fp/stats['valid'], 'FPA recount')
            require(row['boundary_evaluable'] == (stats['true_boundary_pixels'] > 0), 'boundary eligibility')
            if not stats['true_boundary_pixels']: require(row['bf1_um2'] is None, 'undefined BF1')
            else: require(row['bf1_um2'] is not None and 0 <= row['bf1_um2'] <= 1, 'BF1 bounds')
            if stats['true_boundary_pixels'] and row['predicted_boundary_pixels'] == 0: close(row['bf1_um2'], 0., 'empty prediction boundary')
            if method == 'multiverseg_predicted_sparse':
                entry = row['source_context']; path = entry['path']
                require(Path(path).stem in context_index, 'source context index')
                require(entry == context_index[Path(path).stem] and row['source_mask_sha256'] == entry['mask_sha256'] and
                        entry['support_id'] == row['support_id'] and entry['positive_xy'] == row['positive_xy'] and
                        entry['negative_xy'] == row['negative_xy'], 'context points/identity')
                if path not in checked_contexts:
                    require(digest(path) == entry['file_sha256'], 'context file SHA')
                    with np.load(path, allow_pickle=False) as cache:
                        mask = cache['mask']; require(hashlib.sha256(mask.tobytes()).hexdigest() == entry['mask_sha256'], 'context mask SHA')
                    candidate = entry['candidates']; require(candidate['selection_rule'] == 'argmax_predicted_iou' and
                        candidate['selected_index'] == int(np.argmax(candidate['predicted_iou'])), 'GT oracle selection')
                    image = json.loads(fields[entry['support_id']]['variants_json'])['original']
                    require(entry['image_sha256'] == image['image_sha256'] and
                            entry['generator_weight_sha256'] == meta['generator']['checkpoint_sha256'], 'context generator/image')
                    checked_contexts.add(path)
                require(row['support_dense_annotated_pixels'] == 0 and row['support_human_points'] ==
                        len(plan['positive_xy'])+len(plan['negative_xy']), 'sparse annotation ledger')
            aggregate[row['pair_id'], row['click_budget']].append(row)
        wanted = {(p, -1, 'full_mask') for p in eligible} if dense else set(expected)
        require(seen == wanted and len(seen) == result['rows'], 'full row coverage')
        table = {}
        for (pair, budget), group in aggregate.items():
            require(len(group) == (1 if dense else contract['point_groups']), 'point group count')
            first = group[0]; positive = first['positive'] > 0
            mean = lambda values: float(np.mean(values)) if values else None
            table[pair, budget] = dict(pair_id=pair, budget=budget, dataset=first['dataset'],
                protocol=first['protocol'], kind=first['kind'], query_space=first['query_space'],
                positive=positive, distance_um=first['support_query_distance_um'],
                dice=mean([r['dice'] for r in group]) if positive else None,
                bf1=mean([r['bf1_um2'] for r in group if r['bf1_um2'] is not None]) if positive else None,
                precision=mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0. for r in group]) if positive else None,
                recall=mean([r['tp']/(r['tp']+r['fn']) for r in group]) if positive else None,
                absent_FPA=mean([r['false_positive_fraction'] for r in group]) if not positive else None,
                absent_FPA_ge_1percent=mean([r['false_positive_fraction'] >= .01 for r in group]) if not positive else None,
                absent_FPA_ge_5percent=mean([r['false_positive_fraction'] >= .05 for r in group]) if not positive else None)
        tables[method] = table
        records[method] = dict(run=str(folder), rows=len(seen), rows_sha256=result['rows_sha256'],
            metadata_sha256=digest(folder/'run.json'), annotation='full source mask' if dense else 'source points only',
            training='historical J5' if method.startswith('retained_') else 'local-adapted; remote steps0' if meta.get('local_training_steps') else 'public; no target adaptation',
            prior=meta.get('initialization', meta.get('prior', meta.get('context'))))
        # Saved gate replays cover predictions, unlike full row count algebra alone.
        if folder.parent.name.startswith('clean_'):
            gate_parent = root/('gate_clean_'+meta['adapted_method']+'_1025_161448'); gate = gate_parent/folder.name
            gm = read(gate/'run.json'); gr = read(gate_parent/'summary.json')
            require(gr['GT_forbidden_input_API'] and gr['saved_count_replay'] and gr['native_cached_errors'] == [0., 0.], 'clean native gate')
            require(gm['checkpoint_sha256'] == meta['checkpoint_sha256'] and gm['source_sha256'] == meta['source_sha256'], 'clean gate identity')
            require(digest(meta['checkpoint']) == meta['checkpoint_sha256'], 'clean checkpoint SHA')
            for number, line in enumerate((gate/'rows.jsonl').open()):
                row = json.loads(line); gt = labels[row['query_id']]
                probability = np.load(gate/f'prediction_{number:04d}.npy', allow_pickle=False)
                metric = score(probability >= .5, gt == row['target'], gt != 255,
                               json.loads(fields[row['query_id']]['input_mpp_xy_json']))
                for k, value in metric.items(): close(row[k], value, 'clean saved mask metric')
                gates_replayed += 1
        write(output/'status.json', dict(status='running', worker_pid=os.getpid(), audited_method=method,
                                         methods_complete=len(tables), source_contexts_checked=len(checked_contexts)))
        print(json.dumps(dict(stage='audited', method=method, rows=len(seen))), flush=True)
    require(len(checked_contexts) == len(context_index), 'unreferenced/missing source context')
    # Shared draws within each dataset, resampling query WSI rather than rows.
    clusters = sorted({p['query_space'] for p in pairs.values()}); indices = {s: i for i, s in enumerate(clusters)}
    rng = np.random.default_rng(20261025); draws = []
    for dataset in sorted({p['dataset'] for p in pairs.values()}):
        ix = np.array([indices[s] for s in clusters if any(p['dataset'] == dataset and p['query_space'] == s for p in pairs.values())])
        draws.append(rng.choice(ix, size=(2000, len(ix)), replace=True))
    sample = np.concatenate(draws, axis=1)
    def stats(values):
        result = dict(pairs=len(values), present_pairs=sum(v['positive'] for v in values), absent_pairs=sum(not v['positive'] for v in values))
        distributions = {}
        for metric in ('dice', 'bf1', 'precision', 'recall', 'absent_FPA', 'absent_FPA_ge_1percent', 'absent_FPA_ge_5percent'):
            usable = [v for v in values if v[metric] is not None]
            if not usable: result[metric] = None; continue
            sums = np.zeros(len(clusters)); counts = np.zeros(len(clusters))
            for v in usable: sums[indices[v['query_space']]] += v[metric]; counts[indices[v['query_space']]] += 1
            denominator = counts[sample].sum(1); good = denominator > 0
            dist = np.full(2000, np.nan); dist[good] = sums[sample].sum(1)[good]/denominator[good]
            result[metric] = dict(mean=float(np.mean([v[metric] for v in usable])),
                conditional_query_cluster_ci95=np.quantile(dist[good], [.025, .975]).tolist(),
                contributing_query_WSIs=int((counts > 0).sum()), valid_bootstrap_draws=int(good.sum()))
            distributions[metric] = dist
        return result, distributions
    metrics = {}; paired = []; boot = {}
    for method, table in tables.items():
        metrics[method] = {}; boot[method] = {}
        for subset, subset_pairs in (('all_pairs', set(pairs)), ('full_GT_eligible', eligible)):
            metrics[method][subset] = {}; boot[method][subset] = {}
            for budget in sorted({v['budget'] for v in table.values()}):
                values = [v for v in table.values() if v['pair_id'] in subset_pairs and v['budget'] == budget]
                strata = defaultdict(list)
                for v in values:
                    names = ['overall', v['dataset'], v['protocol'], v['kind'], v['dataset']+'|'+v['kind']]
                    if v['distance_um'] is not None:
                        names.append('same_wsi_distance_'+('lt2mm' if v['distance_um'] < 2000 else '2to5mm' if v['distance_um'] < 5000 else 'ge5mm'))
                    for name in names: strata[name].append(v)
                metrics[method][subset][budget] = {}
                for stratum, rows in strata.items():
                    item, distribution = stats(rows); metrics[method][subset][budget][stratum] = item
                    if stratum == 'overall': boot[method][subset][budget] = distribution
    methods = list(tables)
    for i, a in enumerate(methods):
        for b in methods[i+1:]:
            dense_pair = 'multiverseg_full_GT_reference' in (a, b)
            subset = 'full_GT_eligible' if dense_pair else 'all_pairs'
            for budget in contract['report_budgets']:
                ba = 'full_mask' if a == 'multiverseg_full_GT_reference' else budget
                bb = 'full_mask' if b == 'multiverseg_full_GT_reference' else budget
                for metric in ('dice', 'absent_FPA'):
                    da = boot[a][subset][ba][metric]; db = boot[b][subset][bb][metric]; good = np.isfinite(da) & np.isfinite(db)
                    va = metrics[a][subset][ba]['overall'][metric]['mean']; vb = metrics[b][subset][bb]['overall'][metric]['mean']
                    paired.append(dict(a=a, b=b, sparse_budget=budget, a_budget=ba, b_budget=bb, subset=subset,
                        metric=metric, b_minus_a=vb-va, conditional_paired_query_cluster_ci95=np.quantile((db-da)[good], [.025, .975]).tolist()))
    limits = ['Query-WSI bootstrap is exploratory conditional on fixed support tasks and weights; shared supports couple query clusters',
        'No independent-patient, training-seed or population inference', 'GT ignored pixels are not certified negative; FPA valid annotated area only',
        'Full source mask reference uses extra annotation; compare only837 eligible pairs, one reference per pair',
        'Historical J5, local-adapted and public priors/views remain different; no equal-total-training claim',
        'Full predictions not regenerated: full-row checks verify GT denominators/count algebra; saved native gates support mask replay',
        'Feature prototypes are custom diagnostics; no R supervised training; no optimized remote architecture comparison',
        'Sparse point groups are averaged within pair; no source point heatmaps in query; no WSI efficiency claim']
    write(output/'summary.json', dict(status='complete', protocol_sha256=protocol_sha, records=records, metrics=metrics,
        paired_differences=paired, source_query_WSI_components=len(components), component_pair_counts=sorted(components.values()),
        query_WSI_clusters=len(clusters), source_context_masks_audited=len(checked_contexts), saved_clean_gate_masks_replayed=gates_replayed,
        actual_GT_fields=len(labels), partial_query_pairs=dict(partial), full_GT_eligible_pairs=len(eligible),
        source_sha256=digest(__file__), bootstrap_seed=20261025, bootstrap_repetitions=2000, limits=limits))
    text = ['# 支持查询迁移完整比较', '', '固定935对，每对5组支持点，query人工点0；本图适配后的迁移无额外R监督训练。',
        '先对点组平均，再对pair平均。查询WSI聚类配对区间条件于固定支持集，仅作探索性不确定性；不含共享支持依赖或训练seed。', '',
        '| 稀疏支持方法 | 训练条件 | 主3+2 present Dice | absent FPA |', '|---|---|---:|---:|']
    for method in methods:
        if method == 'multiverseg_full_GT_reference': continue
        m = metrics[method]['all_pairs']['3+2']['overall']
        text.append(f"|{method}|{records[method]['training']}|{m['dice']['mean']:.4f}|{m['absent_FPA']['mean']:.4f}|")
    text += ['', '## 同837对的完整源标注参考', '', '| 方法 | 支持标注预算 | present Dice | absent FPA |', '|---|---|---:|---:|']
    for method in methods:
        budget = 'full_mask' if method == 'multiverseg_full_GT_reference' else '3+2'
        m = metrics[method]['full_GT_eligible'][budget]['overall']
        text.append(f"|{method}|{budget}|{m['dice']['mean']:.4f}|{m['absent_FPA']['mean']:.4f}|")
    text += ['', f'支持—查询WSI依赖图只有{len(components)}个连通分量；不能将935对或查询WSI数当作独立总体样本。',
        '所有预算、分域、same/cross、present/absent、预定同片距离分层及配对区间见summary.json。', '', *['- '+v for v in limits]]
    (output/'report.md').write_text('\n'.join(text)+'\n')
    write(output/'status.json', dict(status='complete', worker_pid=os.getpid(), methods=len(methods)))
    print(json.dumps(dict(status='complete', methods=len(methods), source_query_components=len(components), output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try:
        run(a.root, a.output)
    except Exception:
        if a.output.is_dir():
            write(a.output/'failure.json', dict(status='failed', worker_pid=os.getpid(), traceback=traceback.format_exc()))
        raise
