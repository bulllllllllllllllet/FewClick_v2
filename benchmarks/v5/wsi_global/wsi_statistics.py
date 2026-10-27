"""WSI-cluster summaries on accepted complete remote WSI predictions only."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from .completion import atomic_json

BUDGETS = ('3+2', '1+1', '3+1', '5+1')
METHODS = {'fewclick_m3d_image': ('graph_dense', 'region_prototype'),
           'fewclick_m3d_xcell': ('graph_dense', 'region_prototype'),
           'sam_ft': ('feature_prototype',), 'wsi_sam_ft': ('feature_prototype',)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def average(values):
    values = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), 'finite eligible metric values')
    return math.fsum(values) / len(values) if values else None


def measures(row):
    scores = row['scores']; out = {}
    for name in ('standard', 'unprompted'):
        for metric in ('dice', 'recall', 'precision', 'false_positive_fraction', 'false_positive_rate'):
            out[name + '.' + metric] = scores[name][metric]
    out['absent_target.false_positive_rate'] = scores['standard']['false_positive_rate'] if not scores['standard']['positive'] else None
    c = scores['components']
    out['components.mean_area_recall'] = c['mean_area_recall']
    for cutoff in (50, 10):
        out['components.detected_at_' + str(cutoff)] = c['detected_at_' + str(cutoff)] / c['count'] if c['count'] else None
    for part in scores['distance']:
        for metric in ('dice', 'recall', 'false_positive_rate', 'false_positive_fraction'):
            out['distance.' + part['bin'] + '.' + metric] = part[metric]
    out['standard.bf1_um2'] = row['physical_BF1']['bf1_um2']
    return out


def eligibility(row):
    s = row['scores']; b = row['physical_BF1']
    return dict(standard=[s['standard'][k] for k in ('positive', 'negative', 'valid')],
                unprompted=[s['unprompted'][k] for k in ('positive', 'negative', 'valid')],
                censored=s['censored_target_pixels'], components=s['components']['count'],
                distance=[(d['bin'], d['positive'], d['negative'], d['valid']) for d in s['distance']],
                boundary=[b['true_boundary_pixels'], b['boundary_evaluable'], b['GT_mpp_xy']],
                positive=row['positive_native_xy'], negative=row['negative_native_xy'])


def summarize(arms, repetitions=2000, seed=20261028):
    """Average prompts -> classes -> WSIs; budgets remain separate."""
    identities = None; eligibility_by_key = None; metric_names = None
    for name, rows in arms.items():
        keys = [(r['item'], r['wsi_id'], r['target'], r['prompt_group'], r['budget']) for r in rows]
        require(len(keys) == len(set(keys)), 'unique task identities per arm')
        require(identities is None or set(keys) == identities, 'exact paired task identity coverage')
        identities = set(keys)
        current = {k: eligibility(r) for k, r in zip(keys, rows)}
        require(eligibility_by_key is None or eligibility_by_key == current, 'identical GT/prompt/domain/NA eligibility across methods')
        eligibility_by_key = current
        for r in rows:
            names = set(measures(r))
            require(metric_names is None or names == metric_names, 'fixed identical outcome fields')
            metric_names = names
    require(bool(identities), 'nonempty paired cohort')
    slides = sorted({key[1] for key in identities})
    # The same WSI draws apply to all outcomes, budgets and methods.
    draws = np.random.default_rng(seed).integers(len(slides), size=(repetitions, len(slides)))
    tables = []; arrays = {}; eligibility_masks = {}; cluster_rows = []; class_rows = []
    for name, rows in arms.items():
        for budget in BUDGETS:
            selected = [r for r in rows if r['budget'] == budget]
            if not selected:
                continue
            outcomes = [measures(r) for r in selected]
            for metric in sorted(metric_names):
                pairs = defaultdict(list)
                for r, values in zip(selected, outcomes):
                    pairs[(r['wsi_id'], r['target'])].append(values[metric])
                values = {key: average(v) for key, v in pairs.items()}
                per_slide = {sid: average(v for (s, _), v in values.items() if s == sid) for sid in slides}
                vector = np.array([np.nan if per_slide[s] is None else per_slide[s] for s in slides])
                valid = np.isfinite(vector); mask_key = budget, metric
                require(mask_key not in eligibility_masks or np.array_equal(valid, eligibility_masks[mask_key]), 'same eligible WSI masks across methods')
                eligibility_masks[mask_key] = valid
                boot = np.full(repetitions, np.nan)
                if valid.sum() > 1:
                    sampled = vector[draws]; counts = np.isfinite(sampled).sum(1)
                    np.divide(np.nansum(sampled, axis=1), counts, out=boot, where=counts > 0)
                arrays[(name, budget, metric)] = boot
                eligible_boot = boot[np.isfinite(boot)]
                tables.append(dict(arm=name, budget=budget, metric=metric,
                    WSI_macro=average(per_slide.values()), WSI_class_macro=average(values.values()),
                    CI95=np.quantile(eligible_boot, [.025, .975]).tolist() if len(eligible_boot) else None,
                    total_WSIs=len(slides), eligible_WSIs=int(valid.sum()), total_class_slide_pairs=len(values),
                    eligible_class_slide_pairs=sum(v is not None for v in values.values()),
                    total_prompt_rows=len(selected), eligible_prompt_rows=sum(v[metric] is not None for v in outcomes),
                    bootstrap_valid=len(eligible_boot), bootstrap_undefined=repetitions-len(eligible_boot)))
                for sid, value in per_slide.items():
                    cluster_rows.append(dict(arm=name, budget=budget, metric=metric, wsi_id=sid, value=value))
                for target in sorted({key[1] for key in values}):
                    subset = [v for (sid, t), v in values.items() if t == target]
                    class_rows.append(dict(arm=name, budget=budget, metric=metric, target=target,
                                           value=average(subset), total_WSIs=len(subset), eligible_WSIs=sum(v is not None for v in subset)))
    index = {(t['arm'], t['budget'], t['metric']): t for t in tables}
    paired = []
    for a, b in itertools.combinations(sorted(arms), 2):
        for budget, metric in sorted(eligibility_masks):
            av = index[(a, budget, metric)]['WSI_macro']; bv = index[(b, budget, metric)]['WSI_macro']
            delta = arrays[(b, budget, metric)] - arrays[(a, budget, metric)]
            finite = delta[np.isfinite(delta)]
            paired.append(dict(a=a, b=b, budget=budget, metric=metric,
                               b_minus_a=bv-av if bv is not None and av is not None else None,
                               paired_WSI_CI95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                               eligible_WSIs=index[(a, budget, metric)]['eligible_WSIs'],
                               bootstrap_valid=len(finite), bootstrap_undefined=repetitions-len(finite)))
    return dict(tables=tables, paired=paired, WSI_rows=cluster_rows, class_rows=class_rows,
                bootstrap=dict(seed=seed, repetitions=repetitions, WSI_ids=slides,
                               draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                               hierarchy='Within budget: prompt mean per WSI/class, class mean per WSI, equal WSI mean; WSI resampling paired across arms',
                               scope='Conditional within-cohort WSI sampling variation, not patient/seed/population independence; undefined sparse draws retained as NA counts'))


def accepted_boundary(path, fixture):
    s = read(path / 'summary.json')
    require(s['status'] == ('accepted_WSI_boundary_aggregation_fixture' if fixture else 'accepted_complete_WSI_physical_BF1_rows'), 'complete independently accepted BF1 aggregate')
    require(s['rows_sha256'] == sha(path / 'rows.json') and s['source_sha256'] == sha('benchmarks/v5/wsi_global/aggregate_wsi_boundary.py'), 'bound accepted rows and aggregator source')
    for e in s['evidence']:
        require(sha(Path(e['source'])/'summary.json') == e['source_summary_sha256'] and sha(Path(e['acceptance'])/'summary.json') == e['acceptance_summary_sha256'], 'all independent BF1 shard receipts immutable')
    return s, read(path / 'rows.json')


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    arms = {}; records = []; common = None
    plans = read(a.root/'support_query_full_1027_173800/support_tasks_maximum.json')
    expected = {(p['wsi_id'], p['wsi_id'], p['target'], p['prompt_group'], b) for p in plans for b in BUDGETS}
    for path in a.inputs:
        s, rows = accepted_boundary(path, a.fixture)
        require(s['signature']['kind'] == 'interactive', 'interactive tables separated from privileged closed reference')
        methods = {r['method'] for r in rows}; require(len(methods) == 1, 'one original method per source')
        method = next(iter(methods)); require(method in METHODS, 'fixed predeclared method')
        require({r['variant'] for r in rows} == set(METHODS[method]), 'all predeclared variants retained')
        for variant in METHODS[method]:
            key = method + '/' + variant; require(key not in arms, 'no duplicate method input')
            arm = [r for r in rows if r['variant'] == variant]
            if not a.fixture:
                require({(r['item'], r['wsi_id'], r['target'], r['prompt_group'], r['budget']) for r in arm} == expected and len(arm) == 5520, 'entire fixed276x5x4 remote task set')
            arms[key] = arm
        common = {(r['item'], r['wsi_id'], r['target']) for r in rows}
        records.append(dict(path=str(path), summary_sha256=sha(path/'summary.json'), prediction_summary_sha256=s['prediction_summary_sha256']))
    if not a.fixture:
        require(set(arms) == {m+'/'+v for m in METHODS for v in METHODS[m]}, 'all six interactive variants required')
    closed_tables = []
    if a.closed is not None:
        s, closed = accepted_boundary(a.closed, a.fixture)
        require(s['signature']['kind'] == 'closed12', 'privileged closed reference separately recorded')
        keys = [(r['item'], r['wsi_id'], r['target']) for r in closed]
        require(len(keys) == len(set(keys)), 'closed predictions counted once without prompt replication')
        if not a.fixture:
            require(len(closed) == 360 and len({r['wsi_id'] for r in closed}) == 30 and {r['target'] for r in closed} == set(range(12)), 'complete330 tissue plus30 background reference')
        for scope in ('full_tissue', 'common_support_available', 'background'):
            selected = [r for r in closed if (r['target'] == 2 if scope == 'background' else r['target'] != 2 and (scope == 'full_tissue' or (r['item'], r['wsi_id'], r['target']) in common))]
            require(bool(selected), 'nonempty predeclared closed scope')
            if not a.fixture:
                require(len(selected) == {'full_tissue':330, 'common_support_available':276, 'background':30}[scope], 'fixed closed coverage denominator')
            for metric in ('dice', 'recall', 'false_positive_fraction', 'bf1_um2'):
                by_slide = defaultdict(list)
                for r in selected:
                    by_slide[r['wsi_id']].append(r['physical_BF1']['bf1_um2'] if metric == 'bf1_um2' else r[metric])
                per_slide = {sid:average(v) for sid,v in by_slide.items()}
                closed_tables.append(dict(scope=scope, metric=metric, WSI_macro=average(per_slide.values()), rows=len(selected), eligible_rows=sum(v is not None for vs in by_slide.values() for v in vs), per_WSI=per_slide,
                                          supervision='Privileged 12class CE and target ID; source prompts not model inputs; unique standard predictions'))
        records.append(dict(path=str(a.closed), summary_sha256=sha(a.closed/'summary.json')))
    require(a.fixture or a.closed is not None, 'full330/common276/background closed reference mandatory')
    if not a.fixture:
        require(a.closed_retrieval is not None and a.closed_retrieval_acceptance is not None,
                'full5520 independently accepted closed component/distance reference required')
        rs = read(a.closed_retrieval/'summary.json'); receipt = read(a.closed_retrieval_acceptance/'summary.json')
        require(rs['status'] == 'complete_closed12_full30_retrieval_unaccepted'
                and receipt['status'] == 'accepted_closed12_full30_component_distance_scores'
                and receipt['source_summary_sha256'] == sha(a.closed_retrieval/'summary.json')
                and rs['rows_sha256'] == sha(a.closed_retrieval/'rows.json'), 'all closed reference scoring rows independently bound')
        inputs = read(a.closed_retrieval/'contract.json')['inputs']
        for p, digest in inputs.items():
            require(sha(p) == digest, 'closed component/scoring data and native-prediction receipts unchanged')
        require(s['prediction_summary_sha256'] in inputs.values(), 'closed BF1 and retrieval use same accepted native predictions')
        by_key = {(r['item'], r['wsi_id'], r['target']): r for r in closed}
        reference = read(a.closed_retrieval/'rows.json')
        require(len(reference) == 5520, 'complete closed reference task rows')
        for row in reference:
            native = by_key[(row['item'], row['wsi_id'], row['target'])]
            require(all(row['scores']['standard'][k] == native[k] for k in ('tp', 'fp', 'fn', 'positive', 'valid', 'dice', 'recall')), 'closed standard predictions constant across prompt repetitions')
            row['physical_BF1'] = native['physical_BF1']
        require({(r['item'], r['wsi_id'], r['target'], r['prompt_group'], r['budget']) for r in reference} == expected,
                'closed common276 scoring rows match complete interactive tasks')
        arms['closed12_supervised_reference/privileged_reference'] = reference
        records.append(dict(path=str(a.closed_retrieval), summary_sha256=sha(a.closed_retrieval/'summary.json'),
                            acceptance=str(a.closed_retrieval_acceptance), acceptance_sha256=sha(a.closed_retrieval_acceptance/'summary.json')))
    result = summarize(arms)
    for contrast in result['paired']:
        contrast['includes_privileged_reference'] = any(n.startswith('closed12_') for n in (contrast['a'], contrast['b']))
    for name in ('tables', 'paired', 'WSI_rows', 'class_rows'):
        atomic_json(a.output/(name+'.json'), result[name])
    atomic_json(a.output/'closed_reference_unique_predictions.json', closed_tables)
    summary = dict(status='passed_WSI_statistics_fixture' if a.fixture else 'complete_interactive_WSI_statistics_closed_standard_reference_unaccepted', worker_pid=os.getpid(),
                   source_sha256=sha(__file__), inputs=records, bootstrap=result['bootstrap'], arms=list(arms),
                   output_sha256={p.name:sha(p) for p in a.output.glob('*.json')},
                   scope='Separate budget curves with3+2 primary; all fixed arms; NA eligibility verified across methods; WSI conditional bootstrap; no multiplicity-adjusted significance claims',
                   remaining='Independent full statistics acceptance; full-valid28452 domain, probability replay qualification and reviewer closure',
                   limits='Closed12 has multiclass CE/classID; repeated prompts change only censor/distance scoring. Custom SAM transfer prototypes, different views/pretraining/capacity; no equal-total-supervision or global graph claim.')
    atomic_json(a.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--closed', type=Path)
    p.add_argument('--closed-retrieval', type=Path)
    p.add_argument('--closed-retrieval-acceptance', type=Path)
    p.add_argument('--fixture', action='store_true')
    run(p.parse_args())
