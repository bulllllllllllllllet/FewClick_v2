"""Independent scalar/count and WSI-weight bootstrap recount; no producer import."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean

import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))


def mean(values):
    values = [v for v in values if v is not None]
    return fmean(values) if values else None


def percentile(values, q):
    position = (len(values)-1)*q
    lo = math.floor(position); hi = math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def interval(values):
    values = sorted(values)
    return [percentile(values, .025), percentile(values, .975)] if values else None


def metric(row, field):
    section, name = field.rsplit('.', 1); s = row['scores']
    if field == 'standard.bf1_um2':
        b = row['physical_BF1']; n = b['true_boundary_pixels']; p = b['predicted_boundary_pixels']
        if not n:
            return None
        precision = b['matched_predicted_boundary_pixels']/p if p else 0.
        recall = b['matched_true_boundary_pixels']/n
        return 2*precision*recall/(precision+recall) if precision+recall else 0.
    if section == 'components':
        c = s['components']
        if name == 'mean_area_recall':
            return c[name]
        return c[name]/c['count'] if c['count'] else None
    if section == 'absent_target':
        if s['standard']['positive']:
            return None
        d = s['standard']
    elif section.startswith('distance.'):
        d = next(v for v in s['distance'] if v['bin'] == section.split('.', 1)[1])
    else:
        d = s[section]
    tp, fp, pos, neg = [d[k] for k in ('tp', 'fp', 'positive', 'negative')]
    if name == 'dice': return 2*tp/(pos+tp+fp) if pos else None
    if name == 'recall': return tp/pos if pos else None
    if name == 'precision': return tp/(tp+fp) if tp+fp else 0.
    if name == 'false_positive_fraction': return fp/(pos+neg) if pos+neg else None
    if name == 'false_positive_rate': return fp/neg if neg else None
    raise ValueError('Unrecognized fixed metric: '+field)


def recount(arms, tables, seed=20261028, repetitions=2000):
    """Independent Python fmean and bootstrap WSI-multiplicity algebra."""
    slides = sorted({r['wsi_id'] for rows in arms.values() for r in rows})
    fields = {section+'.'+metric for section in ('standard','unprompted') for metric in ('dice','recall','precision','false_positive_fraction','false_positive_rate')}
    fields |= {'absent_target.false_positive_rate','components.mean_area_recall','components.detected_at_50','components.detected_at_10','standard.bf1_um2'}
    fields |= {'distance.'+distance+'.'+metric for distance in ('<1mm','1-5mm','>=5mm') for metric in ('dice','recall','false_positive_rate','false_positive_fraction')}
    expected = {(arm,budget,field) for arm in arms for budget in ('3+2','1+1','3+1','5+1') for field in fields}
    require({(t['arm'],t['budget'],t['metric']) for t in tables} == expected and len(tables) == len(expected), 'all fixed27 outcomes and four separate budgets, no omitted tables')
    draws = np.random.default_rng(seed).integers(len(slides), size=(repetitions, len(slides)))
    multisets = [Counter(map(int, d)) for d in draws]
    computed = {}; clusters = {}; classes = {}
    for table in tables:
        arm, budget, name = [table[k] for k in ('arm', 'budget', 'metric')]
        rows = [r for r in arms[arm] if r['budget'] == budget]
        pairs = defaultdict(list)
        for row in rows:
            pairs[(row['wsi_id'], row['target'])].append(metric(row, name))
        pair_values = {key:mean(v) for key,v in pairs.items()}
        by_slide = {sid:mean(v for (s,t),v in pair_values.items() if s == sid) for sid in slides}
        values = [by_slide[s] for s in slides]
        bootstrap = []
        if sum(v is not None for v in values) > 1:
            for sample in multisets:
                items = [(values[i], count) for i,count in sample.items() if values[i] is not None]
                bootstrap.append(math.fsum(value*count for value,count in items)/sum(count for value,count in items) if items else None)
        else:
            bootstrap = [None]*repetitions
        eligible = [v for v in bootstrap if v is not None]
        checks = dict(WSI_macro=mean(values), WSI_class_macro=mean(pair_values.values()),
                      total_WSIs=len(slides), eligible_WSIs=sum(v is not None for v in values),
                      total_class_slide_pairs=len(pair_values), eligible_class_slide_pairs=sum(v is not None for v in pair_values.values()),
                      total_prompt_rows=len(rows), eligible_prompt_rows=sum(v is not None for vs in pairs.values() for v in vs),
                      bootstrap_valid=len(eligible), bootstrap_undefined=repetitions-len(eligible))
        require(all(close(table[k], v) for k,v in checks.items()), 'every independent mean/eligibility/NA/WSI denominator')
        ci = interval(eligible)
        require(table['CI95'] is None if ci is None else table['CI95'] is not None and all(close(a,b) for a,b in zip(table['CI95'],ci)), 'independent WSI-multiplicity percentile interval')
        computed[(arm,budget,name)] = dict(point=checks['WSI_macro'], bootstrap=bootstrap)
        for sid,value in by_slide.items(): clusters[(arm,budget,name,sid)] = value
        for target in {key[1] for key in pair_values}:
            vs = [v for (s,t),v in pair_values.items() if t == target]
            classes[(arm,budget,name,target)] = dict(value=mean(vs),total_WSIs=len(vs),eligible_WSIs=sum(v is not None for v in vs))
    require(len(computed) == len(tables), 'unique complete method/budget/metric summary identities')
    return computed, clusters, classes


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    s = read(a.source/'summary.json'); fixture = s['status'] == 'passed_full_valid_WSI_statistics_fixture'
    require(fixture or s['status'] == 'complete_full_valid_WSI_statistics_unaccepted', 'complete statistical producer output')
    require(s['source_sha256'] == sha('benchmarks/v5/wsi_global/full_valid_statistics.py'), 'frozen statistics producer')
    require(s['scope'] == 'complete_valid4x_including_support_separate_diagnostic' and s['statistical_core_sha256'] == sha('benchmarks/v5/wsi_global/wsi_statistics.py'), 'separate full-valid domain and frozen statistical core')
    if not fixture:
        gate = read(a.verified_gate/'summary.json') if a.verified_gate else {}
        require(gate.get('status') == 'accepted_full_valid_WSI_statistics_fixture' and gate.get('source_sha256') == sha(__file__), 'same independent statistical auditor fixture')
    for name,digest in s['output_sha256'].items(): require(sha(a.source/name) == digest, 'all original report output bytes')
    arms = {}; closed = None
    for record in s['inputs']:
        path = Path(record['path']); require(sha(path/'summary.json') == record['summary_sha256'], 'immutable accepted input receipt')
        meta = read(path/'summary.json'); rows = read(path/'rows.json'); require(sha(path/'rows.json') == meta['rows_sha256'], 'original accepted input rows')
        require(meta['status'] == ('accepted_full_valid_boundary_aggregation_gate' if fixture else 'accepted_complete_full_valid_physical_BF1_rows') and meta['signature']['evaluation_scope'] == s['scope'], 'actual accepted full-valid input, not remote')
        if meta['signature']['kind'] == 'closed12':
            closed = rows
        else:
            for row in rows:
                arms.setdefault(row['method']+'/'+row['variant'], []).append(row)
    if not fixture:
        expected_arms = {m+'/'+v for m, vs in [('fewclick_m3d_image',('graph_dense','region_prototype')),('fewclick_m3d_xcell',('graph_dense','region_prototype')),('sam_ft',('feature_prototype',)),('wsi_sam_ft',('feature_prototype',))] for v in vs}
        require(set(arms) == expected_arms and closed is not None, 'all six interactive arms and unique closed reference')
    paired_reference = None
    for rows in arms.values():
        current = {}
        for r in rows:
            require(sha(r['prediction']) == r['prediction_sha256'], 'actual immutable prediction metadata')
            p = read(r['prediction']); v = r['scores']; b = r['physical_BF1']
            key = r['item'],r['wsi_id'],r['target'],r['prompt_group'],r['budget']
            current[key] = dict(bounds=p['GT_cell_bounds'],shape=p['GT_shape'],positive=r['positive_native_xy'],negative=r['negative_native_xy'],
                standard=[v['standard'][k] for k in ('positive','negative','valid')],unprompted=[v['unprompted'][k] for k in ('positive','negative','valid')],
                censored=v['censored_target_pixels'],components=v['components']['count'],distance=[(d['bin'],d['positive'],d['negative'],d['valid']) for d in v['distance']],
                boundary=[b['true_boundary_pixels'],b['boundary_evaluable'],b['GT_mpp_xy']])
        require(paired_reference is None or paired_reference == current, 'independent physical-domain/prompt/GT/NA pairing')
        paired_reference = current
    tables = read(a.source/'tables.json'); computed, clusters, classes = recount(arms, tables)
    slides = sorted({r['wsi_id'] for rows in arms.values() for r in rows})
    draws = np.random.default_rng(20261028).integers(len(slides),size=(2000,len(slides)))
    require(s['bootstrap']['seed'] == 20261028 and s['bootstrap']['repetitions'] == 2000 and s['bootstrap']['WSI_ids'] == slides and s['bootstrap']['draw_sha256'] == hashlib.sha256(draws.tobytes()).hexdigest(), 'exact fixed WSI resampling recipe, no prompt/target/patient replacement')
    actual = read(a.source/'WSI_rows.json')
    require(len(actual) == len(clusters) and len({(r['arm'],r['budget'],r['metric'],r['wsi_id']) for r in actual}) == len(actual), 'complete unique per-WSI report rows')
    require(all(close(r['value'], clusters[(r['arm'],r['budget'],r['metric'],r['wsi_id'])]) for r in actual), 'all independent WSI means')
    actual = read(a.source/'class_rows.json')
    require(len(actual) == len(classes) and len({(r['arm'],r['budget'],r['metric'],r['target']) for r in actual}) == len(actual), 'complete unique per-class report rows')
    require(all(all(close(r[k],v) for k,v in classes[(r['arm'],r['budget'],r['metric'],r['target'])].items()) for r in actual), 'all independent per-class means/counts')
    paired = read(a.source/'paired.json'); seen = set()
    for p in paired:
        key = p['a'],p['b'],p['budget'],p['metric']; require(key not in seen, 'unique paired contrast'); seen.add(key)
        av = computed[(p['a'],p['budget'],p['metric'])]; bv = computed[(p['b'],p['budget'],p['metric'])]
        delta = [b-a for a,b in zip(av['bootstrap'],bv['bootstrap']) if a is not None and b is not None]
        expected = interval(delta)
        require(close(p['b_minus_a'],bv['point']-av['point'] if av['point'] is not None and bv['point'] is not None else None), 'paired contrast sign/point')
        require(p['paired_WSI_CI95'] is None if expected is None else p['paired_WSI_CI95'] is not None and all(close(a,b) for a,b in zip(p['paired_WSI_CI95'],expected)), 'independent paired percentile difference interval')
        require(p['bootstrap_valid'] == len(delta) and p['bootstrap_undefined'] == 2000-len(delta), 'all paired bootstrap NA counts')
        require(p['eligible_WSIs'] == sum(value is not None for (arm,budget,metric,sid),value in clusters.items() if (arm,budget,metric) == (p['a'],p['budget'],p['metric'])), 'paired true eligible WSI denominator')
    names = sorted(arms); outcomes = {(t['budget'],t['metric']) for t in tables}
    require(seen == {(a,b,budget,m) for i,a in enumerate(names) for b in names[i+1:] for budget,m in outcomes}, 'all predeclared pairwise contrasts present')
    common = {(r['item'],r['wsi_id'],r['target']) for r in next(iter(arms.values()))}
    for t in read(a.source/'closed_reference_unique_predictions.json'):
        selected = [r for r in closed if (r['target'] == 2 if t['scope'] == 'background' else r['target'] != 2 and (t['scope'] == 'full_tissue' or (r['item'],r['wsi_id'],r['target']) in common))]
        by_slide = defaultdict(list)
        for r in selected: by_slide[r['wsi_id']].append(r['physical_BF1']['bf1_um2'] if t['metric'] == 'bf1_um2' else r[t['metric']])
        means = {sid:mean(v) for sid,v in by_slide.items()}
        require(close(t['WSI_macro'],mean(means.values())) and t['rows'] == len(selected) and t['eligible_rows'] == sum(v is not None for vs in by_slide.values() for v in vs) and all(close(means[sid],v) for sid,v in t['per_WSI'].items()), 'all unique closed330/common276/background standard summaries')
    result = dict(status='accepted_full_valid_WSI_statistics_fixture' if fixture else 'accepted_full_valid_WSI_statistics_qualified',worker_pid=os.getpid(),source_sha256=sha(__file__),producer_source_sha256=s['source_sha256'],source_summary_sha256=sha(a.source/'summary.json'),tables=len(tables),paired=len(paired),
                  scope='Independent count-derived outcomes, Python prompt/class/WSI means, WSI multiplicity bootstrap and manual linear percentiles; all budgets, classes, WSI/NA counts, paired contrasts, closed reference counted uniquely. WSI dependence/priors/qualified prediction limits persist.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
