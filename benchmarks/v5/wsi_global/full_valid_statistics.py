"""Separate source-inclusive full-valid summaries, never remote result tables."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from .completion import atomic_json
from .wsi_statistics import BUDGETS, METHODS, average, summarize


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def accepted_boundary(path, fixture):
    s = read(path/'summary.json')
    wanted = 'accepted_full_valid_boundary_aggregation_gate' if fixture else 'accepted_complete_full_valid_physical_BF1_rows'
    require(s['status'] == wanted and s['signature']['evaluation_scope'] == 'complete_valid4x_including_support_separate_diagnostic', 'accepted full-valid scope, no strict-remote substitution')
    require(sha(path/'rows.json') == s['rows_sha256'], 'all original accepted full-valid rows')
    rows = read(path/'rows.json')
    cell = s['signature']['kind'] == 'interactive' and {r['method'] for r in rows} == {'fewclick_m3d_xcell'}
    aggregator = 'aggregate_fullcell_valid_boundary.py' if cell else 'aggregate_full_valid_boundary.py'
    require(s['source_sha256'] == sha(Path('benchmarks/v5/wsi_global')/aggregator), 'same actually accepted full-valid aggregator')
    for e in s['evidence']:
        require(sha(Path(e['source'])/'summary.json') == e['source_summary_sha256']
            and sha(Path(e['acceptance'])/'summary.json') == e['acceptance_summary_sha256'], 'independent physical BF1 receipts retained')
    return s, rows


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    arms, records = {}, []
    plans = read(a.root/'support_query_full_1027_173800/support_tasks_maximum.json')
    expected = {(p['wsi_id'],p['wsi_id'],p['target'],p['prompt_group'],b) for p in plans for b in BUDGETS}
    common = None
    for path in a.inputs:
        s, rows = accepted_boundary(path, a.fixture)
        require(s['signature']['kind'] == 'interactive', 'privileged reference is a separate source')
        methods = {r['method'] for r in rows}
        require(len(methods) == 1 and next(iter(methods)) in METHODS, 'fixed supported interactive method')
        method = next(iter(methods))
        require({r['variant'] for r in rows} == set(METHODS[method]), 'all original variants, no result-based omission')
        for variant in METHODS[method]:
            name = method+'/'+variant
            require(name not in arms, 'unique original method inputs')
            arm = [r for r in rows if r['variant'] == variant]
            if not a.fixture:
                require(len(arm) == 5520 and {(r['item'],r['wsi_id'],r['target'],r['prompt_group'],r['budget']) for r in arm} == expected, 'entire fixed supported276x5x4 full-valid task set')
            arms[name] = arm
        current = {(r['item'],r['wsi_id'],r['target']) for r in rows}
        require(common is None or common == current, 'same complete support availability across methods')
        common = current
        records.append(dict(path=str(path),summary_sha256=sha(path/'summary.json'),prediction_summary_sha256=s['prediction_summary_sha256']))
    require(a.fixture or set(arms) == {m+'/'+v for m in METHODS for v in METHODS[m]}, 'all six predeclared full interactive arms')
    reference_domains = None
    for rows in arms.values():
        domains = {}
        for row in rows:
            require(sha(row['prediction']) == row['prediction_sha256'], 'accepted prediction metadata bound')
            prediction = read(row['prediction'])
            key = row['item'],row['wsi_id'],row['target'],row['prompt_group'],row['budget']
            domains[key] = prediction['GT_cell_bounds'],prediction['GT_shape']
        require(reference_domains is None or domains == reference_domains, 'same physical query/GT extent across every arm, even empty targets')
        reference_domains = domains
    closed_tables = []
    if a.closed is not None:
        s, rows = accepted_boundary(a.closed, a.fixture)
        require(s['signature']['kind'] == 'closed12', 'actual privileged closed reference')
        keys = [(r['item'],r['wsi_id'],r['target']) for r in rows]
        require(len(keys) == len(set(keys)), 'closed predictions not replicated as prompt samples')
        if not a.fixture:
            require(len(rows) == 360 and len({r['wsi_id'] for r in rows}) == 30, 'complete closed330 tissue plus30 background')
        for scope in ('full_tissue','common_support_available','background'):
            selected = [r for r in rows if (r['target']==2 if scope=='background' else r['target']!=2
                and (scope=='full_tissue' or (r['item'],r['wsi_id'],r['target']) in common))]
            require(bool(selected), 'nonempty fixed closed reference scope')
            if not a.fixture:
                require(len(selected) == {'full_tissue':330,'common_support_available':276,'background':30}[scope], 'all fixed reference denominators')
            for metric in ('dice','recall','false_positive_fraction','bf1_um2'):
                by_slide = defaultdict(list)
                for r in selected:
                    by_slide[r['wsi_id']].append(r['physical_BF1']['bf1_um2'] if metric=='bf1_um2' else r[metric])
                values = {sid:average(v) for sid,v in by_slide.items()}
                closed_tables.append(dict(scope=scope,metric=metric,WSI_macro=average(values.values()),rows=len(selected),
                    eligible_rows=sum(v is not None for vs in by_slide.values() for v in vs),per_WSI=values,
                    supervision='Privileged full-class CE/classID; unique no-click outputs, not equal-information interactive baseline'))
        records.append(dict(path=str(a.closed),summary_sha256=sha(a.closed/'summary.json')))
    require(a.fixture or a.closed is not None, 'full330/common276/background reference mandatory in formal report')
    result = summarize(arms)
    for p in result['paired']:
        p['includes_privileged_reference'] = False
    for name in ('tables','paired','WSI_rows','class_rows'):
        atomic_json(a.output/(name+'.json'),result[name])
    atomic_json(a.output/'closed_reference_unique_predictions.json',closed_tables)
    summary = dict(status='passed_full_valid_WSI_statistics_fixture' if a.fixture else 'complete_full_valid_WSI_statistics_unaccepted',
        worker_pid=os.getpid(),source_sha256=sha(__file__),statistical_core_sha256=sha('benchmarks/v5/wsi_global/wsi_statistics.py'),
        inputs=records,arms=list(arms),bootstrap=result['bootstrap'],
        output_sha256={p.name:sha(p) for p in a.output.glob('*.json')},
        scope='complete_valid4x_including_support_separate_diagnostic',
        limits='WSI-conditional uncertainty, not independent-patient/seed/population evidence. Shared prompts/GT eligibility required; six interactive variants, view/prior/capacity differences disclosed. Closed reference unique and separate; full-valid closed component/distance reference is not inferred from remote rows. Label review provenance pending user confirmation.',
        remaining='Independent count-derived statistics acceptance; full30 quality and uncertainty cannot be established by a single-WSI fixture.')
    atomic_json(a.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('output_sha256','inputs')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--closed',type=Path)
    p.add_argument('--fixture',action='store_true')
    run(p.parse_args())
