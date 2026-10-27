"""Stream accepted native closed12 predictions into GT retrieval evaluation.

This is a privileged twelve-class reference. Its standard predictions ignore
prompts; repeated source groups/budgets are not independent model predictions.
"""
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from .retrieval_stream import RetrievalAccumulator


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(test, reason):
    if not test:
        raise ValueError(reason)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pid = os.getpid()
    def dump(name, data):
        (a.output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    try:
        comp = read(a.components/'summary.json')
        ca = read(a.component_acceptance/'summary.json')
        ra = read(a.raster_acceptance/'summary.json')
        raster = read(a.raster/'summary.json')
        require(ca['source_summary_sha256'] == sha(a.components/'summary.json'), 'component acceptance binds actual source')
        require(ra['source_summary_sha256'] == sha(a.raster/'summary.json'), 'native probability/GT acceptance binds source')
        require(ca['status'] == ('accepted_GT_component_gate' if a.gate_fixture else 'accepted_all30_GT_components_and_prompt_domains'), 'required component acceptance scope')
        require(ra['status'] == ('accepted_closed12_raster_fixture' if a.gate_fixture else 'accepted_closed12_full30_native_raster_and_GT_scores'), 'required full-native prediction acceptance scope')
        contract = read(a.components/'contract.json')
        require(contract['protocol_summary_sha256'] == sha(a.component_protocol/'summary.json'), 'component source-point protocol identity')
        require(raster['signature']['protocol_sha256'] == sha(a.protocol/'summary.json'), 'native prediction protocol identity')
        require(sha(a.raster/'items.json') == raster['items_sha256'], 'accepted prediction item bytes')
        require(sha(a.components/'items.json') == comp['items_sha256'], 'accepted full-image component item bytes')
        sources = {str(p): sha(p) for p in (Path(__file__), Path('benchmarks/v5/wsi_global/retrieval_metrics.py'), Path('benchmarks/v5/wsi_global/retrieval_stream.py'))}
        signature = dict(sources=sources, block_rows=64, native_prediction_sampling='GT cell centre floor', component_area='within entire scoring domain; full-original-GT IDs', component_filters=False)
        if not a.gate_fixture:
            require(a.verified_gate is not None, 'actual streaming integration gate required')
            gate = read(a.verified_gate/'summary.json')
            require(gate['status'] == 'passed_closed12_streaming_retrieval_fixture' and gate['signature'] == signature, 'same-source streaming gate')
            require(a.component_protocol.resolve() == a.protocol.resolve(), 'formal full30 component/prediction protocol is identical')
        data = {r['wsi_id']: r for r in read(a.data/'slides_private.json')}
        component_items = {r['wsi_id']: r for r in read(a.components/'items.json')}
        plans = read(a.protocol/'support_tasks_maximum.json')
        component_plans = read(a.component_protocol/'support_tasks_maximum.json')
        items = read(a.raster/'items.json')
        require(len(items) == (2 if a.gate_fixture else 30), 'complete requested native prediction items')
        if a.gate_fixture:
            require(a.reference_gate is not None, 'independently counted actual retrieval gate required')
            reference = read(a.reference_gate/'rows.json')
            require(read(a.reference_gate/'summary.json')['status'] == 'passed_actual_GT_retrieval_metric_gate', 'accepted rowwise Counter/hypot native fixture')
        dump('contract.json', dict(worker_pid=pid, signature=signature, gate_fixture=a.gate_fixture,
             inputs={str(p): sha(p) for p in [a.components/'summary.json', a.component_acceptance/'summary.json', a.raster/'summary.json', a.raster_acceptance/'summary.json', a.protocol/'summary.json', a.component_protocol/'summary.json']},
             supervision='closed12 full multiclass CE reference, no interactive model input',
             repeated_rows='source groups and budgets change scoring censor/distance only; standard predictions must not be treated as independent samples'))
        rows = []
        for item_index, item in enumerate(items):
            sid = item['wsi_id']; r = data[sid]; ci = component_items[sid]
            gtpath = a.data/sid/'labels.npy'
            require(sha(gtpath) == r['files'][str(gtpath)]['sha256'], 'accepted original GT bytes')
            require(sha(ci['files']['components.npy']['path']) == ci['files']['components.npy']['sha256'], 'accepted wholeGT component raster bytes')
            pd = item['prediction_files']['class_prediction.npy']
            require(sha(pd['path']) == pd['sha256'], 'accepted native categorical prediction bytes')
            gt = np.load(gtpath, mmap_mode='r'); cc = np.load(ci['files']['components.npy']['path'], mmap_mode='r'); pred = np.load(pd['path'], mmap_mode='r')
            sx, sy = r['native_wh'][0]/gt.shape[1], r['native_wh'][1]/gt.shape[0]
            x0, y0, x1, y1 = item['GT_domain']['GT_cell_bounds']
            left, top, right, bottom = item['coverage']['bounds_native']
            require([x0, y0, x1, y1] == [math.ceil(left/sx), math.ceil(top/sy), math.floor(right/sx), math.floor(bottom/sy)], 'whole-cell scoring rectangle')
            if not a.gate_fixture:
                require(ci['remote_GT_bounds'] == [x0, y0, x1, y1], 'full component remote domain matches prediction domain')
            chosen = [p for p in plans if p['wsi_id'] == sid and (not a.gate_fixture or (p['target'] in (0, 7, 10) and p['prompt_group'] == 0))]
            cases = []; accumulators = {}
            for plan in chosen:
                original = next(p for p in component_plans if (p['wsi_id'],p['target'],p['prompt_group']) == (sid,plan['target'],plan['prompt_group']))
                require(original['positive_native_xy'] == plan['positive_native_xy'] and original['negative_native_xy'] == plan['negative_native_xy'], 'actual source prefixes match component protocol')
                for budget in (('1+1', '3+2') if a.gate_fixture else ('3+2', '1+1', '3+1', '5+1')):
                    npos, nneg = map(int, budget.split('+')); key = (plan['target'], plan['prompt_group'], npos)
                    points = plan['positive_native_xy'][:npos]
                    ids = [int(cc[math.floor(y/sy), math.floor(x/sx)]) for x,y in points]
                    if key not in accumulators:
                        accumulators[key] = (RetrievalAccumulator(), points, ids)
                    cases.append((plan, budget, nneg, key))
            xx = (np.arange(x0, x1)+.5)*sx
            for y in range(y0, y1, signature['block_rows']):
                end = min(y+signature['block_rows'], y1); yy = (np.arange(y, end)+.5)*sy
                labels = np.asarray(gt[y:end, x0:x1]); components = np.asarray(cc[y:end, x0:x1])
                sample = np.asarray(pred[np.ix_(np.floor(yy).astype(np.int64)-top, np.floor(xx).astype(np.int64)-left)])
                for (target, group, npos), (acc, points, ids) in accumulators.items():
                    acc.update(labels, components, sample == target, target, ids, xx, yy, r['native_mpp'], points)
                dump('status.json', dict(status='streaming_retrieval_GT_bands', worker_pid=pid, item=item['item'], items_complete=item_index, total_items=len(items), GT_rows_complete=end-y0, GT_rows_total=y1-y0, seconds=time.perf_counter()-started))
                for array in (gt, cc, pred):
                    array._mmap.madvise(mmap.MADV_DONTNEED)
            finished = {key: a[0].finish() for key,a in accumulators.items()}
            folder = a.output/item['item']; folder.mkdir()
            for plan, budget, nneg, key in cases:
                # Deep copy to preserve merged counters for shared positive prefixes.
                result = json.loads(json.dumps(finished[key]))
                if a.gate_fixture:
                    expected = next(z['result'] for z in reference if z['item'] == item['item'] and z['target'] == plan['target'] and z['budget'] == budget)
                    require(result == expected, 'all streaming counters/ratios/components equal independently counted whole-window result')
                compstats = result['components']; filename = folder/f'target{key[0]}_group{key[1]}_positive{key[2]}.npz'
                if not filename.exists():
                    np.savez_compressed(filename, ids=np.array(compstats['ids'],np.uint32), areas=np.array(compstats['areas'],np.int64), hits=np.array(compstats['hits'],np.int64))
                for k in ('ids', 'areas', 'hits'):
                    del compstats[k]
                compstats['file'] = str(filename); compstats['sha256'] = sha(filename)
                rows.append(dict(item=item['item'], wsi_id=sid, target=plan['target'], prompt_group=plan['prompt_group'], budget=budget, positive_native_xy=plan['positive_native_xy'][:key[2]], negative_native_xy=plan['negative_native_xy'][:nneg], method='closed12_supervised_reference', scores=result))
            dump('rows.json', rows)
            print(json.dumps(dict(item=item['item'], completed=item_index+1, total=len(items), rows=len(rows))), flush=True)
            del gt, cc, pred
        require(len(rows) == (12 if a.gate_fixture else 5520), 'all fixed task-budget rows present')
        summary = dict(status='passed_closed12_streaming_retrieval_fixture' if a.gate_fixture else 'complete_closed12_full30_retrieval_unaccepted', worker_pid=pid, signature=signature, items=len(items), rows=len(rows), rows_sha256=sha(a.output/'rows.json'), seconds=time.perf_counter()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       remaining='Independent full scoring acceptance, BF1, common276 versus closed full330, paired uncertainty and other interactive methods; no full goal completion')
        dump('summary.json', summary); dump('status.json', dict(status=summary['status'], worker_pid=pid))
        print(json.dumps(summary), flush=True)
    except Exception:
        dump('status.json', dict(status='failed', worker_pid=pid, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('data', 'protocol', 'component-protocol', 'components', 'component-acceptance', 'raster', 'raster-acceptance', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--gate-fixture', action='store_true')
    p.add_argument('--reference-gate', type=Path)
    p.add_argument('--verified-gate', type=Path)
    run(p.parse_args())
