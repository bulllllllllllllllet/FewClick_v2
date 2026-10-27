"""Independent whole-domain retrieval audit, importing no producer/scorer.

Rebuild the confusion matrix and global component areas/hits once per native
prediction, then independently count physical-distance strata in 173-row bands.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import time
import traceback

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def metrics(tp, fp, positive, negative):
    valid = positive+negative
    return dict(tp=tp, fp=fp, fn=positive-tp, positive=positive, negative=negative,
                valid=valid, dice=2*tp/(positive+tp+fp) if positive else None,
                recall=tp/positive if positive else None,
                precision=tp/(tp+fp) if tp+fp else 0.,
                false_positive_fraction=fp/valid if valid else None,
                false_positive_rate=fp/negative if negative else None)


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter(); pid = os.getpid()
    def dump(n, d):
        (a.output/n).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')
    try:
        source = read(a.source/'summary.json')
        fixture = source['status'] == 'passed_closed12_streaming_retrieval_fixture'
        require(fixture or source['status'] == 'complete_closed12_full30_retrieval_unaccepted', 'complete retrieval execution required')
        if not fixture:
            require(a.verified_gate is not None, 'actual independent native retrieval audit fixture required')
            gate = read(a.verified_gate/'summary.json')
            require(gate['status'] == 'accepted_closed12_retrieval_fixture' and gate['source_sha256'] == sha(__file__), 'same independent auditor recipe')
        for p, h in source['signature']['sources'].items():
            require(sha(p) == h, 'scoring producer source remains frozen')
        require(sha(a.source/'rows.json') == source['rows_sha256'], 'all source scoring rows bound')
        for p, h in read(a.source/'contract.json')['inputs'].items():
            require(sha(p) == h, 'all original source acceptance receipts/protocol bound')
        rows = read(a.source/'rows.json')
        indexed = {(r['item'],r['target'],r['prompt_group'],r['budget']): r for r in rows}
        require(len(indexed) == len(rows) == (12 if fixture else 5520), 'unique complete task-budget identity set')
        items = read(a.raster/'items.json'); ci = {r['wsi_id']:r for r in read(a.components/'items.json')}
        data = {r['wsi_id']:r for r in read(a.data/'slides_private.json')}
        plans = read(a.protocol/'support_tasks_maximum.json')
        domains = {(r['wsi_id'],r['target'],r['prompt_group'],r['budget']):r for r in read(a.components/'task_domains.json')}
        require(len(items) == (2 if fixture else 30), 'all native prediction domains')
        visited = set(); evidence = []
        for number, item in enumerate(items):
            sid = item['wsi_id']; r = data[sid]; component = ci[sid]
            gtpath = a.data/sid/'labels.npy'
            require(sha(gtpath) == r['files'][str(gtpath)]['sha256'], 'actual original GT identity')
            for d in component['files'].values():
                require(sha(d['path']) == d['sha256'], 'all original full-image component/table/domain bytes')
            d = item['prediction_files']['class_prediction.npy']
            require(sha(d['path']) == d['sha256'], 'actual native prediction identity')
            gt = np.load(gtpath, mmap_mode='r'); cc = np.load(component['files']['components.npy']['path'], mmap_mode='r')
            pred = np.load(d['path'], mmap_mode='r')
            with np.load(component['files']['component_table.npz']['path'], allow_pickle=False) as z:
                classes = z['classes']; accepted_areas = z['remote_areas']
            gh, gw = gt.shape; w, h = r['native_wh']; sx, sy = w/gw, h/gh; mx, my = r['native_mpp']
            left, top, right, bottom = item['coverage']['bounds_native']
            xs = np.array([x for x in range(gw) if x*sx >= left and (x+1)*sx <= right], np.int64)
            ys = np.array([y for y in range(gh) if y*sy >= top and (y+1)*sy <= bottom], np.int64)
            require([int(xs[0]), int(ys[0]), int(xs[-1])+1, int(ys[-1])+1] == item['GT_domain']['GT_cell_bounds'], 'independent physical GT-cell footprint')
            x_native = np.array([(int(x)+.5)*sx for x in xs])
            px = np.array([math.floor(v)-left for v in x_native], np.int64)
            chosen = [p for p in plans if p['wsi_id'] == sid and (not fixture or (p['target'] in (0,7,10) and p['prompt_group'] == 0))]
            cases = {}; tasks = []
            for plan in chosen:
                for budget in (('1+1','3+2') if fixture else ('3+2','1+1','3+1','5+1')):
                    npos,nneg = map(int,budget.split('+')); key = (plan['target'],plan['prompt_group'],npos)
                    ids = sorted(set(int(cc[math.floor(y/sy),math.floor(x/sx)]) for x,y in plan['positive_native_xy'][:npos]))
                    require(all(i>0 and classes[i] == plan['target'] for i in ids), 'true positive-point full-image component classes')
                    if key not in cases:
                        cases[key] = dict(ids=ids, points=plan['positive_native_xy'][:npos], counts=[Counter() for _ in range(3)])
                    tasks.append((plan,budget,nneg,key))
            area = np.zeros(len(classes),np.int64); hits = np.zeros(len(classes),np.int64); matrix = np.zeros((12,12),np.int64)
            for offset in range(0,len(ys),173):
                yy = ys[offset:offset+173]
                labels = np.asarray(gt[np.ix_(yy,xs)]); ids = np.asarray(cc[np.ix_(yy,xs)])
                native_y = np.array([(int(y)+.5)*sy for y in yy])
                py = np.array([math.floor(y)-top for y in native_y],np.int64)
                guess = np.asarray(pred[np.ix_(py,px)]); valid = labels != 255
                require(np.array_equal(ids == 0, ~valid), 'void and original component raster binding')
                require(np.all(guess < 12), 'categorical class range')
                matrix += np.bincount(labels[valid].astype(np.int64)*12+guess[valid],minlength=144).reshape(12,12)
                area += np.bincount(ids.reshape(-1),minlength=len(classes))
                hits += np.bincount(ids[(labels == guess)&valid],minlength=len(classes))
                for (target,group,npos), case in cases.items():
                    eligible = valid.copy()
                    for gid in case['ids']:
                        eligible &= ids != gid
                    truth = labels == target; predicted = guess == target
                    nearest = np.full(labels.shape,np.inf)
                    for x,y in case['points']:
                        nearest = np.minimum(nearest,np.hypot((x_native[None,:]-x)*mx,(native_y[:,None]-y)*my))
                    bins = (nearest < 1000, (nearest >= 1000)&(nearest < 5000), nearest >= 5000)
                    for counter, mask in zip(case['counts'],bins):
                        mask = mask & eligible
                        counter.update(tp=int(np.count_nonzero(mask&truth&predicted)), fp=int(np.count_nonzero(mask&~truth&predicted)),
                                       positive=int(np.count_nonzero(mask&truth)), negative=int(np.count_nonzero(mask&~truth)))
                dump('status.json', dict(status='independent_full_retrieval_counts',worker_pid=pid,item=item['item'],items_complete=number,total_items=len(items),GT_rows_complete=offset+len(yy),GT_rows_total=len(ys)))
                for array in (gt,cc,pred):
                    array._mmap.madvise(mmap.MADV_DONTNEED)
            require(matrix.tolist() == item['confusion'], 'complete standard confusion matrix equals accepted native GT scoring')
            if not fixture:
                require(np.array_equal(area,accepted_areas), 'entire per-component remote areas equal independently accepted CCL table')
            for plan,budget,nneg,key in tasks:
                target = plan['target']; identity = (item['item'],target,plan['prompt_group'],budget)
                row = indexed[identity]; visited.add(identity); case = cases[key]; result = row['scores']
                require(row['wsi_id'] == sid and row['method'] == 'closed12_supervised_reference', 'method/slide identity')
                require(row['positive_native_xy'] == case['points'] and row['negative_native_xy'] == plan['negative_native_xy'][:nneg], 'actual source points and budget prefixes')
                tp = int(matrix[target,target]); positive = int(matrix[target].sum()); fp = int(matrix[:,target].sum())-tp; negative = int(matrix.sum())-positive
                removed = int(area[case['ids']].sum()); removed_hits = int(hits[case['ids']].sum())
                require(result['standard'] == metrics(tp,fp,positive,negative), 'all standard metrics')
                require(result['unprompted'] == metrics(tp-removed_hits,fp,positive-removed,negative), 'all post-censor metrics retain every non-target false positive')
                require(result['censored_target_pixels'] == removed, 'entire positive-component censor count')
                eligible_ids = np.flatnonzero((classes == target)&(area > 0)&~np.isin(np.arange(len(classes)),case['ids']))
                compstats = result['components']; path = Path(compstats['file']); require(sha(path) == compstats['sha256'], 'all saved predicted component counter bytes')
                with np.load(path,allow_pickle=False) as z:
                    require(z['ids'].dtype == np.uint32 and z['areas'].dtype == z['hits'].dtype == np.int64, 'exact saved component counter types')
                    require(np.array_equal(z['ids'],eligible_ids) and np.array_equal(z['areas'],area[eligible_ids]) and np.array_equal(z['hits'],hits[eligible_ids]), 'every component ID/area/prediction hit counter')
                ratios = hits[eligible_ids]/area[eligible_ids]
                require(compstats['count'] == len(eligible_ids) and compstats['mean_area_recall'] == (float(np.mean(ratios)) if len(ratios) else None), 'global component macro recall without band averaging')
                require(compstats['detected_at_50'] == int(np.count_nonzero(ratios >= .5)) and compstats['detected_at_10'] == int(np.count_nonzero(ratios >= .1)), 'every fixed area-recall component detection count')
                for expected_bin, counter, actual in zip(('<1mm','1-5mm','>=5mm'),case['counts'],result['distance']):
                    require(actual == dict(bin=expected_bin,**metrics(counter['tp'],counter['fp'],counter['positive'],counter['negative'])), 'all independent Euclidean physical-distance scores')
                if not fixture:
                    d = domains[(sid,target,plan['prompt_group'],budget)]
                    require((positive,positive-removed,negative,len(eligible_ids)) == (d['remote_target_pixels'],d['unprompted_target_pixels'],d['negative_GT_pixels'],d['unprompted_components']), 'accepted full30 task-domain denominators')
            evidence.append(dict(item=item['item'],GT_pixels=int(matrix.sum()),tasks=len(tasks),component_ids=int(np.count_nonzero(area[1:]))))
            print(json.dumps(evidence[-1]),flush=True)
            del gt,cc,pred
        require(visited == set(indexed), 'every row independently accounted for')
        result = dict(status='accepted_closed12_retrieval_fixture' if fixture else 'accepted_closed12_full30_component_distance_scores', worker_pid=pid, items=len(items), tasks=len(visited),
                      source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),numpy=np.__version__,evidence=evidence,seconds=time.perf_counter()-started,
                      scope='All GT cells, whole-image component IDs/areas/prediction hits, source prefixes, standard and uncensored non-target FP, Euclidean physical pixel-distance scores; no producer scorer imported',
                      remaining='BF1, common276/full330 report and paired uncertainty; closed12 privileged reference only, interactive main30 methods pending')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for n in ('source','data','protocol','components','raster','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verified-gate',type=Path)
    run(p.parse_args())
