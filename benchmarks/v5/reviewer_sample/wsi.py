"""Three full slides, identical windows, paired window/owner/fused readouts.

Persist lossless projected window binary masks and both whole-slide binary masks.
Probabilities are fused in float64 streaming stripes as in the accepted worker;
the scoped owner/mean comparison needs no alternative probability-fusion search.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import pyvips
import torch
from benchmarks.v5.whole_slide_once import worker as raster
from benchmarks.v5.whole_slide_once.data import inputs, prompts, boxes_at, grid, gt_extent
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def load(method, checkpoint):
    if method == 'proto_ms':
        from .model import build
        return build(method, checkpoint).eval()
    if method in ['M', 'no_cross_scale']:
        from benchmarks.v5.hg_paper_ablation.model import build
        return build(method, checkpoint).eval()
    if method == 'HG_mix':
        from benchmarks.v5.hg_combination_study.model import build
        return build(method, checkpoint).eval()
    assert method == 'sansa'
    from benchmarks.v5.same_slide_retrain.model import build
    from benchmarks.v5.sansa_cache.model import CachedSANSA
    return CachedSANSA(build('sansa', checkpoint)).eval()


def encode(model, value, method):
    return model.encode(value if method == 'sansa' else model.base.transform(value))


def decode(model, se, qe, point, method, local=False):
    if method == 'sansa': return model.decode(se, qe, point, local)['pixel_probability']
    return model.cached_decode(se, qe, point, local, with_support=False)[0]['pixel_probability']


def remote_crop(extent, excluded):
    x, y, right, bottom = extent
    mask = np.ones((bottom-y, right-x), bool)
    for x0, y0, x1, y1 in excluded:
        left, top, rr, bb = max(x, x0), max(y, y0), min(right, x1), min(bottom, y1)
        if left < rr and top < bb: mask[top-y:bb-y, left-x:rr-x] = False
    return mask


def distance_to_center(box, extent, case):
    x0, y0, x1, y1 = extent
    ax, ay = case['gt_affine'][:2]
    dx = (np.arange(x0, x1)+.5)/ax-(box[0]+box[2]/2)
    dy = (np.arange(y0, y1)+.5)/ay-(box[1]+box[3]/2)
    return dy[:, None]**2+dx[None, :]**2


def accumulation_gate():
    # Identical windows: owner chooses nearest center; average can differ.
    case = dict(gt_affine=[1., 1., 0., 0.], gt_width=6, gt_height=2)
    total = np.zeros((2, 6), np.float64); count = np.zeros((2, 6), np.uint16)
    distance = np.full((2, 6), np.inf); owner = np.zeros((2, 6), bool)
    for box, val in [([0, 0, 4, 2], .25), ([2, 0, 4, 2], .75)]:
        extent = gt_extent(box, case); x0, y0, x1, y1 = extent
        proj = raster.projected(np.full((512, 512), val, np.float32), box, extent, case)
        delta = distance_to_center(box, extent, case); take = delta < distance[y0:y1, x0:x1]
        distance[y0:y1, x0:x1][take] = delta[take]; owner[y0:y1, x0:x1][take] = (proj >= .5)[take]
        total[y0:y1, x0:x1] += proj; count[y0:y1, x0:x1] += 1
    assert (count > 0).all() and np.isfinite(distance).all()
    assert np.array_equal(owner[0], [False, False, False, True, True, True])
    assert np.array_equal((total/count >= .5)[0], [False, False, True, True, True, True])
    assert np.array_equal(remote_crop([1, 0, 5, 2], [[2, 0, 4, 1]]),
                          raster.remote_valid(0, 2, 6, [[2, 0, 4, 1]])[:, 1:5])


@torch.no_grad()
def infer(model, method, case, supports, out, reference=None):
    out.mkdir(exist_ok=False); start = time.monotonic(); colors = raster.palette()
    gt = pyvips.Image.new_from_file(case['gt_path'], access='random')
    groups, bw, bh = grid(case, supports); total_windows = sum(len(xs) for _, xs in groups)
    se, points, excluded, sums, owners, masks, counts, binaries, previous = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for support in supports:
        target = support['target']; se[target] = encode(model, inputs(case['wsi_path'], support['geometry']['native_boxes']), method)
        points[target] = prompts(support)
        sup = support['geometry']['native_boxes'][2]
        excluded[target] = [gt_extent([x, y, bw, bh], case) for y, xs in groups for x in xs
                            if raster.intersects(boxes_at(x, y, bw, bh)[2], sup)]
        sums[target] = np.zeros((0, gt.width), np.float64); owners[target] = np.zeros((0, gt.width), bool)
        dest = out/f'target_{target}'; dest.mkdir()
        masks[target] = {kind: np.lib.format.open_memmap(dest/(kind+'.npy'), mode='w+', dtype=np.uint8,
             shape=(gt.height, math.ceil(gt.width/8))) for kind in ['fused', 'owner']}
        binaries[target] = (dest/'window_masks.bin').open('xb')
        counts[target] = {kind: {domain: defaultdict(int) for domain in ['all', 'remote']} for kind in ['fused', 'owner']}
        if reference:
            previous[target] = np.load(reference/f'target_{target}/mask.npy', mmap_mode='r')
    weight = np.zeros((0, gt.width), np.uint16); best_distance = np.full((0, gt.width), np.inf)
    cursor = done = 0
    # JSONL rows bind binary offsets, projected geometry and metrics for replay.
    rows_file = (out/'windows.jsonl').open('x')
    try:
        for gi, (y, xs) in enumerate(groups):
            bottom = gt_extent([0, y, bw, bh], case)[3]; need = bottom-cursor
            if need > len(weight):
                extra = need-len(weight)
                weight = np.concatenate([weight, np.zeros((extra, gt.width), np.uint16)])
                best_distance = np.concatenate([best_distance, np.full((extra, gt.width), np.inf)])
                for target in sums:
                    sums[target] = np.concatenate([sums[target], np.zeros((extra, gt.width), np.float64)])
                    owners[target] = np.concatenate([owners[target], np.zeros((extra, gt.width), bool)])
            for x in xs:
                box = [x, y, bw, bh]; extent = gt_extent(box, case); x0, y0, x1, y1 = extent
                q = encode(model, inputs(case['wsi_path'], boxes_at(*box)), method)
                dist = distance_to_center(box, extent, case)
                chosen = dist < best_distance[y0-cursor:y1-cursor, x0:x1]
                best_distance[y0-cursor:y1-cursor, x0:x1][chosen] = dist[chosen]
                frozen = {}
                for support in supports:
                    target = support['target']; local = box == support['geometry']['native_boxes'][0]
                    prob = decode(model, se[target], se[target] if local else q, points[target], method, local)
                    assert torch.isfinite(prob).all() and prob.shape == (1, 512, 512)
                    value = raster.projected(prob[0].cpu().numpy(), box, extent, case)
                    sums[target][y0-cursor:y1-cursor, x0:x1] += value
                    pred = value >= .5; owners[target][y0-cursor:y1-cursor, x0:x1][chosen] = pred[chosen]
                    packed = np.packbits(pred, axis=None); offset = binaries[target].tell()
                    binaries[target].write(packed.tobytes())
                    assert np.array_equal(np.unpackbits(packed, count=pred.size).reshape(pred.shape), pred)
                    frozen[target] = (pred, offset, len(packed))
                weight[y0-cursor:y1-cursor, x0:x1] += 1
                # All predictions are fixed before the first query-label read.
                labels = raster.label_strip(gt.crop(x0, y0, x1-x0, y1-y0), 0, y1-y0, colors); valid = labels != 255
                for target, (pred, offset, length) in frozen.items():
                    stats = {domain: raster.metrics(raster.tally(pred, labels == target, domain_valid)) for domain, domain_valid in
                             [('all', valid), ('remote', valid & remote_crop(extent, excluded[target]))]}
                    rows_file.write(json.dumps(dict(window=done, target=target, extent=extent, box=box, offset=offset, bytes=length, metrics=stats))+'\n')
                done += 1
                if done == 1 or done % 25 == 0:
                    rows_file.flush()
                    dump(out/'progress.json', dict(status='inference_running', pid=os.getpid(), windows=done, total=total_windows,
                         method=method, targets=len(supports), seconds=time.monotonic()-start))
            safe = gt_extent([0, groups[gi+1][0], bw, bh], case)[1] if gi+1 < len(groups) else gt.height
            n = safe-cursor
            if n:
                assert (weight[:n] > 0).all() and np.isfinite(best_distance[:n]).all()
                fixed = {t: {'fused': sums[t][:n]/weight[:n] >= .5, 'owner': owners[t][:n]} for t in sums}
                labels = raster.label_strip(gt, cursor, n, colors); valid = labels != 255
                for target, outputs in fixed.items():
                    for kind, pred in outputs.items():
                        packed = np.packbits(pred, axis=1); masks[target][kind][cursor:safe] = packed
                        if kind == 'fused' and reference: assert np.array_equal(packed, previous[target][cursor:safe]), 'original fused mask mismatch'
                        raster.add(counts[target][kind]['all'], raster.tally(pred, labels == target, valid))
                        raster.add(counts[target][kind]['remote'], raster.tally(pred, labels == target, valid & raster.remote_valid(cursor, n, gt.width, excluded[target])))
                    sums[target] = sums[target][n:].copy(); owners[target] = owners[target][n:].copy()
                weight = weight[n:].copy(); best_distance = best_distance[n:].copy(); cursor = safe
    finally:
        rows_file.close()
        for f in binaries.values(): f.close()
        for by in masks.values():
            for arr in by.values(): arr.flush()
    assert cursor == gt.height and done == total_windows
    dump(out/'progress.json', dict(status='independent_disk_replay', pid=os.getpid(), windows=done, total=total_windows, seconds=time.monotonic()-start))
    # Independent scoring from disk; no model probabilities or online counters reused.
    replay = {t: {k: {d: defaultdict(int) for d in ['all', 'remote']} for k in ['fused', 'owner']} for t in counts}
    for y in range(0, gt.height, 512):
        n = min(512, gt.height-y); labels = raster.label_strip(gt, y, n, colors); valid = labels != 255
        for target in counts:
            for kind in ['fused', 'owner']:
                arr = np.load(out/f'target_{target}'/(kind+'.npy'), mmap_mode='r')
                pred = np.unpackbits(arr[y:y+n], axis=1, count=gt.width).astype(bool)
                raster.add(replay[target][kind]['all'], raster.tally(pred, labels == target, valid))
                raster.add(replay[target][kind]['remote'], raster.tally(pred, labels == target, valid & raster.remote_valid(y, n, gt.width, excluded[target])))
    assert replay == counts
    readers = {t: (out/f'target_{t}/window_masks.bin').open('rb') for t in counts}
    windows = {t: {d: [] for d in ['all', 'remote']} for t in counts}; replayed = 0; cached_window = None
    try:
        for line in (out/'windows.jsonl').open():
            row = json.loads(line); target = row['target']; x0, y0, x1, y1 = row['extent']
            if cached_window != row['window']:
                labels = raster.label_strip(gt.crop(x0, y0, x1-x0, y1-y0), 0, y1-y0, colors); cached_window = row['window']
            f = readers[target]; f.seek(row['offset']); packed = np.frombuffer(f.read(row['bytes']), np.uint8)
            pred = np.unpackbits(packed, count=(x1-x0)*(y1-y0)).reshape(y1-y0, x1-x0).astype(bool)
            valid = labels != 255
            for domain, mask in [('all', valid), ('remote', valid & remote_crop(row['extent'], excluded[target]))]:
                actual = raster.metrics(raster.tally(pred, labels == target, mask)); assert actual == row['metrics'][domain]
                windows[target][domain].append(actual)
            replayed += 1
    finally:
        for f in readers.values(): f.close()
    assert replayed == done*len(supports)
    targets = {}
    for target in counts:
        summary = {}
        for domain, values in windows[target].items():
            present = [v for v in values if v['positive']]; absent = [v for v in values if not v['positive'] and v['valid']]
            summary[domain] = dict(present=len(present), absent=len(absent), zero_valid=sum(v['valid'] == 0 for v in values),
                present_Dice=float(np.mean([v['dice'] for v in present])) if present else None,
                absent_FPA=float(np.mean([v['false_positive_area_fraction'] for v in absent])) if absent else None)
        targets[target] = dict(window=summary, **{kind: {d: raster.metrics(counts[target][kind][d]) for d in ['all', 'remote']} for kind in ['fused', 'owner']},
             binary_sha256={name: sha(out/f'target_{target}'/name) for name in ['window_masks.bin', 'fused.npy', 'owner.npy']})
    dump(out/'report.json', dict(status='complete_accepted_sample_WSI', method=method, slide=case['wsi_id'], patient_id=case['patient_id'],
        targets=targets, windows=done, total_seconds=time.monotonic()-start, full_coverage=True, independent_window_and_whole_disk_replay=True,
        original_fused_mask_exact=bool(reference), support_reset_per_slide_target=True,
        probability_storage='Float64 stripe sums consumed once; window GT-grid binary masks and owner/fused masks stored losslessly. No arbitrary post-hoc fusion search.',
        limitations='Fixed first3 previously evaluated slides; mechanistic diagnostic, not new blind test or evidence of population significance.'))
    dump(out/'acceptance.json', dict(status='complete_accepted_sample_WSI', report_sha256=sha(out/'report.json'), windows_sha256=sha(out/'windows.jsonl'),
         windows=done, targets=len(supports), original_fused_mask_exact=bool(reference)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--method', required=True)
    p.add_argument('--case-index', type=int, required=True); p.add_argument('--gate-only', action='store_true'); a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text()); ext = json.loads((a.root/'wsi_contract.json').read_text())
    for path, digest in ext['sources'].items(): assert sha(path) == digest, path
    assert sha(a.root/'cases.json') == c['cases_sha256']; case = json.loads((a.root/'cases.json').read_text())[a.case_index]
    assert case['eligible'] and not any(case['membership'].values())
    checkpoint = c['reference_checkpoints'].get(a.method)
    if a.method == 'proto_ms':
        formal = a.root/'proto_ms/formal'; acc = json.loads((formal/'acceptance.json').read_text())
        assert acc['status'] == 'complete_accepted_training'; checkpoint = dict(path=str(formal/'checkpoint.pt'), sha256=acc['checkpoint_sha256'])
    assert checkpoint and sha(checkpoint['path']) == checkpoint['sha256']
    if a.method == 'sansa':
        receipt = Path(ext['sansa_cache_gate']); assert sha(receipt) == ext['sansa_cache_gate_sha256']
        assert json.loads(receipt.read_text())['status'] == 'accepted_exact_cache_equivalence'
    out = a.root/'wsi'/a.method/f'case_{a.case_index:02}'; out.mkdir(parents=True, exist_ok=False)
    dump(out/'worker.json', dict(pid=os.getpid(), method=a.method, gpu=os.environ['CUDA_VISIBLE_DEVICES'], checkpoint=checkpoint))
    raster.configure(); model = load(a.method, checkpoint['path']); state = tree_sha(model.state_dict()); accumulation_gate()
    raster.encode, raster.decode = encode, decode
    raster.engineering(model, a.method, case, case['supports'][0], out)
    if not a.gate_only:
        reference = Path(c['original_wsi_root'])/'formal'/f'case_{a.case_index:02}'/a.method if a.method in ['HG_mix', 'sansa'] else None
        infer(model, a.method, case, case['supports'], out/'readouts', reference)
        assert tree_sha(model.state_dict()) == state
    dump(out/'acceptance.json', dict(status='accepted_engineering_only' if a.gate_only else 'complete_accepted', checkpoint_sha256=checkpoint['sha256'],
         cases_sha256=c['cases_sha256'], unchanged_model_state=True))


if __name__ == '__main__': main()
