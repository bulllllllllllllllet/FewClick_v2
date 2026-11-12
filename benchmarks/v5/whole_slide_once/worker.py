"""Same-slide full-grid inference and independently replayed unique-pixel Dice."""
import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyvips
import torch
import yaml
from PIL import Image
from scipy.ndimage import map_coordinates
from .data import inputs, prompts, boxes_at, grid, gt_extent, rgb_box
from benchmarks.v5.wsi_quality_study.prepare import dump, labels_for
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.sam_current_fair.resources import acquire


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load(method, checkpoint):
    if method == 'HG_mix':
        from benchmarks.v5.hg_combination_study.model import build
        return build('HG_mix', checkpoint).eval()
    from benchmarks.v5.same_slide_retrain.model import build
    return build('sansa', checkpoint).eval()


def encode(model, x, method):
    return model.encode(model.base.transform(x)) if method == 'HG_mix' else model.encode(x)


def decode(model, se, qe, point, method, local=False):
    if method == 'HG_mix':
        return model.cached_decode(se, qe, point, local, with_support=False)[0]['pixel_probability']
    return model.decode(se, qe, point, local)['pixel_probability']


def intersects(a, b):
    return min(a[0]+a[2], b[0]+b[2]) > max(a[0], b[0]) and min(a[1]+a[3], b[1]+b[3]) > max(a[1], b[1])


def projected(probability, box, extent, case):
    x0, y0, x1, y1 = extent
    ax, ay = case['gt_affine'][:2]
    xx = (((np.arange(x0, x1)+.5)/ax-box[0])*512/box[2]-.5)
    yy = (((np.arange(y0, y1)+.5)/ay-box[1])*512/box[3]-.5)
    return map_coordinates(probability, np.asarray(np.broadcast_arrays(yy[:, None], xx[None, :])),
                           order=1, mode='nearest', prefilter=False).astype(np.float32)


def palette():
    return yaml.safe_load(Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml').read_text())['data']['class_map']


def label_strip(gt, y, height, colors):
    rgb_image = gt.crop(0, y, gt.width, height).extract_band(0, n=3)
    rgb = np.frombuffer(rgb_image.write_to_memory(), np.uint8).reshape(height, gt.width, 3)
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for color in colors:
        labels[(rgb == color['rgb']).all(-1)] = color['id']
    return labels


def remote_valid(y, height, width, excluded):
    valid = np.ones((height, width), bool)
    for x0, y0, x1, y1 in excluded:
        lo, hi = max(y, y0), min(y+height, y1)
        if lo < hi:
            valid[lo-y:hi-y, x0:x1] = False
    return valid


def tally(prediction, truth, valid):
    return dict(tp=int(np.count_nonzero(prediction & truth & valid)),
                fp=int(np.count_nonzero(prediction & ~truth & valid)),
                fn=int(np.count_nonzero(~prediction & truth & valid)), valid=int(valid.sum()))


def add(total, values):
    for key, value in values.items():
        total[key] += value


def metrics(total):
    tp, fp, fn = (int(total[k]) for k in ['tp', 'fp', 'fn'])
    return dict(**{k: int(v) for k, v in total.items()}, positive=tp+fn,
                dice=2*tp/(2*tp+fp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else None,
                false_positive_area_fraction=fp/total['valid'] if total['valid'] else None)


@torch.no_grad()
def engineering(model, method, case, support, out):
    state = tree_sha(model.state_dict())
    g = support['geometry']
    x = inputs(case['wsi_path'], g['native_boxes'])
    point = prompts(support)
    se = encode(model, x, method)
    probability = decode(model, se, se, point, method, True)
    full = model(support=x, query=x, prompts=point, local=True, shared=True, with_support=False)['query']['pixel_probability']
    assert torch.equal(probability, full)
    # RGB and support labels reproduce the accepted historical input coordinates.
    from benchmarks.v5.paper_multiscale.data import native_context
    for scale in range(3):
        assert np.array_equal(rgb_box(case['wsi_path'], g['native_boxes'][scale]),
                              native_context({'multiscale_json': json.dumps(g)}, scale))
    old = np.asarray(Image.open(support['support_labels']))
    assert sha(support['support_labels']) == support['support_labels_sha256']
    expected = labels_for(dict(annotation=case['gt_path'], gt_affine=case['gt_affine']), g['native_boxes'][0], 512)
    assert np.array_equal(expected, old), 'historical support coordinate replay differs'
    bw, bh = g['native_boxes'][0][2:]
    boxes = [[0, 0, bw, bh], [case['width']-bw, case['height']-bh, bw, bh],
             [bw, bh, bw, bh]]
    for box in boxes:
        views = boxes_at(*box)
        q = inputs(case['wsi_path'], views)
        qe = encode(model, q, method)
        pred = decode(model, se, qe, point, method)
        full = model(support=x, query=q, prompts=point, local=False, shared=False,
                     with_support=False)['query']['pixel_probability']
        assert torch.equal(pred, full) and torch.isfinite(pred).all()
        assert pred.shape == (1, 512, 512)
    # Test the actual GT-grid resampler and overlapped probability fusion.
    accumulator = np.zeros((48, 64), np.float64)
    count = np.zeros_like(accumulator)
    toy = dict(gt_affine=[1., 1., 0., 0.], gt_width=64, gt_height=48)
    pieces = []
    for i, box in enumerate([[0, 0, 40, 48], [24, 0, 40, 48]]):
        value = np.full((512, 512), [.25, .75][i], np.float32)
        extent = gt_extent(box, toy)
        pred = projected(value, box, extent, toy)
        x0, y0, x1, y1 = extent
        accumulator[y0:y1, x0:x1] += pred
        count[y0:y1, x0:x1] += 1
        pieces.append(pred)
    assert (count > 0).all()
    fused = accumulator/count
    assert np.all(fused[:, :24] == .25) and np.all(fused[:, 24:40] == .5) and np.all(fused[:, 40:] == .75)
    assert np.array_equal(np.unpackbits(np.packbits(fused >= .5, axis=1), axis=1, count=64), fused >= .5)
    assert tree_sha(model.state_dict()) == state
    dump(out/'engineering.json', dict(status='accepted_real_WSI_engineering', method=method,
         local_cached_full_exact=True, edge_corner_remote_cached_full_exact=True,
         original_support_RGB_and_GT_exact=True, real_overlap_resampling_and_pack_exact=True,
         frozen_weights_unchanged=True, support_clicks=5, no_query_human_clicks=True))


@torch.no_grad()
def infer(model, method, case, supports, out):
    out.mkdir(exist_ok=False)
    colors = palette()
    gt = pyvips.Image.new_from_file(case['gt_path'], access='random')
    groups, bw, bh = grid(case, supports)
    total_windows = sum(len(xs) for _, xs in groups)
    encoded_support = {}
    points = {}
    excluded = {}
    accumulators = {}
    counts = {}
    mask_files = {}
    statistics = {}
    for support in supports:
        target = support['target']
        sx = inputs(case['wsi_path'], support['geometry']['native_boxes'])
        encoded_support[target] = encode(model, sx, method)
        points[target] = prompts(support)
        sup = support['geometry']['native_boxes'][2]
        excluded[target] = [gt_extent([x, y, bw, bh], case) for y, xs in groups for x in xs
                            if intersects(boxes_at(x, y, bw, bh)[2], sup)]
        accumulators[target] = np.zeros((0, case['gt_width']), np.float64)
        counts[target] = dict(all=defaultdict(int), remote=defaultdict(int))
        directory = out/f'target_{target}'
        directory.mkdir()
        mask_files[target] = np.lib.format.open_memmap(directory/'mask.npy', mode='w+', dtype=np.uint8,
               shape=(case['gt_height'], math.ceil(case['gt_width']/8)))
        statistics[target] = dict(target=target, support=support, exclusion_rectangles=excluded[target])
    weight = np.zeros((0, case['gt_width']), np.uint16)
    cursor = 0
    done = 0
    start = time.monotonic()
    timing = dict(query_encoding_seconds=0., decode_seconds=0., projection_and_score_seconds=0.)
    for gi, (y, xs) in enumerate(groups):
        top, bottom = gt_extent([0, y, bw, bh], case)[1::2]
        need = bottom-cursor
        if need > len(weight):
            extra = need-len(weight)
            weight = np.concatenate([weight, np.zeros((extra, case['gt_width']), np.uint16)])
            for target in accumulators:
                accumulators[target] = np.concatenate([accumulators[target], np.zeros((extra, case['gt_width']), np.float64)])
        for x in xs:
            box = [x, y, bw, bh]
            views = boxes_at(*box)
            extent = gt_extent(box, case)
            x0, y0, x1, y1 = extent
            t = time.monotonic()
            qx = inputs(case['wsi_path'], views)
            qe = encode(model, qx, method)
            torch.cuda.synchronize()
            timing['query_encoding_seconds'] += time.monotonic()-t
            for support in supports:
                target = support['target']
                local = box == support['geometry']['native_boxes'][0]
                t = time.monotonic()
                pred = decode(model, encoded_support[target], encoded_support[target] if local else qe,
                              points[target], method, local)
                assert torch.isfinite(pred).all()
                value = pred[0].cpu().numpy()
                timing['decode_seconds'] += time.monotonic()-t
                t = time.monotonic()
                accumulators[target][y0-cursor:y1-cursor, x0:x1] += projected(value, box, extent, case)
                timing['projection_and_score_seconds'] += time.monotonic()-t
            weight[y0-cursor:y1-cursor, x0:x1] += 1
            done += 1
            if done == 1 or done % 25 == 0:
                dump(out/'progress.json', dict(status='inference_running', pid=os.getpid(), method=method,
                     windows=done, total=total_windows, targets=len(supports), seconds=time.monotonic()-start,
                     timings=timing))
        safe = gt_extent([0, groups[gi+1][0], bw, bh], case)[1] if gi+1 < len(groups) else case['gt_height']
        n = safe-cursor
        if n:
            assert (weight[:n] > 0).all(), 'uncovered whole-slide output pixels'
            # These rows cannot receive another tile. Predictions are fixed before GT scoring.
            fixed = {target: accumulators[target][:n]/weight[:n] >= .5 for target in accumulators}
            t = time.monotonic()
            labels = label_strip(gt, cursor, n, colors)
            valid = labels != 255
            for target, pred in fixed.items():
                mask_files[target][cursor:safe] = np.packbits(pred, axis=1)
                truth = labels == target
                add(counts[target]['all'], tally(pred, truth, valid))
                add(counts[target]['remote'], tally(pred, truth, valid & remote_valid(cursor, n, gt.width, excluded[target])))
                accumulators[target] = accumulators[target][n:].copy()
            timing['projection_and_score_seconds'] += time.monotonic()-t
            weight = weight[n:].copy()
            cursor = safe
    assert cursor == case['gt_height'] and done == total_windows
    for array in mask_files.values():
        array.flush()
    dump(out/'progress.json', dict(status='independent_disk_mask_scoring', pid=os.getpid(),
                                   windows=done, total=total_windows, targets=len(supports)))
    masks = {target: np.load(out/f'target_{target}/mask.npy', mmap_mode='r') for target in counts}
    replay = {target: dict(all=defaultdict(int), remote=defaultdict(int)) for target in counts}
    for y in range(0, gt.height, 512):
        n = min(512, gt.height-y)
        labels = label_strip(gt, y, n, colors)
        valid = labels != 255
        for target, array in masks.items():
            pred = np.unpackbits(array[y:y+n], axis=1, count=gt.width).astype(bool)
            add(replay[target]['all'], tally(pred, labels == target, valid))
            add(replay[target]['remote'], tally(pred, labels == target,
                valid & remote_valid(y, n, gt.width, excluded[target])))
    for target in counts:
        assert dict(replay[target]['all']) == dict(counts[target]['all'])
        assert dict(replay[target]['remote']) == dict(counts[target]['remote'])
        statistics[target].update(whole_WSI=metrics(counts[target]['all']),
                                  remote_only_WSI=metrics(counts[target]['remote']),
                                  mask_sha256=sha(out/f'target_{target}/mask.npy'))
    result = dict(status='complete_accepted_whole_slide_Dice', method=method, slide=case['wsi_id'],
           patient_id=case['patient_id'], targets=statistics, windows=done, output_shape=[gt.height, gt.width],
           unique_pixels=gt.height*gt.width, total_seconds=time.monotonic()-start, timings=timing,
           full_output_covered=True, independent_disk_score_exact=True,
           support_reset_per_slide=True, no_query_human_clicks=True,
           encoded_query_reused_across_targets=method == 'HG_mix',
           limitations='SANSA encode caches normalized input only; native network image encoding remains in each decode. '
           'HG-mix caches neural query features within a tile across targets; no persistent full-WSI neural cache benchmark claimed. '
           'Audited target train/val disjoint, historical AAAI test exposure. Whole-slide physical BF1 not yet scored.')
    dump(out/'report.json', result)
    dump(out/'acceptance.json', dict(status='complete_accepted_whole_slide_Dice', report_sha256=sha(out/'report.json'),
         targets=len(supports), windows=done, gt_sha256=sha(case['gt_path']), full_output_covered=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--phase', choices=['pilot', 'formal', 'extra'], default='pilot')
    a = parser.parse_args()
    c = json.loads((a.root/'contract.json').read_text())
    assert sha(a.root/'cases.json') == c['cases_sha256']
    for f, digest in c['sources'].items():
        assert sha(f) == digest, f
    case = json.loads((a.root/'cases.json').read_text())[a.case_index]
    assert case['eligible'] and not any(case['membership'].values())
    supports = ([next(s for s in case['supports'] if s['target'] == case['pilot_target'])]
                if a.phase == 'pilot' else case['supports'])
    out = a.root/a.phase/f'case_{a.case_index:02}'
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'worker.json', dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], case=case['wsi_id'], phase=a.phase))
    identity = {key: dict(path=case[key], sha256=sha(case[key]), size=Path(case[key]).stat().st_size,
                         mtime_ns=Path(case[key]).stat().st_mtime_ns) for key in ['wsi_path', 'gt_path']}
    dump(out/'input_identity.json', identity)
    configure()
    try:
        for method, checkpoint in c['checkpoints'].items():
            assert sha(checkpoint['path']) == checkpoint['sha256']
            lease = acquire(c, int(os.environ['CUDA_VISIBLE_DEVICES']), method+'_'+a.phase)
            try:
                model = load(method, checkpoint['path'])
                (out/(method+'_gate')).mkdir()
                engineering(model, method, case, supports[0], out/(method+'_gate'))
                infer(model, method, case, supports, out/method)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            finally:
                lease.close()
        dump(out/'acceptance.json', dict(status='complete_accepted_paired_WSI_Dice', targets=len(supports),
             methods=list(c['checkpoints']), case_sha256=sha(a.root/'cases.json'),
             input_identity_sha256=sha(out/'input_identity.json')))
        for item in identity.values():
            assert Path(item['path']).stat().st_size == item['size']
            assert Path(item['path']).stat().st_mtime_ns == item['mtime_ns']
    except Exception as error:
        dump(out/'failure.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
