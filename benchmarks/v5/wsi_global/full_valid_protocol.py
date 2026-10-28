"""Freeze full-valid WSI evaluation before predictions, using metadata only.

This diagnostic includes the support region. It never replaces the original
context-disjoint remote table. No image, label array or prediction is opened.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    root = a.root
    data = root / 'main30_full_1027_171918'
    original = root / 'support_query_full_1027_173800'
    full = root / 'full_valid_inputs_full_1028_034700'
    components = root / 'gt_components_full_1027_231500'
    receipts = {}
    for name, directory, expected in [
        ('input', root / 'full_valid_inputs_acceptance_1028_034700', 'accepted_all28452_full_valid_inputs_and_native24_bridge'),
        ('support', root / 'support_query_acceptance_1027_173800', 'accepted_source_band_query_protocol'),
        ('components', root / 'gt_components_acceptance_1027_231500', 'accepted_all30_GT_components_and_prompt_domains'),
        ('data', root / 'main30_acceptance_1027_171918', 'accepted_main30_data_and_geometry_qualified'),
    ]:
        receipt = read(directory / 'summary.json')
        require(receipt['status'] == expected, 'accepted ' + name)
        if name == 'support':
            require(receipt['actual_native_RGB_views'] == 744 and receipt['protocol_summary_sha256'] == sha(original / 'summary.json'), 'all accepted source images and points')
        receipts[name] = dict(path=str(directory / 'summary.json'), sha256=sha(directory / 'summary.json'))
    require(read(receipts['input']['path'])['source_summary_sha256'] == sha(full / 'summary.json'), 'same accepted all28452 inputs')
    require(read(receipts['components']['path'])['source_summary_sha256'] == sha(components / 'summary.json'), 'same accepted original components')
    require(read(receipts['data']['path'])['source_summary_sha256'] == sha(data / 'summary.json'), 'same accepted original full-GT geometry')
    require(sha(components / 'task_domains.json') == read(components / 'summary.json')['tasks_sha256'], 'accepted original scoring-only component IDs')
    original_summary = read(original / 'summary.json')
    for name, digest in original_summary['output_sha256'].items():
        require(sha(original / name) == digest, 'original support protocol bytes')
    full_summary = read(full / 'summary.json')
    require(sha(full / 'inputs.jsonl') == full_summary['inputs_sha256'], 'all physical input identities')
    metadata = read(data / 'slides_private.json')
    require(sha(data / 'slides_private.json') == read(data / 'summary.json')['slides_sha256'], 'all30 original provenance')
    queries = {q['wsi_id']: q for q in read(original / 'query_inputs.json')}
    entries = [json.loads(line) for line in (full / 'inputs.jsonl').open()]
    plans = read(original / 'support_tasks_maximum.json')
    domains = {(d['wsi_id'], d['target'], d['prompt_group'], d['budget']): d for d in read(components / 'task_domains.json')}
    coverage = read(original / 'coverage.json')
    availability = {(c['wsi_id'], c['target']): c['status'] == 'available' for c in coverage}
    require(len(availability) == 330 and sum(availability.values()) == 276, 'all original supported and unavailable tissue/WSI pairs')
    slides = []
    total_mask_bytes = Counter()
    for r in metadata:
        sid = r['wsi_id']; q = queries[sid]
        require(sha(r['grid']) == r['files'][r['grid']]['sha256'] and sha(q['grid']) == q['grid_sha256'], 'unchanged original full/remote grids')
        grid = [json.loads(line) for line in Path(r['grid']).open()]
        remote = [json.loads(line) for line in Path(q['grid']).open()]
        selected = [e for e in entries if e['wsi_id'] == sid]
        require([e['tile'] for e in selected] == [t['number'] for t in grid], 'original complete grid, no tissue filtering')
        bounds = r['geometry']['valid_fine_bounds_xyxy']
        boxes = [t['native_boxes'][0] for t in remote]
        rb = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[0]+b[2] for b in boxes), max(b[1]+b[3] for b in boxes)]
        w, h = r['native_wh']; gw, gh = r['gt_wh']; sx, sy = w/gw, h/gh
        def cells(b):
            return [math.ceil(b[0]/sx), math.ceil(b[1]/sy), math.floor(b[2]/sx), math.floor(b[3]/sy)]
        gt = cells(bounds); old_gt = cells(rb)
        require(all(e['native_boxes'][0][0] + e['native_boxes'][0][2] <= rb[0] for e in selected if e['preparation'] != 'remote_reuse_candidate'), 'new fine predictions cannot contribute to old remote crop')
        require(sum(e['preparation'] == 'remote_reuse_candidate' for e in selected) == len(remote), 'all old remote fields conserved')
        widths = [gt[2]-gt[0], old_gt[2]-old_gt[0]]
        heights = [gt[3]-gt[1], old_gt[3]-old_gt[1]]
        tasks = sum(p['wsi_id'] == sid for p in plans) * 4
        # NumPy v1 header is128 bytes for these uint8 2-D mask dimensions.
        for name, width, height in zip(('full_valid', 'remote'), widths, heights):
            total_mask_bytes[name] += tasks * (128 + height*((width+7)//8))
        slides.append(dict(wsi_id=sid, source_wsi=r['source_wsi'], patient_id_as_recorded=r['patient_id_as_recorded'], split=r['split'],
            native_wh=r['native_wh'], gt_wh=r['gt_wh'], native_mpp=r['native_mpp'], GT_to_native_scale_xy=[sx, sy],
            full_grid=r['grid'], full_grid_sha256=sha(r['grid']), full_tiles=len(grid), remote_tiles=len(remote),
            full_native_bounds=bounds, full_GT_bounds=gt, remote_native_bounds=rb, remote_GT_bounds=old_gt,
            remote_crop_in_full_GT=[old_gt[0]-gt[0], old_gt[1]-gt[1], old_gt[2]-gt[0], old_gt[3]-gt[1]],
            full_raw_rectangle_fraction=(bounds[2]-bounds[0])*(bounds[3]-bounds[1])/(w*h),
            source_cut_native=q['source_cut_native'], gap_native=q['gap_native'], available_pairs=sum(c['wsi_id'] == sid and c['status'] == 'available' for c in coverage)))
    tasks = []
    for p in plans:
        require(availability[p['wsi_id'], p['target']] and p['target'] != 2 and p['gt_simulated_source_clicks'], 'unchanged available positive tissue tasks')
        for budget in ('3+2', '1+1', '3+1', '5+1'):
            npos, nneg = map(int, budget.split('+'))
            d = domains[p['wsi_id'], p['target'], p['prompt_group'], budget]
            tasks.append(dict(task_id=f'{p["wsi_id"]}:{p["target"]}:{p["prompt_group"]}:{budget}', wsi_id=p['wsi_id'], target=p['target'], prompt_group=p['prompt_group'], budget=budget,
                support_id=p['support_id'], source_tile=p['source_tile'], gt_simulated_source_clicks=True,
                positive_xy=p['positive_xy'][:npos], negative_xy=p['negative_xy'][:nneg],
                positive_native_xy=p['positive_native_xy'][:npos], negative_native_xy=p['negative_native_xy'][:nneg],
                scoring_only_positive_component_ids=d['positive_component_ids']))
    require(len(tasks) == len({t['task_id'] for t in tasks}) == 5520 and len(slides) == 30, 'all fixed tasks and slides')
    scope = dict(evaluation_scope='complete_valid4x_including_support_separate_diagnostic',
        inference='Original selected weights, immutable encoded inputs and exact original support point prefixes. Query GT/component/distance metadata never enters model, prompts, candidate selection or thresholds.',
        query_prompt_maps=False, query_encoder_calls=0, probability_threshold='>=0.5', native_resize='PIL bilinear FP32', stitch='original row-major FP32 uniform overlap mean',
        GT_sampling='nearest native pixel at centre of every complete original GT cell inside fixed rectangle; original GT255 ignored; GT0 valid; background2 negative',
        standard='All valid full-domain pixels; includes source region and support-hit components. May be easier than strict remote and must not replace it.',
        unprompted='Secondary: censor entire target components containing source positives using full-GT component IDs; preserve all nontarget false positives. Never censor for the standard diagnostic.',
        BF1='Same fixed physical Euclidean2um, both-sided4-neighbor boundaries; frame/void handling and NA eligibility inherited from main30 contract.',
        statistics='Each budget separate; average5prompt groups per WSI/target then targets per WSI then WSIs equally. Paired WSI bootstrap conditional on spatial cohort; no patient independence or training-seed inference.',
        remote_invariance_acceptance='Every original remote512 probability digest and every saved original remote Boolean mask cell must match full-valid crop. Metadata nonoverlap alone is not prediction acceptance.',
        full_valid_coverage='All valid4x grid rectangles, not100% tissue, not anatomically independently registered, not a cross-tile graph.',
        availability='All54 unavailable source pairs retained with original reasons; no new support selection or GT-guided replacement.',
        fullcell='XCell requires its own real native detections/74 features/true counts/caches, model gates and acceptance; image-only gates cannot certify it.',
        privileged_reference='Closed12 whole-class model, no clicks,360rows, separate from interactive arms.',
        expected_rows={'fewclick_m3d_image':11040, 'sam_ft':5520, 'wsi_sam_ft':5520, 'fewclick_m3d_xcell':11040},
        packed_mask_bytes_per_single_variant=dict(total_mask_bytes), storage_qualification='Exact planned mask payload plus128-byte NumPy headers only; excludes caches, detector/features, provenance/digests/NPZ and native exports. Not a free-space guarantee.',
        phase='protocol_only_predictions_and_acceptance_pending')
    files = {'slides.json': slides, 'availability.json': coverage, 'contract.json': scope}
    for name, value in files.items():
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    with (a.output/'tasks.jsonl').open('x') as f:
        for t in tasks:
            f.write(json.dumps(t)+'\n')
    sources = dict(data_summary=dict(path=str(data/'summary.json'), sha256=sha(data/'summary.json')),
        original_summary=dict(path=str(original/'summary.json'), sha256=sha(original/'summary.json')),
        full_input_summary=dict(path=str(full/'summary.json'), sha256=sha(full/'summary.json')),
        component_summary=dict(path=str(components/'summary.json'), sha256=sha(components/'summary.json')))
    result = dict(status='prepared_full_valid_evaluation_protocol_unaccepted', worker_pid=os.getpid(), source_sha256=sha(__file__), receipts=receipts, sources=sources,
        slides=30, tasks=5520, available_pairs=276, unavailable_pairs=54, query_label_arrays_opened=0, prediction_arrays_opened=0,
        files={name: sha(a.output/name) for name in (*files, 'tasks.jsonl')}, seconds=time.perf_counter()-started,
        remaining='Independent protocol audit; full-valid GPU model/decode gates and predictions, saved-mask/GT audit, actual remote invariance, BF1/statistics; fullcell input extension.')
    (a.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
