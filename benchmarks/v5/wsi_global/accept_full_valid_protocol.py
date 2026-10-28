"""Independently recount full-valid task prefixes and physical score domains."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = read(a.source/'summary.json')
    require(source['status'] == 'prepared_full_valid_evaluation_protocol_unaccepted' and source['source_sha256'] == sha('benchmarks/v5/wsi_global/full_valid_protocol.py'), 'actual frozen protocol producer')
    for name, digest in source['files'].items():
        require(sha(a.source/name) == digest, 'every protocol output digest')
    for pointer in list(source['sources'].values()) + list(source['receipts'].values()):
        require(sha(pointer['path']) == pointer['sha256'], 'all source and acceptance identities')
    root = a.root; original = root/'support_query_full_1027_173800'
    original_summary = read(original/'summary.json')
    for name, digest in original_summary['output_sha256'].items():
        require(sha(original/name) == digest, 'all original task/availability/source bytes')
    require(sha(root/'gt_components_full_1027_231500/task_domains.json') == read(root/'gt_components_full_1027_231500/summary.json')['tasks_sha256'], 'all original scoring component metadata')
    require(sha(root/'main30_full_1027_171918/slides_private.json') == read(root/'main30_full_1027_171918/summary.json')['slides_sha256'], 'original30-slide metadata identity')
    data = {s['wsi_id']:s for s in read(root/'main30_full_1027_171918/slides_private.json')}
    queries = {q['wsi_id']:q for q in read(original/'query_inputs.json')}
    plans = {(p['wsi_id'],p['target'],p['prompt_group']):p for p in read(original/'support_tasks_maximum.json')}
    domains = {(d['wsi_id'],d['target'],d['prompt_group'],d['budget']):d for d in read(root/'gt_components_full_1027_231500/task_domains.json')}
    availability = read(a.source/'availability.json')
    require(availability == read(original/'coverage.json'), 'all original unavailable reasons preserved verbatim')
    require(len(availability) == 330 and sum(c['status'] == 'available' for c in availability) == 276, 'all11 tissue labels/30WSIs, no background positive target')
    tasks = [json.loads(line) for line in (a.source/'tasks.jsonl').open()]
    identities = [(t['wsi_id'],t['target'],t['prompt_group'],t['budget']) for t in tasks]
    expected = {(sid,c,g,b) for sid,c,g in plans for b in ('3+2','1+1','3+1','5+1')}
    require(len(tasks) == len(set(identities)) == 5520 and set(identities) == expected, 'all original5520 budget-expanded tasks, no extras/omissions')
    require(len({t['task_id'] for t in tasks}) == 5520, 'unique task IDs')
    for t in tasks:
        p = plans[t['wsi_id'],t['target'],t['prompt_group']]
        npos, nneg = map(int,t['budget'].split('+'))
        require(t['task_id'] == f'{t["wsi_id"]}:{t["target"]}:{t["prompt_group"]}:{t["budget"]}', 'canonical whole-domain task identity')
        require(t['support_id'] == p['support_id'] and t['source_tile'] == p['source_tile'] and t['gt_simulated_source_clicks'] is True, 'unchanged actual source input')
        for key in ('positive_xy','positive_native_xy'):
            require(t[key] == p[key][:npos] and len(t[key]) == npos, 'every positive normalized/native source prefix')
        for key in ('negative_xy','negative_native_xy'):
            require(t[key] == p[key][:nneg] and len(t[key]) == nneg, 'every negative normalized/native source prefix')
        require(t['scoring_only_positive_component_ids'] == domains[t['wsi_id'],t['target'],t['prompt_group'],t['budget']]['positive_component_ids'], 'unchanged scoring-only component censor IDs')
    slides = read(a.source/'slides.json')
    require(len(slides) == len({s['wsi_id'] for s in slides}) == 30 and {s['wsi_id'] for s in slides} == set(data), 'all original30 slides')
    counts = Counter();storage = Counter();fine_fields = 0
    for s in slides:
        sid = s['wsi_id']; r = data[sid]; q = queries[sid]
        require((s['source_wsi'],s['patient_id_as_recorded'],s['split']) == (r['source_wsi'],r['patient_id_as_recorded'],r['split']), 'original patient/slide/spatial split provenance')
        require(s['native_wh'] == r['native_wh'] and s['gt_wh'] == r['gt_wh'] and s['native_mpp'] == r['native_mpp'], 'real physical shape and resolution')
        require(s['source_cut_native'] == q['source_cut_native'] and s['gap_native'] == q['gap_native'], 'unchanged source band and gap')
        require(sha(s['full_grid']) == s['full_grid_sha256'] == r['files'][r['grid']]['sha256'] and sha(q['grid']) == q['grid_sha256'], 'entire original grids')
        fine_fields += s['full_tiles']
        w,h = r['native_wh'];gw,gh = r['gt_wh'];sx,sy = w/gw,h/gh
        expected_bounds = []
        for which,path in [('full',r['grid']),('remote',q['grid'])]:
            grid = [json.loads(line) for line in Path(path).open()]
            xs = sorted({g['native_boxes'][0][0] for g in grid});ys = sorted({g['native_boxes'][0][1] for g in grid})
            fw,fh = grid[0]['native_boxes'][0][2:]
            require(len(grid) == len(xs)*len(ys) == s[which+'_tiles'], 'complete unfiltered Cartesian domain')
            b = [xs[0],ys[0],xs[-1]+fw,ys[-1]+fh]
            xx = np.flatnonzero((np.arange(gw)*sx >= b[0]) & ((np.arange(gw)+1)*sx <= b[2]))
            yy = np.flatnonzero((np.arange(gh)*sy >= b[1]) & ((np.arange(gh)+1)*sy <= b[3]))
            cells = [int(xx[0]),int(yy[0]),int(xx[-1])+1,int(yy[-1])+1]
            require(s[which+'_native_bounds'] == b and s[which+'_GT_bounds'] == cells, 'independent complete-GT-cell physical predicates')
            require(s['GT_to_native_scale_xy'] == [sx,sy], 'extent affine preserved without registration claim')
            expected_bounds.append(cells)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header,dict(descr='|u1',fortran_order=False,shape=(len(yy),(len(xx)+7)//8)))
            require(header.tell() == 128, 'actual NumPy packed mask header length')
            storage['full_valid' if which == 'full' else 'remote'] += sum(t['wsi_id'] == sid for t in tasks)*(header.tell()+len(yy)*((len(xx)+7)//8))
        full,remote = expected_bounds
        require(s['remote_crop_in_full_GT'] == [remote[0]-full[0],remote[1]-full[1],remote[2]-full[0],remote[3]-full[1]], 'all original remote crop offsets')
        require(all(full[i] <= remote[i] for i in (0,1)) and all(remote[i] <= full[i] for i in (2,3)), 'remote rectangle fully retained')
        require(s['available_pairs'] == sum(c['wsi_id'] == sid and c['status'] == 'available' for c in availability), 'per-WSI unavailable retention')
        counts['full_cells'] += (full[2]-full[0])*(full[3]-full[1]);counts['remote_cells'] += (remote[2]-remote[0])*(remote[3]-remote[1])
        b=s['full_native_bounds']; require(s['full_raw_rectangle_fraction'] == (b[2]-b[0])*(b[3]-b[1])/(w*h), 'raw rectangle coverage, no inferred tissue fraction')
    require(fine_fields == 28452, 'all28452 original valid fields')
    contract = read(a.source/'contract.json')
    require(contract['packed_mask_bytes_per_single_variant'] == dict(storage) and contract['expected_rows'] == dict(fewclick_m3d_image=11040,sam_ft=5520,wsi_sam_ft=5520,fewclick_m3d_xcell=11040), 'independent storage/variant denominators')
    require(contract['evaluation_scope'] == 'complete_valid4x_including_support_separate_diagnostic' and contract['phase'] == 'protocol_only_predictions_and_acceptance_pending' and not contract['query_prompt_maps'] and contract['probability_threshold'] == '>=0.5', 'fixed separate diagnostic and inference restrictions')
    result = dict(status='accepted_full_valid_metadata_protocol_not_predictions',worker_pid=os.getpid(),source_sha256=sha(__file__),source_summary_sha256=sha(a.source/'summary.json'),slides=30,tasks=5520,full_fields=fine_fields,
        counts=dict(counts),packed_mask_bytes_per_single_variant=dict(storage),available_pairs=276,unavailable_pairs=54,GT_label_arrays_read=0,prediction_arrays_read=0,seconds=time.perf_counter()-started,
        scope='All fixed source-prefix tasks, provenance, original complete GT-cell domains, remote crop offsets and packed-mask storage. No model/GT accuracy, actual mask invariance, boundary/statistical result or fullcell cache acceptance.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root','source','output'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
