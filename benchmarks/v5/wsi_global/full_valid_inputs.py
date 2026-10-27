"""Plan full valid4x inputs without reading labels or altering remote tasks."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyvips

from .cell_cache_inputs import native_views
from .completion import atomic_json


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok: raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False, parents=True); start = time.perf_counter(); pyvips.cache_set_max(0)
    root = a.root; data = root/'main30_full_1027_171918'; protocol = root/'support_query_full_1027_173800'
    accepted = root/'main30_acceptance_1027_171918/summary.json'; pa = root/'support_query_acceptance_1027_173800/summary.json'
    old = root/'cell_native_inputs_full_1028_012700'; old_accept = root/'cell_native_inputs_acceptance_1028_023400/summary.json'
    require(read(accepted)['source_summary_sha256'] == sha(data/'summary.json') and read(pa)['protocol_summary_sha256'] == sha(protocol/'summary.json'), 'independently accepted full grids and frozen remote protocol')
    require(read(old_accept)['source_summary_sha256'] == sha(old/'summary.json') and read(old/'summary.json')['input_manifest_sha256'] == sha(old/'inputs.jsonl'), 'all accepted native input geometry')
    signature = dict(source_sha256=sha(__file__), native_bridge_sha256=sha('benchmarks/v5/wsi_global/cell_cache_inputs.py'),
                     data_summary_sha256=sha(data/'summary.json'), protocol_summary_sha256=sha(protocol/'summary.json'),
                     original_input_summary_sha256=sha(old/'summary.json'), input_acceptance_sha256=sha(old_accept))
    if not a.gate:
        require(a.verified_gate is not None, 'actual native8x3 bridge gate required')
        g = read(a.verified_gate/'summary.json')
        require(g['status'] == 'passed_full_valid_input_native_gate' and g['signature'] == signature, 'same source/data native bridge gate')
    original = [json.loads(line) for line in (old/'inputs.jsonl').open()]
    by_geometry = {(r['wsi_id'], tuple(map(tuple, r['native_boxes']))):r for r in original}
    require(len(by_geometry) == 20456, 'unique previously accepted support/query footprints')
    qdata = {q['wsi_id']:q for q in read(protocol/'query_inputs.json')}
    rows = []; coverage = []
    for r in read(data/'slides_private.json'):
        sid = r['wsi_id']; q = qdata[sid]; grid = Path(r['grid'])
        require(sha(grid) == r['files'][str(grid)]['sha256'] and sha(q['grid']) == q['grid_sha256'], 'immutable full/remote grids')
        full = [json.loads(line) for line in grid.open()]; remote = {t['number']:t for t in map(json.loads, Path(q['grid']).open())}
        first_remote_x = min(t['native_boxes'][0][0] for t in remote.values()); counts = Counter(); zones = Counter()
        for t in full:
            number = t['number']; boxes = t['native_boxes']; w,h = r['native_wh']; centres = []
            for x,y,bw,bh in boxes:
                require(0 <= x and 0 <= y and min(bw,bh)>0 and x+bw <= w and y+bh <= h, 'every true native input footprint inside source')
                centres.append((x+bw/2, y+bh/2))
            require(all(max(abs(c[0]-centres[0][0]),abs(c[1]-centres[0][1])) <= .5 for c in centres), 'frozen integer-rounding half-pixel centre policy')
            prior = by_geometry.get((sid,tuple(map(tuple,boxes))))
            kind = 'remote_reuse_candidate' if number in remote else 'support_reuse_candidate' if prior else 'new_input'
            if number in remote:
                require(t == remote[number] and prior is not None and prior['kind'] == 'query', 'every original remote record preserved exactly')
            elif prior:
                require(prior['kind'] == 'support', 'nonremote existing input is selected support')
            require(kind == 'remote_reuse_candidate' or boxes[0][0]+boxes[0][2] <= first_remote_x, 'added fine views do not overlap strict remote fine domain')
            coarse = boxes[2]
            zone = 'source_band' if coarse[0]+coarse[2] <= q['source_cut_native'] else 'strict_remote' if coarse[0] >= q['source_cut_native']+q['gap_native'] else 'context_crosses_source_or_gap'
            require((zone == 'strict_remote') == (number in remote), 'all4x geometric domain membership')
            counts[kind] += 1; zones[zone] += 1
            rows.append(dict(index=len(rows), key=f'{sid}:full_valid:{number}', kind='full_valid_query', tile=number,
                wsi_id=sid, source_wsi=r['source_wsi'], native_wh=r['native_wh'], native_mpp=r['native_mpp'],
                native_boxes=boxes, source_cut_native=q['source_cut_native'], gap_native=q['gap_native'], zone=zone,
                preparation=kind, existing_input_key=prior['key'] if prior else None, existing_input_index=prior['index'] if prior else None,
                reuse_status='geometry_candidate_only_requires_method_specific_cache_gate' if prior else 'unprepared'))
        coverage.append(dict(wsi_id=sid, patient_id_as_recorded=r['patient_id_as_recorded'], split=r['split'],
            full_grid=str(grid), full_grid_sha256=sha(grid), full_tiles=len(full), counts=dict(counts), zones=dict(zones),
            valid_fine_bounds_native=r['geometry']['valid_fine_bounds_xyxy'], GT_full_cell_domain=r['GT_full_cell_domain_xyxy'],
            native_area_fraction=r['geometry']['covered_area_fraction'], tissue_filter=False,
            scoring_scope='Full valid domain includes supports/context crossings; separate application diagnostic, never a disjoint remote benchmark'))
    totals = Counter(r['preparation'] for r in rows)
    require(len(rows) == 28452 and totals == dict(remote_reuse_candidate=20208,support_reuse_candidate=248,new_input=7996), 'complete fixed original grids and cache candidate accounting')
    atomic_json(a.output/'contract.json', dict(worker_pid=os.getpid(),signature=signature,gate=a.gate,query_GT_access=False,
        scope='Geometry and native-read preparation only; no cells, encoder caches, probability reuse or quality accepted'))
    with (a.output/'inputs.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    atomic_json(a.output/'coverage.json',coverage)
    evidence = []
    if a.gate:
        old_gate = [json.loads(line) for line in (root/'cell_native_input_gate_1028_012500/inputs.jsonl').open()]
        choices = [next(r for r in rows if r['existing_input_key'] == next(p['key'] for p in old_gate if p['kind'] == k)) for k in ('support','query')]
        fresh = [r for r in rows if r['preparation'] == 'new_input']
        choices += [fresh[0],fresh[-1],next(r for r in reversed(fresh) if r['wsi_id']==fresh[0]['wsi_id']),next(r for r in fresh if r['native_boxes'][0][2]%2)]
        chosen = {r['key']:r for r in choices}
        for r in fresh[len(fresh)//2:]+fresh:
            if len(chosen) == 8: break
            chosen.setdefault(r['key'],r)
        old_refs = {r['key']:r for r in old_gate}
        for row in chosen.values():
            slide = pyvips.Image.new_from_file(row['source_wsi'],access='random')
            require([slide.width,slide.height] == row['native_wh'], 'real source header')
            views = []; folder = a.output/row['key'].replace(':','_'); folder.mkdir()
            for scale,native,image,metadata in native_views(slide,row):
                if row['existing_input_key'] in old_refs:
                    ref = old_refs[row['existing_input_key']]['HE_reference'][scale]
                    require(sha(ref['path']) == ref['sha256'] and np.array_equal(image,np.asarray(Image.open(ref['path']).convert('RGB'))), 'actual source/query HE equals original accepted input')
                path = folder/f'HE_scale{scale}.png';Image.fromarray(image).save(path)
                views.append(dict(scale=scale,native_RGB_sha256=hashlib.sha256(native.tobytes()).hexdigest(),HE_array_sha256=hashlib.sha256(image.tobytes()).hexdigest(),path=str(path),sha256=sha(path),**metadata))
            evidence.append(dict(key=row['key'],preparation=row['preparation'],views=views))
        atomic_json(a.output/'native_evidence.json',evidence)
    result = dict(status='passed_full_valid_input_native_gate' if a.gate else 'complete_full_valid_input_manifest_unaccepted',worker_pid=os.getpid(),signature=signature,
        inputs=len(rows),views=3*len(rows),slides=len(coverage),counts=dict(totals),interactive_additional_candidates=7996,closed12_additional_inputs=8244,
        inputs_sha256=sha(a.output/'inputs.jsonl'),coverage_sha256=sha(a.output/'coverage.json'),native_probes=len(evidence),
        native_evidence_sha256=sha(a.output/'native_evidence.json') if a.gate else None,seconds=time.perf_counter()-start,
        remaining='Independent all-input/native bridge audit; method-specific actual encoder/cache reuse gates; extra cell/logit caches, full-valid predictions/coverage/quality acceptance. Current remote benchmark unchanged.')
    atomic_json(a.output/'summary.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);run(p.parse_args())
