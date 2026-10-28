"""Independently replay all fresh alias tensors and raw RGB without a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyvips
import torch


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def tensors(tree, path='encoded'):
    if isinstance(tree, torch.Tensor):
        return {path: tree}
    if isinstance(tree, dict):
        return {name: tensor for key in tree for name, tensor in tensors(tree[key], path+'.'+str(key)).items()}
    require(isinstance(tree, (list, tuple)), 'valid encoded structure')
    return {name: tensor for i, value in enumerate(tree) for name, tensor in tensors(value, path+'.'+str(i)).items()}


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    pyvips.cache_set_max(0)
    summary = read(a.source/'summary.json')
    contract = read(a.source/'contract.json')
    require(summary['status'] == 'passed_fullcell_same_GPU_full_valid_alias_fixture', 'actual finished GPU fixture')
    require(sha(a.source/'results.json') == summary['results_sha256'], 'fresh output references')
    for path, digest in contract['sources'].items():
        require(sha(path) == digest, 'same original gate source')
    root = a.root
    old = root/'fullcell_CRC_GPU_gate_1028_025000'
    old_receipt = root/'fullcell_CRC_cache_accept_gate_1028_031000/summary.json'
    require(sha(old/'summary.json') == contract['base_summary_sha256'] and sha(old_receipt) == contract['base_acceptance_sha256'], 'original accepted gate bound')
    require(read(old_receipt)['source_summary_sha256'] == sha(old/'summary.json'), 'original independent acceptance')
    original = {e['identity']['key']: e for e in map(json.loads, (old/'cache_manifest.jsonl').open())}
    rows = read(a.source/'results.json')
    require(len(rows) == 8 and {r['original_key'] for r in rows} == set(original), 'all original8 inputs retained')
    full = root/'full_valid_inputs_full_1028_034700'
    all_inputs = {r['key']: r for r in map(json.loads, (full/'inputs.jsonl').open())}
    checks = []
    zero = []
    n_tensors = 0
    for r in rows:
        entry = original[r['original_key']]
        row = all_inputs[r['full_valid_key']]
        require(row == r['full_valid_geometry'] and row['existing_input_key'] == r['original_key'], 'full-grid alias matches original source')
        require(entry['path'] == r['original_encoded_path'] and entry['sha256'] == r['original_encoded_sha256'] == sha(entry['path']), 'original payload preserved')
        require(sha(r['fresh_encoded_path']) == r['fresh_encoded_sha256'], 'new same-GPU encoder bytes bound')
        saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
        fresh = torch.load(r['fresh_encoded_path'], map_location='cpu', weights_only=False)
        a_tensors, b_tensors = tensors(saved['encoded']), tensors(fresh)
        require(a_tensors.keys() == b_tensors.keys() == {t['path'] for t in r['tensors']}, 'all fresh/saved tensor paths')
        for key, value in a_tensors.items():
            other = b_tensors[key]
            require(value.dtype == other.dtype and value.shape == other.shape and torch.isfinite(other).all()
                    and torch.equal(value, other), 'exact entire fresh GPU encoding on disk: '+key)
        n_tensors += len(a_tensors)
        slide = pyvips.Image.new_from_file(row['source_wsi'], access='random')
        for v in r['views']:
            scale = v['scale']
            x, y, w, h = row['native_boxes'][scale]
            native = np.frombuffer(slide.crop(x,y,w,h).extract_band(0,n=3).write_to_memory(), np.uint8).reshape(h,w,3)
            image = np.asarray(Image.fromarray(native).resize((512,512), Image.Resampling.BILINEAR))
            c = entry['identity']['cell_records'][scale]
            require(v['path'] == c['path'] and sha(c['path']) == c['sha256'] == v['sha256'], 'same accepted NPZ')
            require(hashlib.sha256(native.tobytes()).hexdigest() == v['native_RGB_sha256'] == c['native_RGB_sha256'], 'independent fresh native RGB')
            require(hashlib.sha256(image.tobytes()).hexdigest() == v['HE_RGB_sha256'] == entry['identity']['RGB_array_sha256'][scale], 'independent HE512 bridge')
            with np.load(c['path'], allow_pickle=False) as z:
                total = int(z['total'])
                features = z['features']
                require(total == v['true_cells'] and len(features) == v['retained_cells'] == min(255,total)
                        and features.shape[1] == 74 and np.isfinite(features).all(), 'real retained74 and uncapped counts')
            if total == 0:
                zero.append(dict(key=r['original_key'], scale=scale))
            checks.append(dict(key=r['full_valid_key'], scale=scale, original_role=r['original_kind'], native_and_NPZ_exact=True))
    require(len(checks) == 24 and zero == summary['zero_cell_views'] and len(zero) == 2
            and n_tensors == summary['tensor_checks'], 'all24 views/two zero cases/all tensors')
    result = dict(status='accepted_fullcell_same_GPU_full_valid_alias_fixture_only', worker_pid=os.getpid(),
        source_sha256=sha(__file__), source_summary_sha256=sha(a.source/'summary.json'), inputs=8,
        views=24, support_aliases=sum(r['original_kind']=='support' for r in rows), remote_aliases=sum(r['original_kind']=='query' for r in rows),
        tensor_checks=n_tensors, zero_cell_views=zero, independent_checks=checks, seconds=time.perf_counter()-started,
        scope='Fresh GPU encoder/disk replay for existing real8 fields; original cells reused and original payload identity preserved. No detector/reg64 rerun, new7996 fields, full20456 cache or full28452 prediction acceptance.')
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'independent_checks'}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
