"""Independent new native-cell/74D/byte audit with same-GPU encoder probes.

Rebuilds geometric features from instances and saved reg64, not the detector or
reg64 extraction itself. Gate probes all8 inputs; full cache probes fixed5.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import pyvips
import torch

from .accept_fullcell_cache import audit_view
from .accept_fullcell_alias_gate import tensors
from .completion import atomic_json
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    pid = os.getpid()
    def dump(n, d):
        atomic_json(a.output/n, d)
    try:
        torch.set_num_threads(4)
        pyvips.cache_set_max(0)
        s = read(a.source/'summary.json')
        gate = s['status'] == 'passed_fullcell_valid_extension_gate'
        require(gate or s['status'] == 'complete_fullcell_valid_extension_unaccepted', 'complete actual extension, not process launch')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == s['signature']['visible_devices'] == '1'
                and torch.cuda.is_available(), 'same physical GPU1 encoder probes')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = a.root
        full = root/'full_valid_inputs_full_1028_034700'
        require(sha(full/'summary.json') == s['signature']['input_summary_sha256'], 'same complete native input source')
        require(sha(a.source/'inputs.jsonl') == s['input_manifest_sha256'] and sha(a.source/'cache_manifest.jsonl') == s['cache_manifest_sha256'], 'all prepared input/cache manifest bytes')
        for p, digest in s['signature']['sources'].items():
            require(sha(p) == digest, 'unchanged producer dependency: '+p)
        dependencies = {p:sha(p) for p in ('benchmarks/v5/wsi_global/accept_fullcell_cache.py',
            'benchmarks/v5/wsi_global/audit_fullcell_inputs.py', 'benchmarks/v5/wsi_global/accept_fullcell_alias_gate.py')}
        if not gate:
            require(a.gate_acceptance is not None, 'same independent actual8 receipt required')
            ga = read(a.gate_acceptance/'summary.json')
            require(ga['status'] == 'accepted_fullcell_valid_extension_gate' and ga['producer_signature'] == s['signature']
                    and ga['source_sha256'] == sha(__file__) and ga['dependencies'] == dependencies, 'same independent auditor recipe gate')
        all_new = [r for r in map(json.loads, (full/'inputs.jsonl').open()) if r['preparation'] == 'new_input']
        require(len(all_new) == 7996, 'all original7996 new positions')
        if gate:
            native = read(root/'full_valid_inputs_gate_1028_034700/native_evidence.json')
            keys = {p['key'] for p in native if p['preparation'] == 'new_input'}
            for r in all_new:
                if len(keys) == 8:
                    break
                keys.add(r['key'])
            expected = [r for r in all_new if r['key'] in keys]
        else:
            expected = all_new
        rows = list(map(json.loads, (a.source/'inputs.jsonl').open()))
        entries = list(map(json.loads, (a.source/'cache_manifest.jsonl').open()))
        require(rows == expected and len(entries) == len(rows) == s['inputs'], 'complete fixed native ordering and no alias redetection')
        probe_indices = set(range(8)) if gate else {0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1}
        base = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        best = read(base/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
        require(sha(best['path']) == s['signature']['checkpoint_sha256'], 'actual selected XCell model')
        device = torch.device('cuda:0')
        model = load('fewclick_m3d_xcell', Path(best['path']), device, sha(base/'tasks_1025_125400/contract.json'))
        checks = []
        zero = []
        total_cells = retained = tensor_checks = 0
        with torch.inference_mode():
            for index, (r, e) in enumerate(zip(rows, entries)):
                d = e['identity']
                require(e['index'] == r['index'] and d['key'] == r['key'] and d['kind'] == 'full_valid_query'
                    and d['wsi_id'] == r['wsi_id'], 'actual new field role/id')
                require(d['geometry'] == dict(tile=r['tile'], native_boxes=r['native_boxes'], native_mpp=r['native_mpp'], source_wsi=r['source_wsi']), 'all3 physical input boxes')
                require(Path(e['path']).stat().st_size == e['bytes'] and sha(e['path']) == e['sha256'], 'all actual encoded cache bytes')
                payload = torch.load(e['path'], map_location='cpu', weights_only=False)
                require(payload['identity'] == d and payload['checkpoint_sha256'] == best['checkpoint_sha256']
                    and payload['method'] == 'fewclick_m3d_xcell', 'same checkpoint and actual payload')
                for t in tensors(payload['encoded']).values():
                    require(torch.isfinite(t).all(), 'every encoded value finite')
                probe = index in probe_indices
                if probe:
                    slide = pyvips.Image.new_from_file(r['source_wsi'], access='random')
                    require([slide.width,slide.height] == r['native_wh'], 'actual source header')
                    raw = {}
                require(len(d['cell_records']) == len(d['RGB_array_sha256']) == 3, 'all3 actual cell views')
                for scale, c in enumerate(d['cell_records']):
                    arrays, detail = audit_view(c, r['native_boxes'][scale], r['native_mpp'])
                    require(c['scale'] == scale, 'actual ordered cell view')
                    total_cells += detail['total']
                    retained += detail['retained']
                    if detail['total'] == 0:
                        zero.append(dict(key=r['key'], scale=scale))
                    if probe:
                        x,y,w,h = r['native_boxes'][scale]
                        native = np.frombuffer(slide.crop(x,y,w,h).extract_band(0,n=3).write_to_memory(), np.uint8).reshape(h,w,3)
                        image = np.asarray(Image.fromarray(native).resize((512,512), Image.Resampling.BILINEAR))
                        require(hashlib.sha256(native.tobytes()).hexdigest() == c['native_RGB_sha256']
                            and hashlib.sha256(image.tobytes()).hexdigest() == d['RGB_array_sha256'][scale], 'fresh native/PIL input on fixed probes')
                        suffix = '' if scale == 0 else f'_{scale}'
                        raw['image'+suffix] = he_to_tensor(image.copy())[None].to(device)
                        raw['cells'+suffix] = torch.from_numpy(arrays['features'].copy())[None].to(device)
                        raw['cell_valid'+suffix] = torch.ones(1,len(arrays['features']),dtype=torch.bool,device=device)
                        raw['total_cell_count' if scale == 0 else f'total_{scale}'] = torch.tensor([detail['total']],dtype=torch.float32,device=device)
                if probe:
                    fresh = tensors(model.core.encode_image(raw))
                    saved = tensors(payload['encoded'])
                    require(fresh.keys() == saved.keys(), 'all original encoder tensor paths')
                    for key, value in fresh.items():
                        require(torch.equal(value, saved[key].to(device)), 'exact independent fresh GPU encode: '+key)
                    tensor_checks += len(fresh)
                    checks.append(dict(key=r['key'], views=3, tensor_checks=len(fresh), exact=True))
                    del raw, fresh
                dump('status.json', dict(status='auditing_new_fullcell_inputs', worker_pid=pid, completed=index+1, total=len(rows)))
        require((total_cells, retained, zero) == (s['sum_true_instances_across_overlapping_views'], s['sum_retained_across_views'], s['zero_cell_views']), 'all true/retained/zero view counts')
        result = dict(status='accepted_fullcell_valid_extension_gate' if gate else 'accepted_fullcell_valid_extension_qualified',
            worker_pid=pid, source_sha256=sha(__file__), source_summary_sha256=sha(a.source/'summary.json'),
            producer_signature=s['signature'], dependencies=dependencies, inputs=len(rows), views=3*len(rows),
            sum_true_instances_across_overlapping_views=total_cells, sum_retained_across_views=retained,
            zero_cell_views=zero, fresh_encoder_probes=checks, tensor_checks=tensor_checks, seconds=time.perf_counter()-started,
            scope='All saved instance/physical geometry/count/unknown+reg74/encoded byte audits; raw RGB/fresh encoder replay at all8 gate or fixed5 full fields. No independent detector/reg64 extraction rerun or subset-invariance claim.',
            remaining='Formal7996 requires accepted original20456; merged28452 real-cell caches and full-valid predictions/quality not accepted by this receipt.')
        dump('summary.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('producer_signature','fresh_encoder_probes')}), flush=True)
    except Exception:
        dump('status.json', dict(status='failed', worker_pid=pid, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root','source','output'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--gate-acceptance', type=Path)
    run(p.parse_args())
