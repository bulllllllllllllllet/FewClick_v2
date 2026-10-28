"""Fresh same-GPU encoder proof for immutable full-valid cell-cache aliases.

Consumes accepted real cells, never reruns detection or changes cell selection.
This eight-input fixture cannot accept the full cache or any new cell fields.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pyvips
import torch

from .cell_cache_inputs import native_views
from .completion import atomic_json
from .fullcell_cache import append_view, transfer
from benchmarks.v5.external_evaluation.evaluate_adapted import load


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def equal_encoded(fresh, saved, path='encoded'):
    if isinstance(fresh, torch.Tensor):
        require(isinstance(saved, torch.Tensor) and fresh.shape == saved.shape and fresh.dtype == saved.dtype,
                'same encoded tensor structure: '+path)
        require(torch.isfinite(fresh).all() and torch.equal(fresh, saved.to(fresh.device)), 'exact fresh encoded tensor: '+path)
        return [dict(path=path, shape=list(fresh.shape), dtype=str(fresh.dtype), exact=True)]
    if isinstance(fresh, dict):
        require(isinstance(saved, dict) and fresh.keys() == saved.keys(), 'same encoded dictionary')
        return [item for key in fresh for item in equal_encoded(fresh[key], saved[key], path+'.'+str(key))]
    require(type(fresh) is type(saved) and isinstance(fresh, (tuple, list)) and len(fresh) == len(saved), 'same encoded sequence')
    return [item for i, value in enumerate(fresh) for item in equal_encoded(value, saved[i], path+'.'+str(i))]


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pid = os.getpid()
    def dump(name, value):
        atomic_json(a.output/name, value)
    try:
        torch.set_num_threads(4)
        pyvips.cache_set_max(0)
        root = a.root
        base = root/'fullcell_CRC_GPU_gate_1028_025000'
        acceptance = root/'fullcell_CRC_cache_accept_gate_1028_031000'
        inputs = root/'full_valid_inputs_full_1028_034700'
        full_acceptance = root/'full_valid_inputs_acceptance_1028_034700'
        summary = read(base/'summary.json')
        receipt = read(acceptance/'summary.json')
        require(summary['status'] == 'passed_main30_fullcell_cache_gate' and receipt['status'] == 'accepted_main30_fullcell_cache_gate'
                and receipt['source_summary_sha256'] == sha(base/'summary.json'), 'independently accepted real8 cell gate')
        require(read(full_acceptance/'summary.json')['source_summary_sha256'] == sha(inputs/'summary.json')
                and read(inputs/'summary.json')['inputs_sha256'] == sha(inputs/'inputs.jsonl'), 'accepted full28452 geometry')
        for p, digest in summary['signature']['sources'].items():
            require(sha(p) == digest, 'unchanged original cell/encoder source')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == summary['signature']['visible_devices'] == '1'
                and torch.cuda.is_available(), 'same accepted physical GPU1')
        free, total = torch.cuda.mem_get_info()
        require(free >= 12*1024**3, '12GiB free before small encoder fixture; stop if unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        require(sha(base/'cache_manifest.jsonl') == summary['cache_manifest_sha256'], 'accepted original cache manifest')
        entries = [json.loads(line) for line in (base/'cache_manifest.jsonl').open()]
        rows = [json.loads(line) for line in (inputs/'inputs.jsonl').open()]
        aliases = {r['existing_input_key']: r for r in rows if r['existing_input_key'] is not None}
        require(len(entries) == 8, 'all eight accepted fields, not a chosen result subset')
        training = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451')
        best = read(training/'best_checkpoint.json')
        require(sha(best['path']) == best['checkpoint_sha256'] == summary['signature']['checkpoint_sha256'], 'same selected full-label weights')
        model = load('fewclick_m3d_xcell', Path(best['path']), torch.device('cuda:0'),
                     sha('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400/contract.json'))
        sources = {str(p): sha(p) for p in [Path(__file__), Path('benchmarks/v5/wsi_global/cell_cache_inputs.py'),
                   Path('benchmarks/v5/wsi_global/fullcell_cache.py')]}
        dump('contract.json', dict(worker_pid=pid, sources=sources, base_summary_sha256=sha(base/'summary.json'),
             base_acceptance_sha256=sha(acceptance/'summary.json'), full_input_acceptance_sha256=sha(full_acceptance/'summary.json'),
             checkpoint_sha256=best['checkpoint_sha256'], physical_GPU='1', shared_device=True,
             query_GT_access=False, detector_calls=0, cell_feature_model_calls=0, tolerance='exact torch.equal',
             scope='same-GPU raw HE plus existing real cells re-encoding, fixture only', free_GPU_bytes_before_load=free))
        results = []
        zero = []
        with torch.inference_mode():
            for entry in entries:
                old = entry['identity']
                row = aliases[old['key']]
                require(old['geometry'] == dict(native_boxes=row['native_boxes'], native_mpp=row['native_mpp'], source_wsi=row['source_wsi']), 'all3 original physical boxes/source/MPP')
                require(row['kind'] == 'full_valid_query' and row['preparation'] == ('support_reuse_candidate' if old['kind'] == 'support' else 'remote_reuse_candidate'), 'explicit original role alias')
                require(sha(entry['path']) == entry['sha256'], 'original encoded file unchanged')
                payload = torch.load(entry['path'], map_location='cpu', weights_only=False)
                require(payload['identity'] == old and payload['checkpoint_sha256'] == best['checkpoint_sha256'], 'original payload identity')
                slide = pyvips.Image.new_from_file(row['source_wsi'], access='random')
                require([slide.width, slide.height] == row['native_wh'], 'actual native source header')
                raw = {}
                views = []
                for scale, native, image, details in native_views(slide, row):
                    c = old['cell_records'][scale]
                    require(c['scale'] == scale and sha(c['path']) == c['sha256'], 'unchanged original real NPZ')
                    native_digest = hashlib.sha256(native.tobytes()).hexdigest()
                    he_digest = hashlib.sha256(image.tobytes()).hexdigest()
                    require(native_digest == c['native_RGB_sha256'] and he_digest == old['RGB_array_sha256'][scale], 'fresh native/PIL pixels exact')
                    with np.load(c['path'], allow_pickle=False) as z:
                        arrays = {key: z[key] for key in z.files}
                    require(details['detection_size'] == c['details']['detection_size'] and details['detection_mpp_xy'] == c['details']['detection_mpp_xy'], 'preserved actual cell physical geometry')
                    append_view(raw, scale, image, arrays, torch.device('cuda:0'))
                    total_cells = int(arrays['total'])
                    if total_cells == 0:
                        zero.append(dict(key=old['key'], scale=scale))
                    views.append(dict(scale=scale, path=c['path'], sha256=c['sha256'], native_RGB_sha256=native_digest,
                                      HE_RGB_sha256=he_digest, true_cells=total_cells, retained_cells=len(arrays['features'])))
                encoded = model.core.encode_image(raw)
                tensor_checks = equal_encoded(encoded, payload['encoded'])
                fresh_path = a.output/('fresh_'+str(len(results))+'.pth')
                torch.save(transfer(encoded, 'cpu'), fresh_path)
                results.append(dict(original_key=old['key'], original_kind=old['kind'], full_valid_key=row['key'],
                    original_encoded_path=entry['path'], original_encoded_sha256=entry['sha256'],
                    fresh_encoded_path=str(fresh_path), fresh_encoded_sha256=sha(fresh_path),
                    full_valid_geometry=row, views=views, tensors=tensor_checks, exact=True))
                dump('status.json', dict(status='fresh_GPU_fullcell_alias_encode', worker_pid=pid, completed=len(results), total=8))
                del encoded, raw, payload
        require(len(results) == 8 and sum(r['original_kind'] == 'support' for r in results) == 4 and len(zero) == 2, 'all4 support/all4 remote and two actual zero-cell views')
        for p, digest in sources.items():
            require(sha(p) == digest, 'unchanged gate sources throughout')
        dump('results.json', results)
        dump('summary.json', dict(status='passed_fullcell_same_GPU_full_valid_alias_fixture', worker_pid=pid,
             source_sha256=sha(__file__), inputs=8, views=24, support_aliases=4, remote_aliases=4, zero_cell_views=zero,
             tensor_checks=sum(len(r['tensors']) for r in results), results_sha256=sha(a.output/'results.json'),
             peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.perf_counter()-started,
             scope='Existing accepted eight fields reencoded exactly on same GPU with real saved cells; no detector or feature extraction rerun.',
             remaining='Independent all24 source/NPZ/encoded artifact replay, full20456 cache acceptance, new7996 cells/cache, actual merged28452 predictions and complete quality/statistics.'))
    except Exception:
        dump('status.json', dict(status='failed', worker_pid=pid, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
