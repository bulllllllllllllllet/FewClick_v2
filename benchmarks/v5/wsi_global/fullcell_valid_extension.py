"""Actual native cells and XCell caches for the fixed new full-valid fields."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pyvips
import torch

from .cell_cache_inputs import native_views
from .completion import atomic_json
from .fullcell_cache import append_view, finite, transfer
from .fullcell_alias_gate import equal_encoded
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector, CachedCenterCropper, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS


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


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    pid = os.getpid()
    def dump(name, value):
        atomic_json(a.output/name, value)
    try:
        torch.set_num_threads(4)
        pyvips.cache_set_max(0)
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and torch.cuda.is_available(), 'same original GPU1 required')
        free, _ = torch.cuda.mem_get_info()
        require(free >= 20*1024**3, '20GiB available before actual new-field detector gate; no retry/fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        root = a.root
        inputs = root/'full_valid_inputs_full_1028_034700'
        ia = root/'full_valid_inputs_acceptance_1028_034700/summary.json'
        original = root/'fullcell_CRC_GPU_gate_1028_025000'
        oa = root/'fullcell_CRC_cache_accept_gate_1028_031000/summary.json'
        old = read(original/'summary.json')
        require(read(ia)['source_summary_sha256'] == sha(inputs/'summary.json') and read(inputs/'summary.json')['inputs_sha256'] == sha(inputs/'inputs.jsonl'), 'accepted full original geometry')
        require(read(oa)['source_summary_sha256'] == sha(original/'summary.json') and read(oa)['status'] == 'accepted_main30_fullcell_cache_gate', 'original real-cell recipe accepted')
        for p, digest in old['signature']['sources'].items():
            require(sha(p) == digest, 'unchanged canonical detector/features/encoder')
        records = [r for r in map(json.loads, (inputs/'inputs.jsonl').open()) if r['preparation'] == 'new_input']
        require(len(records) == 7996 and all(r['existing_input_key'] is None for r in records), 'exact newly required cells, no detector rerun for aliases')
        if a.gate:
            probes = read(root/'full_valid_inputs_gate_1028_034700/native_evidence.json')
            keys = {p['key'] for p in probes if p['preparation'] == 'new_input'}
            require(len(keys) == 6, 'six originally accepted new native probes')
            for r in records:
                if len(keys) == 8:
                    break
                keys.add(r['key'])
            records = [r for r in records if r['key'] in keys]
        require(len(records) == (8 if a.gate else 7996), 'fixed gate/full counts')
        base = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        best = read(base/'adaptation_1025_131451/train_fewclick_m3d_xcell_1025_131451/best_checkpoint.json')
        require(sha(best['path']) == best['checkpoint_sha256'] == old['signature']['checkpoint_sha256'], 'unchanged model weights')
        sources = {str(p): sha(p) for p in [Path(__file__), Path('benchmarks/v5/wsi_global/fullcell_cache.py'),
            Path('benchmarks/v5/wsi_global/fullcell_alias_gate.py'), Path('benchmarks/v5/wsi_global/cell_cache_inputs.py'),
            Path(inspect.getfile(inspect.unwrap(prepare_variant))), Path(inspect.getfile(InstanSegDetector)), Path(inspect.getfile(load_feature_models)),
            Path('benchmarks/v5/paper_multiscale/model.py'), Path('benchmarks/v5/external_evaluation/evaluate_adapted.py')]}
        weights = {k: dict(path=str(v), sha256=sha(v)) for k, v in WEIGHTS.items() if k in ('ctranspath', 'xcell')}
        weights['instanseg'] = dict(path=str(DEFAULT_INSTANSEG_CHECKPOINT), sha256=sha(DEFAULT_INSTANSEG_CHECKPOINT))
        require(weights == old['signature']['weights'], 'same three actual feature/detector weights')
        signature = dict(sources=sources, weights=weights, checkpoint_sha256=best['checkpoint_sha256'],
            input_summary_sha256=sha(inputs/'summary.json'), input_acceptance_sha256=sha(ia),
            original_gate_acceptance_sha256=sha(oa), device='cuda:0', visible_devices='1',
            torch=torch.__version__, cuda=torch.version.cuda, cpu_threads=4, seed=20261025,
            cell_order='manifest ascending then scale0/1/2; independent serial RNG stream; no subset or partition invariance claim')
        if not a.gate:
            require(a.verified_gate is not None and a.gate_acceptance is not None, 'actual new-field gate and independent receipt required')
            require(read(a.verified_gate/'summary.json')['signature'] == signature
                    and read(a.gate_acceptance/'summary.json')['status'] == 'accepted_fullcell_valid_extension_gate'
                    and read(a.gate_acceptance/'summary.json')['source_summary_sha256'] == sha(a.verified_gate/'summary.json'), 'same accepted new-field recipe')
            ba = root/'fullcell_CRC_cache_accept_full_1028_031000/summary.json'
            require(read(ba)['status'] == 'accepted_main30_fullcell_cache_qualified'
                    and read(ba)['source_summary_sha256'] == sha(root/'fullcell_CRC_cache_full_1028_031000/summary.json'), 'full original20456 accepted before formal extension')
        dump('contract.json', dict(worker_pid=pid, signature=signature, gate=a.gate, inputs=len(records),
             query_GT_access=False, failure_policy='stop preserve; no retry/resume/fallback', shared_GPU=True,
             original20456_modified=False, free_GPU_bytes_before_load=free))
        with (a.output/'inputs.jsonl').open('x') as stream:
            for r in records:
                stream.write(json.dumps(r)+'\n')
        # Preserve the original constructor order and actual preparation function.
        model = load('fewclick_m3d_xcell', Path(best['path']), device, sha(base/'tasks_1025_125400/contract.json'))
        detector = InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT, device)
        ctp, xcell, compatibility = load_feature_models(device)
        cropper = CachedCenterCropper()
        dump('feature_compatibility.json', compatibility)
        zero = []
        total_cells = retained = encoded_bytes = tensor_checks = 0
        timing = dict(native_read=0., cell_prepare=0., encode=0., write=0.)
        current = None
        with torch.inference_mode(), (a.output/'cache_manifest.jsonl').open('x') as stream:
            for index, r in enumerate(records):
                for p, digest in sources.items():
                    require(sha(p) == digest, 'frozen extension source')
                if current != r['wsi_id']:
                    slide = pyvips.Image.new_from_file(r['source_wsi'], access='random')
                    current = r['wsi_id']
                    require([slide.width, slide.height] == r['native_wh'], 'actual source header')
                folder = a.output/r['key'].replace(':', '_')
                folder.mkdir()
                raw = {}
                cells = []
                rgb = []
                views = iter(native_views(slide, r))
                for scale in range(3):
                    tick = time.perf_counter()
                    k, native, image, details = next(views)
                    require(k == scale, 'ordered native3 views')
                    timing['native_read'] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    arrays, d = prepare_variant(native, r['native_mpp'], details['metadata'], detector, ctp, xcell, device, cropper=cropper)
                    torch.cuda.synchronize()
                    timing['cell_prepare'] += time.perf_counter()-tick
                    require(d['detection_size'] == details['detection_size'] and d['detection_mpp_xy'] == details['detection_mpp_xy'], 'unchanged canonical detector bridge')
                    tick = time.perf_counter()
                    path = folder/f'cells_scale{scale}.npz'
                    np.savez_compressed(path, **arrays)
                    timing['write'] += time.perf_counter()-tick
                    if a.gate:
                        with np.load(path, allow_pickle=False) as z:
                            require(all(np.array_equal(z[key], value) for key, value in arrays.items()), 'all actual NPZ arrays survive disk')
                    append_view(raw, scale, image, arrays, device)
                    total = int(arrays['total'])
                    total_cells += total
                    retained += len(arrays['features'])
                    if total == 0:
                        zero.append(dict(key=r['key'], scale=scale))
                    cells.append(dict(scale=scale, path=str(path), sha256=sha(path), details=d,
                                      native_RGB_sha256=hashlib.sha256(native.tobytes()).hexdigest()))
                    rgb.append(hashlib.sha256(image.tobytes()).hexdigest())
                    dump('status.json', dict(status='preparing_new_full_valid_actual_cells', worker_pid=pid,
                         completed_inputs=index, total_inputs=len(records), key=r['key'], scale=scale, true_cells=total))
                tick = time.perf_counter()
                encoded = model.core.encode_image(raw)
                torch.cuda.synchronize()
                timing['encode'] += time.perf_counter()-tick
                finite(encoded)
                identity = dict(wsi_id=r['wsi_id'], kind='full_valid_query', key=r['key'], geometry=dict(tile=r['tile'],
                    native_boxes=r['native_boxes'], native_mpp=r['native_mpp'], source_wsi=r['source_wsi']), RGB_array_sha256=rgb, cell_records=cells)
                path = folder/'encoded.pth'
                torch.save(dict(encoded=transfer(encoded, 'cpu'), identity=identity, method='fewclick_m3d_xcell', checkpoint_sha256=best['checkpoint_sha256']), path)
                if a.gate:
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    tensor_checks += len(equal_encoded(encoded, saved['encoded']))
                if index == 0:
                    estimate = (path.stat().st_size+sum(Path(c['path']).stat().st_size for c in cells))*len(records)
                    require(shutil.disk_usage(a.output).free > 1.5*estimate, 'new fullcell cache actual disk headroom')
                encoded_bytes += path.stat().st_size
                stream.write(json.dumps(dict(index=r['index'], path=str(path), sha256=sha(path), bytes=path.stat().st_size, identity=identity))+'\n')
                stream.flush()
                dump('status.json', dict(status='encoded_new_full_valid_actual_cells', worker_pid=pid, completed_inputs=index+1, total_inputs=len(records), key=r['key']))
                del raw, encoded, arrays
                gc.collect()
        result = dict(status='passed_fullcell_valid_extension_gate' if a.gate else 'complete_fullcell_valid_extension_unaccepted',
            worker_pid=pid, signature=signature, inputs=len(records), views=3*len(records),
            input_manifest_sha256=sha(a.output/'inputs.jsonl'), cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),
            zero_cell_views=zero, sum_true_instances_across_overlapping_views=total_cells, sum_retained_across_views=retained,
            encoded_bytes=encoded_bytes, tensor_disk_checks=tensor_checks, timing_seconds=timing,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.perf_counter()-started,
            scope='New real native cells/CTP/XCell plus original encoder, no query GT; no detector subset invariance claim.',
            remaining='Independent geometry/count/74/byte/fresh encoder acceptance; full7996 after main20456 accepted, merged28452 cache, full-valid predictions/quality.')
        dump('summary.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'signature'}), flush=True)
    except Exception:
        dump('status.json', dict(status='failed', worker_pid=pid, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', action='store_true')
    p.add_argument('--verified-gate', type=Path)
    p.add_argument('--gate-acceptance', type=Path)
    run(p.parse_args())
