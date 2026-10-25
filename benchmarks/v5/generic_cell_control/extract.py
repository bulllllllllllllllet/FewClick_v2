"""Sharded raw H&E extraction with explicit parity and a native full-path gate."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .gate import Capture, ExistingInstances
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector, CachedCenterCropper, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS


def run(catalog, output, gate=False, shard=0, shards=1, verified_gate=None):
    require(shards > 0 and 0 <= shard < shards, 'invalid shard')
    require(not gate or (shard == 0 and shards == 1), 'native gate covers both domains/splits')
    output.mkdir(parents=True, exist_ok=False); started = time.time(); torch.set_num_threads(4)
    require(torch.cuda.is_available(), 'native CUDA extraction required')
    device = torch.device('cuda:0')
    rows = pd.read_parquet(catalog/'tiles.parquet').to_dict('records')
    require(len({r['tile_id'] for r in rows}) == len(rows) == 1737, 'fixed complete catalog required')
    ordered = sorted(rows, key=lambda r: r['tile_id'])
    selected = [next(r for r in ordered if r['dataset'] == d and r['split'] == s)
                for d in ('crc', 'tiger') for s in ('train', 'val')] if gate else ordered[shard::shards]
    require(bool(selected), 'empty shard')
    source_files = [Path(__file__), Path(inspect.getfile(Capture)), Path(inspect.getfile(prepare_variant)),
        Path(inspect.getfile(load_feature_models)), Path(inspect.getfile(InstanSegDetector)), Path(inspect.getfile(native_context))]
    recipe = dict(catalog_sha256=digest(catalog/'tiles.parquet'),
        sources={str(p):digest(p) for p in source_files},
        weights={k:dict(path=str(v),sha256=digest(v)) for k,v in WEIGHTS.items() if k != 'cpsam'},
        detector_sha256=digest(DEFAULT_INSTANSEG_CHECKPOINT),
        geometry_policy='same retained nucleus identities, first10 geometry exact, reference74 max error2e-5',
        raw_policy='public CTransPath768 captured before reference XCell; no tissue GT input',
        fine_instances='original stored mask', context_instances='same native RGB/MPP and InstanSeg redetection')
    if not gate:
        require(verified_gate is not None, 'same-source full-path native gate required')
        accepted = json.loads((verified_gate/'summary.json').read_text())
        previous = json.loads((verified_gate/'run.json').read_text())
        require(accepted['status'] == 'complete' and accepted['gate'] and accepted['scale_rows'] == 12 and
                accepted['fields'] == 4 and previous['recipe'] == recipe, 'native gate source/data/weights mismatch')
    for row in selected:
        row['labels'] = '/unavailable/tissue_GT.png'
        for key in ('labels_array_sha256', 'counts_json', 'eligible_json', 'interior_counts_json'): row.pop(key, None)
    write(output/'selection.json', selected)
    write(output/'run.json', dict(worker_pid=os.getpid(), device=str(device), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        recipe=recipe, gate=gate, shard=shard, shards=shards, fields=len(selected), catalog=str(catalog),
        selection_sha256=digest(output/'selection.json'), verified_gate=str(verified_gate) if verified_gate else None,
        original_XCell_training='internal data, not current CRC/TIGER; reference parity only',
        transform_fitted=False, tissue_GT_available=False))
    encoder, reference_xcell, compatibility = load_feature_models(device)
    encoder = Capture(encoder).eval(); detector = InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT, device)
    cropper = CachedCenterCropper(); completed = 0
    with (output/'rows.jsonl').open('x') as stream, torch.inference_mode():
        for row in selected:
            variant = json.loads(row['variants_json'])['original']; geometry = json.loads(row['multiscale_json'])
            require(digest(variant['cells']) == variant['cells_sha256'] and
                    digest(row['multiscale_cache']) == row['multiscale_sha256'], 'reference cell cache SHA')
            for scale in (0, 1, 2):
                tick = time.time()
                if scale == 0:
                    require(digest(variant['image']) == variant['image_sha256'], 'fine RGB SHA')
                    rgb = np.asarray(Image.open(variant['image']).convert('RGB')).copy(); mpp = json.loads(row['input_mpp_xy_json'])
                    with np.load(variant['cells'], allow_pickle=False) as cache:
                        reference = cache['features'].copy(); total = int(cache['total'])
                        active_detector = ExistingInstances(cache['instances'].copy())
                else:
                    rgb = native_context(row, scale); mpp = geometry['native_mpp']; active_detector = detector
                    with np.load(row['multiscale_cache'], allow_pickle=False) as cache:
                        reference = cache[f'cells_{scale}'].copy(); total = int(cache[f'total_{scale}'])
                h, w = rgb.shape[:2]; metadata = dict(input_to_image_affine=[w/512,0,0,0,h/512,0], source_image_size=[w,h])
                encoder.parts.clear()
                arrays, details = prepare_variant(rgb, mpp, metadata, active_detector, encoder,
                    reference_xcell, device, cropper=cropper)
                raw = np.concatenate(encoder.parts, axis=0) if encoder.parts else np.empty((0,768),np.float32)
                require(raw.shape == (len(reference),768) and np.isfinite(raw).all(), 'raw generic H&E features')
                require(int(arrays['total']) == total and arrays['features'].shape == reference.shape, 'nucleus count/order')
                require(np.array_equal(arrays['features'][:,:10], reference[:,:10]), 'nucleus geometry/type')
                error = float(np.abs(arrays['features']-reference).max()) if len(reference) else 0.
                require(error <= 2e-5, 'reference XCell parity differs; refusing cell substitution')
                path = output/f'{row["tile_id"]}_scale{scale}.npz'
                np.savez_compressed(path, raw768=raw, geometry10=reference[:,:10], total=np.asarray(total),
                    reference74=reference, instances=arrays['instances'])
                with np.load(path,allow_pickle=False) as saved:
                    require(np.array_equal(saved['raw768'],raw) and np.array_equal(saved['geometry10'],reference[:,:10]) and
                            int(saved['total']) == total, 'disk replay')
                record = dict(tile_id=row['tile_id'],dataset=row['dataset'],split=row['split'],scale=scale,
                    cells=len(raw),total=total,output=str(path),output_sha256=digest(path),
                    reference_max_error=error,seconds=time.time()-tick,details=details)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); completed += 1
                write(output/'status.json',dict(status='running',worker_pid=os.getpid(),scale_rows=completed,
                    total_scale_rows=len(selected)*3,fields_completed=completed//3,seconds=time.time()-started))
                print(json.dumps({k:record[k] for k in ('tile_id','dataset','split','scale','cells','total','reference_max_error','seconds')}),flush=True)
    require(completed == len(selected)*3, 'incomplete shard')
    write(output/'summary.json',dict(status='complete',gate=gate,worker_pid=os.getpid(),shard=shard,shards=shards,
        fields=len(selected),scale_rows=completed,rows_sha256=digest(output/'rows.jsonl'),
        selection_sha256=digest(output/'selection.json'),recipe=recipe,compatibility=compatibility,
        seconds=time.time()-started,tissue_GT_used=False,transform_fitted=False,controlled_training_complete=False))
    write(output/'status.json',dict(status='complete',gate=gate,worker_pid=os.getpid(),fields=len(selected),scale_rows=completed))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--gate',action='store_true')
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    p.add_argument('--verified-gate',type=Path); a=p.parse_args()
    if a.output.exists(): raise SystemExit('output already exists; preserve previous artifacts')
    try: run(a.catalog,a.output,a.gate,a.shard_index,a.shards,a.verified_gate)
    except Exception:
        if a.output.is_dir():
            failure=dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc())
            write(a.output/'failure.json',failure);write(a.output/'status.json',failure)
        raise
