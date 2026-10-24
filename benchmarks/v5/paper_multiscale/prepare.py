"""Prepare native-context RGB and image-only cells; persist every completed tile."""
import argparse
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .data import native_context
from benchmarks.v5.multidomain_data.prepare import dump, read_rows, sha
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector, CachedCenterCropper, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models,WEIGHTS


def main():
    p = argparse.ArgumentParser(); p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1); p.add_argument('--limit', type=int)
    p.add_argument('--exclude-prepared',type=Path,help='Verified existing gate tiles, reused without recomputation')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    rows = sorted(read_rows(a.catalog/'tiles.parquet'), key=lambda r: r['tile_id'])
    if a.exclude_prepared:
        previous=read_rows(a.exclude_prepared/'tiles.parquet')
        for row in previous:
            if sha(Path(row['multiscale_cache']).read_bytes())!=row['multiscale_sha256']:
                raise ValueError('gate cache changed before reuse')
        exclude={r['tile_id'] for r in previous};rows=[r for r in rows if r['tile_id'] not in exclude]
    if a.limit:
        # Balanced first examples, including both splits and every retained domain.
        selected = []
        for dataset in sorted({r['dataset'] for r in rows}):
            for split in ('train','val'):
                selected.extend([r for r in rows if r['dataset']==dataset and r['split']==split][:a.limit])
        rows = selected
    rows = [r for i,r in enumerate(rows) if i%a.shards==a.shard]
    if not rows: raise ValueError('empty preparation shard')
    started = time.time(); torch.set_num_threads(4); device = torch.device('cuda:0')
    detector = InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT, device)
    ctp, xcell, audit = load_feature_models(device); cropper = CachedCenterCropper()
    dump(a.output/'run.json', dict(worker_pid=os.getpid(), input_sha256=sha((a.catalog/'tiles.parquet').read_bytes()),
        audit=audit, shard=a.shard, shards=a.shards, native_context_cells=True, tissue_GT_used=False,
        source_sha256=sha(Path(__file__).read_bytes()),
        weight_sha256={k:sha(path.read_bytes()) for k,path in WEIGHTS.items() if k!='cpsam'}|
                      {'instanseg':sha(DEFAULT_INSTANSEG_CHECKPOINT.read_bytes())}))
    results = []
    try:
        with (a.output/'completed.jsonl').open('w') as log:
            for row in rows:
                arrays = {}; geometry = json.loads(row['multiscale_json']); details = []
                for scale in (1,2):
                    rgb = native_context(row, scale)
                    height, width = rgb.shape[:2]
                    meta = dict(input_to_image_affine=[width/512,0,0,0,height/512,0], source_image_size=[width,height])
                    cells, info = prepare_variant(rgb, geometry['native_mpp'], meta, detector, ctp, xcell, device, cropper=cropper)
                    arrays.update({f'image_{scale}':np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)),
                                   f'cells_{scale}':cells['features'], f'total_{scale}':cells['total']})
                    details.append(info)
                path = a.output/f'{row["tile_id"]}.npz'; np.savez(path, **arrays)
                with np.load(path,allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[k],v) for k,v in arrays.items()): raise ValueError('cache roundtrip')
                item = row | dict(multiscale_cache=str(path), multiscale_sha256=sha(path.read_bytes()))
                results.append(item); log.write(json.dumps(dict(tile_id=row['tile_id'],cache=str(path),details=details))+'\n'); log.flush()
                dump(a.output/'status.json',dict(status='preparing',worker_pid=os.getpid(),completed=len(results),total=len(rows),seconds=time.time()-started))
                print('CONTEXT',len(results),'/',len(rows),row['dataset'],row['split'],round(time.time()-started,2),flush=True)
        pd.DataFrame(results).to_parquet(a.output/'tiles.parquet', index=False)
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(results),seconds=time.time()-started))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),completed=len(results),error=repr(exc))); raise


if __name__ == '__main__': main()
