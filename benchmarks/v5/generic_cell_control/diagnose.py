"""Inspect failed parity entries without resuming or modifying formal shards."""
import argparse
import json
from pathlib import Path

import pyvips
import numpy as np
from PIL import Image
import torch

from .gate import Capture, ExistingInstances
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector, CachedCenterCropper, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models


def run(shards, output, with_prefix=False):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0')
    encoder,xcell,_=load_feature_models(device);encoder=Capture(encoder).eval()
    detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,device);cropper=CachedCenterCropper();results=[]
    with torch.inference_mode():
        for shard in shards:
            state=json.loads((shard/'status.json').read_text())
            if state['status'] != 'failed':raise ValueError('diagnostic only accepts terminated failed shards')
            rows=json.loads((shard/'selection.json').read_text())
            done={(r['tile_id'],r['scale']) for r in map(json.loads,(shard/'rows.jsonl').read_text().splitlines())}
            row,scale=next((r,s) for r in rows for s in range(3) if (r['tile_id'],s) not in done)
            if with_prefix:
                # Reproduce call/shape history before the failure, not just a fresh-item call.
                for prior in rows:
                    for prior_scale in range(3):
                        if (prior['tile_id'],prior_scale)==(row['tile_id'],scale):break
                        if (prior['tile_id'],prior_scale) not in done:raise ValueError('non-prefix completed records')
                        pv=json.loads(prior['variants_json'])['original'];pg=json.loads(prior['multiscale_json'])
                        if prior_scale==0:
                            prgb=np.asarray(Image.open(pv['image']).convert('RGB')).copy();pmpp=json.loads(prior['input_mpp_xy_json'])
                            with np.load(pv['cells'],allow_pickle=False) as pc:pd=ExistingInstances(pc['instances'].copy())
                        else:prgb=native_context(prior,prior_scale);pmpp=pg['native_mpp'];pd=detector
                        ph,pw=prgb.shape[:2];pmeta=dict(input_to_image_affine=[pw/512,0,0,0,ph/512,0],source_image_size=[pw,ph])
                        encoder.parts.clear();prepare_variant(prgb,pmpp,pmeta,pd,encoder,xcell,device,cropper=cropper)
                    if prior['tile_id']==row['tile_id']:break
            variant=json.loads(row['variants_json'])['original'];geometry=json.loads(row['multiscale_json'])
            if scale==0:
                rgb=np.asarray(Image.open(variant['image']).convert('RGB')).copy();mpp=json.loads(row['input_mpp_xy_json'])
                with np.load(variant['cells'],allow_pickle=False) as cache:
                    reference=cache['features'].copy();total=int(cache['total']);active=ExistingInstances(cache['instances'].copy())
            else:
                rgb=native_context(row,scale);mpp=geometry['native_mpp'];active=detector
                with np.load(row['multiscale_cache'],allow_pickle=False) as cache:
                    reference=cache[f'cells_{scale}'].copy();total=int(cache[f'total_{scale}'])
            h,w=rgb.shape[:2];metadata=dict(input_to_image_affine=[w/512,0,0,0,h/512,0],source_image_size=[w,h])
            encoder.parts.clear();arrays,details=prepare_variant(rgb,mpp,metadata,active,encoder,xcell,device,cropper=cropper)
            generated=arrays['features'];same=generated.shape==reference.shape
            difference=generated-reference if same else None
            path=output/f'{row["tile_id"]}_scale{scale}.npz'
            np.savez_compressed(path,generated74=generated,reference74=reference,instances=arrays['instances'],
                raw768=np.concatenate(encoder.parts,axis=0) if encoder.parts else np.empty((0,768),np.float32))
            record=dict(tile_id=row['tile_id'],scale=scale,with_prefix=with_prefix,failed_shard=str(shard),reference_cells=len(reference),
                generated_cells=len(generated),reference_total=total,generated_total=int(arrays['total']),
                geometry_max_error=float(np.abs(difference[:,:10]).max()) if same and len(reference) else None,
                appearance_max_error=float(np.abs(difference[:,10:]).max()) if same and len(reference) else None,
                geometry_different_entries=int(np.count_nonzero(difference[:,:10])) if same else None,
                per_geometry_column_max_error=np.abs(difference[:,:10]).max(0).tolist() if same and len(reference) else None,
                mismatched_first20=[dict(cell=int(i),column=int(j),reference=float(reference[i,j]),generated=float(generated[i,j]))
                    for i,j in np.argwhere(difference[:,:10]!=0)[:20]] if same else None,
                details=details,output=str(path),output_sha256=digest(path))
            results.append(record);write(output/'progress.json',results);print(json.dumps(record),flush=True)
    write(output/'summary.json',dict(status='complete_diagnosis',results=results,source_sha256=digest(__file__),
        scope='Diagnostic repeat of failed entries only; formal failures unchanged, no identity tolerance relaxed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--with-prefix',action='store_true')
    a=p.parse_args();run(a.shards,a.output,a.with_prefix)
