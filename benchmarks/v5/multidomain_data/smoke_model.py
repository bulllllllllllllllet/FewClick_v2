"""Check standard inputs against real frozen checkpoints, without ranking models."""
import argparse
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
import torch

from .prepare import read_rows,dump,sha
from benchmarks.v5.phase_6_mask_decoder.boundary_data import load_baseline
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel,dense_loss
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import encode_image,encode_support,query_features
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();device=torch.device('cuda:0');torch.set_num_threads(4)
    try:
        run=json.loads((DENSE_RUN/'run.json').read_text());metadata=json.loads((Path(run['checkpoint']).parent/'run_metadata.json').read_text())
        j5=load_baseline(metadata,Path(run['checkpoint']),device)
        dense=DenseContextModel().to(device).eval().requires_grad_(False)
        checkpoint=torch.load(DENSE_RUN/'dense128_epoch_031.pth',map_location=device,weights_only=False)
        dense.load_state_dict(checkpoint['model'],strict=True)
        rows=[]
        for row in read_rows(a.inputs):
            if sha(Path(row['image']).read_bytes())!=row['image_sha256'] or sha(Path(row['cells']).read_bytes())!=row['cells_sha256']:
                raise ValueError('input cache identity changed')
            image=np.asarray(Image.open(row['image']).convert('RGB')).copy()
            with np.load(row['cells']) as cells:
                features=cells['features'].copy();total=int(cells['total'])
            inputs=dict(image=he_to_tensor(image)[None].to(device),cells=torch.from_numpy(features)[None].to(device),
                cell_valid=torch.ones(1,len(features),dtype=torch.bool,device=device),total_cell_count=torch.tensor([total],device=device,dtype=torch.float32))
            # PUMA is a held-out disease: tensor-only forward with fixed arbitrary
            # positions. Do not read its tissue labels or compute a target score.
            if row['dataset']=='puma':
                xy=[[.5,.5],[.1,.1]];truth=valid=None
            else:
                gt=np.asarray(Image.open(row['labels']));valid=torch.from_numpy(gt!=255)[None].to(device)
                candidates=[int(v) for v in np.unique(gt) if v!=255 and (gt==v).sum()>=32 and ((gt!=v)&(gt!=255)).sum()>=32]
                if not candidates:raise ValueError(f'no mixed target for declared smoke: {row["tile_id"]}')
                target=candidates[0];truth=torch.from_numpy(gt==target)[None].to(device)
                # One positive and one negative actual pixel center; GT simulation.
                xy=[]
                for mask in [gt==target,(gt!=target)&(gt!=255)]:
                    y,x=np.argwhere(mask)[len(np.argwhere(mask))//2];xy.append([(float(x)+.5)/512,(float(y)+.5)/512])
            prompts=dict(positive_xy=torch.tensor([[xy[0]]],device=device,dtype=torch.float32),negative_xy=torch.tensor([[xy[1]]],device=device,dtype=torch.float32),
                positive_mask=torch.ones(1,1,dtype=torch.bool,device=device),negative_mask=torch.ones(1,1,dtype=torch.bool,device=device),prompt_size_id=torch.tensor([1],device=device))
            with torch.inference_mode():
                encoded=encode_image(j5,inputs)
                task=encode_support(j5,encoded,prompts)
                f=query_features(j5,encoded,task,prompts,query_id=row['tile_id'],support_id=row['tile_id'])
                out=dense(f)
                remote=query_features(j5,encoded,task,prompts,query_id='other-tile',support_id=row['tile_id'])
                if remote['prompt_maps'].count_nonzero():raise ValueError('source heatmap copied to remote query')
                if out['pixel_probability'].shape!=(1,512,512) or not torch.isfinite(out['pixel_logits']).all():raise ValueError('model output contract')
                if truth is not None:
                    loss=dense_loss(out,dict(truth=truth,valid=valid))
                    if not torch.isfinite(loss):raise ValueError('non-finite real checkpoint loss')
                else:loss=None
            rows.append(dict(dataset=row['dataset'],quality=row['quality'],tile_id=row['tile_id'],cells=len(features),
                output_shape=list(out['pixel_probability'].shape),finite_loss=None if loss is None else float(loss),
                puma_GT_read=False if row['dataset']=='puma' else None,remote_heatmaps_zero=True))
        reloaded=DenseContextModel().to(device).eval().requires_grad_(False);reloaded.load_state_dict(checkpoint['model'],strict=True)
        with torch.inference_mode():
            if not torch.equal(out['pixel_probability'],reloaded(f)['pixel_probability']):raise ValueError('checkpoint reloading mismatch')
        dump(a.output/'summary.json',dict(status='passed',worker_pid=os.getpid(),seconds=time.time()-start,rows=rows,
            scope='input/model/loss compatibility only; no Dice/BF1, checkpoint selection or PUMA target evaluation',
            dense_checkpoint_reload_exact=True,source_sha256=sha(Path(__file__).read_bytes())))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-start))
        print('MODEL_SMOKE_PASSED',len(rows),round(time.time()-start,2),flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
