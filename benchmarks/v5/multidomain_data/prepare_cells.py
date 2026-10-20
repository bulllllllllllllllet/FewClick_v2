"""Quality-matched image-only nuclei and XCell inputs, without tissue GT inputs."""
import argparse
from collections import Counter, defaultdict, deque
import json
import os
from pathlib import Path
import time

import pyvips  # Must precede torch/torchvision.
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F

from .prepare import dump, read_rows, sha
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_cell_models, cells_from_instances, attended_crop, WEIGHTS
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features


def filter_padding(inst, metadata):
    """Remove nuclei centered outside actual RGB support; no annotation access."""
    ids=np.unique(inst);ids=ids[ids>0]
    if not len(ids):return inst
    centers=np.asarray(ndimage.center_of_mass(np.ones_like(inst,np.uint8),inst,ids))
    affine=metadata['input_to_image_affine'];w,h=metadata['source_image_size']
    x=(centers[:,1]+.5)/inst.shape[1]*512*affine[0]+affine[2]
    y=(centers[:,0]+.5)/inst.shape[0]*512*affine[4]+affine[5]
    bad=ids[(x<0)|(x>=w)|(y<0)|(y>=h)]
    if len(bad):
        inst=inst.copy();inst[np.isin(inst,bad)]=0
    return inst


@torch.inference_mode()
def prepare_variant(image, mpp, metadata, detector, ctp, xcell, device):
    size=tuple(max(1,round(image.shape[1-i]*mpp[i]/.25)) for i in range(2))
    high=np.asarray(Image.fromarray(image).resize(size,Image.Resampling.BILINEAR)).copy()
    actual_mpp=[image.shape[1-i]*mpp[i]/size[i] for i in range(2)]
    inst=np.asarray(detector.eval(high,diameter=18,batch_size=1)[0],np.int32)
    inst=filter_padding(inst,metadata)
    meta,ids,total=cells_from_instances(inst)
    if len(ids):
        area=np.bincount(inst.ravel())[ids].astype(np.float64)
        meta[:,2]=np.log1p(area*actual_mpp[0]*actual_mpp[1]/(.25*.25))
    tensors=[]
    for label in ids:
        crop=attended_crop(high,inst,int(label))
        x=torch.from_numpy(crop.copy()).permute(2,0,1).float()[None]/255
        x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0]
        tensors.append((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
    reg=np.empty((0,64),np.float32)
    if tensors:
        raw=torch.cat([ctp(torch.stack(tensors[i:i+32]).to(device)).float() for i in range(0,len(tensors),32)])
        padded=raw.new_zeros((1,255,768));valid=raw.new_zeros((1,255))
        padded[0,:len(raw)]=raw;valid[0,:len(raw)]=1
        _,r,_,_=xcell(None,padded,valid);reg=r[0,:len(raw)].cpu().numpy()
    features=encode_xcell_features(meta,reg)
    if features.shape!=(len(ids),74) or not np.isfinite(features).all():
        raise ValueError('invalid prepared cell features')
    # All domains use explicit unknown nuclear identity, never tissue-GT labels.
    if len(ids) and not np.all(features[:,3]==1):
        raise ValueError('unknown nuclear identity encoding mismatch')
    return dict(instances=inst,meta=meta,reg=reg,features=features,total=np.asarray(total)),dict(
        detection_size=list(size),detection_mpp_xy=actual_mpp,nuclear_class='0 unknown',
        area_unit='log1p area in 0.25MPP equivalent pixels',cell_feature_dim=74,
        total_cell_count=total,retained_cell_count=len(ids),geometric_padding_filter=True)


def balanced_selection(rows,limit):
    groups=defaultdict(deque)
    for row in sorted(rows,key=lambda r:(r['selection_hash'],r['quality'])):
        groups[(row['dataset'],row['organ'],row['quality'])].append(row)
    selected=[]
    while groups and len(selected)<limit:
        for key in sorted(list(groups)):
            selected.append(groups[key].popleft())
            if not groups[key]:del groups[key]
            if len(selected)>=limit:break
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=100);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.limit<1:raise ValueError('positive limit required')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'cells').mkdir()
    rows=read_rows(a.inputs);selected=balanced_selection(rows,a.limit)
    if a.smoke:
        selected=[]
        for dataset in sorted({r['dataset'] for r in rows}):
            first=min((r for r in rows if r['dataset']==dataset),key=lambda r:r['selection_hash'])
            selected.extend(r for r in rows if r['tile_id']==first['tile_id'] and r['quality'] in ('original','0.8'))
    dump(a.output/'selection.json',selected)
    device=torch.device('cuda:0');torch.set_num_threads(a.threads)
    start=time.time();results=[]
    try:
        detector,ctp,xcell,audit=load_cell_models(device)
        dump(a.output/'run.json',dict(inputs=str(a.inputs),input_manifest_sha256=sha(a.inputs.read_bytes()),
            worker_pid=os.getpid(),device=str(device),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            weights={k:dict(path=str(v),sha256=sha(v.read_bytes())) for k,v in WEIGHTS.items()},
            model_compatibility=audit,source_sha256=sha(Path(__file__).read_bytes()),tissue_GT_used=False))
        with (a.output/'progress.jsonl').open('w') as log:
            for i,row in enumerate(selected):
                tick=time.time()
                if sha(Path(row['image']).read_bytes())!=row['image_sha256']:raise ValueError('RGB cache identity changed')
                image=np.asarray(Image.open(row['image']).convert('RGB')).copy()
                arrays,details=prepare_variant(image,json.loads(row['input_mpp_xy_json']),json.loads(row['preparation_json']),detector,ctp,xcell,device)
                path=a.output/'cells'/f'{row["tile_id"]}_{row["quality"]}.npz'
                np.savez_compressed(path,**arrays)
                with np.load(path) as saved:
                    for key,value in arrays.items():
                        if not np.array_equal(saved[key],value):raise ValueError('cell cache roundtrip mismatch')
                result=dict(row,cell_features_status='complete',cells=str(path),cells_sha256=sha(path.read_bytes()),
                    cell_preparation_json=json.dumps(details),cell_preparation_seconds=time.time()-tick)
                results.append(result);log.write(json.dumps({k:result[k] for k in ['tile_id','dataset','organ','quality','cells','cell_preparation_seconds']})+'\n');log.flush()
                dump(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=i+1,total=len(selected),elapsed_seconds=time.time()-start))
                print('CELLS',i+1,'/',len(selected),row['dataset'],row['quality'],'nuclei',details['total_cell_count'],'seconds',round(time.time()-tick,2),flush=True)
        pd.DataFrame(results).to_parquet(a.output/'inputs_with_cells.parquet',index=False)
        durations=[r['cell_preparation_seconds'] for r in results]
        dump(a.output/'summary.json',dict(status='complete',worker_pid=os.getpid(),inputs=len(results),
            dataset_counts=dict(Counter(r['dataset'] for r in results)),seconds=time.time()-start,
            median_seconds_per_variant=float(np.median(durations)),mean_seconds_per_variant=float(np.mean(durations)),
            tissue_GT_used=False,all_arrays_roundtrip_verified=True))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(results),elapsed_seconds=time.time()-start))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),completed=len(results),error=repr(exc)));raise


if __name__=='__main__':main()
