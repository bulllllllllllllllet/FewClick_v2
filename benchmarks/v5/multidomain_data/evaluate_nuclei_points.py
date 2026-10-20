"""Class-agnostic nucleus detection on six fixed OCELOT train cell fields."""
import argparse
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from .fast_nuclei import InstanSegDetector
from .geometry import standardize,quality_version
from .prepare import dump,sha
from .prepare_cells import filter_padding
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_cell_models

DATA=Path('/nfs-medical3/zyh/OCELOT_1016_122128/extracted/ocelot2023_v1.0.1')


def match_points(pred,gt,radius):
    if not len(pred) or not len(gt):return dict(tp=0,fp=len(pred),fn=len(gt))
    distance=np.linalg.norm(pred[:,None,:]-gt[None,:,:],axis=2)
    cost=np.where(distance<=radius,distance/(radius+1e-9),max(len(pred),len(gt))+1)
    i,j=linear_sum_assignment(cost);tp=int((distance[i,j]<=radius).sum())
    return dict(tp=tp,fp=len(pred)-tp,fn=len(gt)-tp)


def score(counts):
    tp,fp,fn=(counts[k] for k in ['tp','fp','fn'])
    return dict(counts,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0')
    meta=json.loads((DATA/'metadata.json').read_text())['sample_pairs'];selected=[]
    for organ in sorted({r['organ'] for r in meta.values()}):
        identity,r=next((k,v) for k,v in sorted(meta.items()) if v['organ']==organ and v['subset']=='train')
        selected.append(dict(id=identity,metadata=r))
    dump(a.output/'selection.json',dict(cases=selected,selection='first train ID per organ, before prediction',radii_um=[3,5],qualities=['original','0.8','1.0']))
    started=time.perf_counter();rows=[]
    try:
        cellpose,ctp,xcell,_=load_cell_models(device);del ctp,xcell
        instanseg=InstanSegDetector(a.checkpoint,device)
        for case in selected:
            identity=case['id'];m=case['metadata'];mpp=[m['cell']['resized_mpp_x'],m['cell']['resized_mpp_y']]
            rgb=np.asarray(Image.open(DATA/f'images/train/cell/{identity}.jpg').convert('RGB')).copy()
            raw_gt=np.loadtxt(DATA/f'annotations/train/cell/{identity}.csv',delimiter=',',ndmin=2)
            if raw_gt.shape[1]!=3 or set(np.unique(raw_gt[:,2]))-{1,2}:raise ValueError('OCELOT cell-point labels')
            w,h=(round(256/v) for v in mpp);box=[(rgb.shape[1]-w)//2,(rgb.shape[0]-h)//2,w,h]
            image,_,geometry=standardize(rgb,np.ones(rgb.shape[:2],np.uint8),box,mpp)
            extent=np.array([w*mpp[0],h*mpp[1]])
            gt_um=(raw_gt[:,:2]+.5-np.array(box[:2]))*np.array(mpp)
            for quality in ['original','0.8','1.0']:
                value,_=quality_version(image,geometry['input_mpp_xy'],mpp,quality,geometry)
                folder=a.output/f'{identity}_{quality}';folder.mkdir();Image.fromarray(value).save(folder/'image.png')
                high=np.asarray(Image.fromarray(value).resize((1024,1024),Image.Resampling.BILINEAR)).copy()
                for name,model in [('cellpose_sam',cellpose),('instanseg',instanseg)]:
                    torch.cuda.synchronize();tick=time.perf_counter();instances=model.eval(high,diameter=18,batch_size=1)[0];torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    instances=filter_padding(instances,geometry)
                    ids=np.unique(instances);ids=ids[ids>0]
                    centers=np.asarray(ndimage.center_of_mass(np.ones_like(instances,np.uint8),instances,ids)).reshape(-1,2)
                    pred_um=(centers[:,::-1]+.5)/np.array(instances.shape[::-1])*extent
                    np.savez_compressed(folder/f'{name}.npz',instances=instances,gt_um=gt_um,pred_um=pred_um)
                    scores={str(radius):score(match_points(pred_um,gt_um,radius)) for radius in [3,5]}
                    row=dict(id=identity,organ=m['organ'],quality=quality,model=name,predicted=len(ids),GT=len(raw_gt),seconds=seconds,scores=scores)
                    rows.append(row);dump(folder/f'{name}.json',row)
                    print('POINTS',json.dumps(row),flush=True)
                dump(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(rows),total=36))
        summary={}
        for model in ['cellpose_sam','instanseg']:
            summary[model]={}
            for quality in ['original','0.8','1.0']:
                group=[r for r in rows if r['model']==model and r['quality']==quality]
                summary[model][quality]={str(radius):score({k:sum(r['scores'][str(radius)][k] for r in group) for k in ['tp','fp','fn']}) for radius in [3,5]}
        dump(a.output/'summary.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.perf_counter()-started,summary=summary,rows=rows,
            scope='6 fixed train fields; class-agnostic point matching, not instance-boundary PQ or FewClick tissue Dice',
            source_sha256=sha(Path(__file__).read_bytes()),radii_um=[3,5],test_used=False,puma_used=False))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(rows)))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
