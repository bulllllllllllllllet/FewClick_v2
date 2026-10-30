"""Real image-only legacy Cellpose/attended-CTransPath/XCell input gate.

Expert points are used only after extraction to quantify label correspondence.
No GT instance mask or tissue label is passed into the feature branch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pyvips  # import before torch as required by existing preparation path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_cell_models, attended_crop, cells_from_instances


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.time()
    contract=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_cell_probe_manifest_1030_182424/contract.json')
    manifest=[json.loads(s) for s in (contract.parent/'cells.jsonl').read_text().splitlines()]
    gates=json.loads(contract.read_text())['gates'][:4];ids=[g['cell_id'].split(':')[1] for g in gates]
    rows=[]
    status=dict(status='loading_models',worker_pid=os.getpid(),fields=ids,completed=0)
    def save(): (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();device=torch.device('cuda:0');detector,ctp,xcell,loading=load_cell_models(device)
    for identity in ids:
        source=next(x for x in manifest if x['field_id']==identity)
        assert hashlib.sha256(Path(source['image']).read_bytes()).hexdigest()==source['image_sha256']
        with Image.open(source['image']) as im: rgb=np.asarray(im.convert('RGB')).copy()
        mx,my=source['mpp_xy'];size=(round(rgb.shape[1]*mx/.25),round(rgb.shape[0]*my/.25))
        high=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)).copy()
        mpp=np.array([rgb.shape[1]*mx/size[0],rgb.shape[0]*my/size[1]])
        inst=np.asarray(detector.eval(high,diameter=18,batch_size=1)[0],np.int32)
        metadata,labels,total=cells_from_instances(inst);embeddings=[]
        with torch.inference_mode():
            for label in labels:
                patch=attended_crop(high,inst,int(label))
                value=torch.from_numpy(patch.copy()).permute(2,0,1).float()[None]/255
                value=F.interpolate(value,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0]
                embeddings.append((value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
            if not embeddings:raise ValueError('empty detector output in fixed gate; preserve failure')
            raw=torch.cat([ctp(torch.stack(embeddings[i:i+32]).to(device)).float() for i in range(0,len(embeddings),32)])
            padded=raw.new_zeros((1,255,768));valid=raw.new_zeros((1,255));padded[0,:len(raw)]=raw;valid[0,:len(raw)]=1
            _,features,_,_=xcell(None,padded,valid);features=features[0,:len(raw)].cpu().numpy()
        assert features.shape==(len(labels),64) and np.isfinite(features).all()
        # Inference is finished. Expert annotations enter only the evaluation stage.
        gt=np.loadtxt(source['annotation'],delimiter=',',ndmin=2)
        gt_um=(gt[:,:2]+.5)*np.array([mx,my])
        pred_um=(metadata[:,:2]*np.array(size)+.5)*mpp
        distance=np.linalg.norm(pred_um[:,None,:]-gt_um[None,:,:],axis=-1)
        cost=np.where(distance<=5,distance/5,1000)
        pi,gi=linear_sum_assignment(cost);keep=distance[pi,gi]<=5;pi,gi=pi[keep],gi[keep]
        np.savez_compressed(a.output/f'{identity}.npz',instances=inst,instance_ids=labels,
                            features=features,pred_um=pred_um,matched_pred_index=pi,matched_GT_index=gi)
        row=dict(field=identity,all_detected=total,retained=len(labels),GT_points=len(gt),matched_at5um=len(pi),
                 unmatched_GT=len(gt)-len(pi),feature_shape=list(features.shape),GT_used_for_inference=False,
                 physical_crop_um=32,recipe='legacy Cellpose-SAM + mask-attended CTransPath + XCell255 context; detector/version conditions disclosed')
        rows.append(row);status.update(status='running',completed=len(rows),rows=rows);save();print(json.dumps(row),flush=True)
    report=dict(status='complete_legacy_branch_and_point_correspondence_gate',worker_pid=os.getpid(),rows=rows,
        loading=loading,elapsed_seconds=time.time()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quality_claim=False,scope='four fixed train fields, not a fair encoder comparison or downstream tissue quality')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete');save()


if __name__=='__main__':main()
