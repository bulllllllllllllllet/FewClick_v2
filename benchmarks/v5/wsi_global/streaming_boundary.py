"""Whole-domain physical BF1 using row halos, identical to the frozen metric.

Only distances <= tolerance matter. An extra boundary-neighbour row beyond
the physical distance radius prevents chunk seams from changing eligibility.
The scoring-domain frame and invalid pixels follow the existing metric.
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmarks.v5.fair_comparison.metrics import boundary


def physical_boundary_counts(read_rows,shape,mpp,tolerance_um=2.,block_rows=256):
    height,width=map(int,shape);mx,my=map(float,mpp)
    if min(height,width,block_rows)<=0 or not all(np.isfinite([mx,my,tolerance_um])) or min(mx,my)<=0 or tolerance_um<0:
        raise ValueError('invalid physical boundary domain')
    threshold=tolerance_um+1e-9
    halo=math.ceil(threshold/my)+1
    npred=ntrue=matched_pred=matched_true=0
    blocks=[]
    for start in range(0,height,block_rows):
        end=min(start+block_rows,height);lo=max(0,start-halo);hi=min(height,end+halo)
        prediction,truth,valid=read_rows(lo,hi)
        prediction=np.asarray(prediction,bool);truth=np.asarray(truth,bool);valid=np.asarray(valid,bool)
        if prediction.shape!=truth.shape or truth.shape!=valid.shape or truth.shape!=(hi-lo,width):
            raise ValueError('callback must supply aligned full-width rows including halo')
        pe=boundary(prediction,valid);te=boundary(truth,valid);core=slice(start-lo,end-lo)
        pc=pe[core];tc=te[core]
        pcount=int(pc.sum());tcount=int(tc.sum());pm=tm=0
        # EDT on an all-true array has implicit-frame behaviour. An empty
        # reference edge set must yield no matches anywhere in the core.
        if pcount and te.any():
            distance=distance_transform_edt(~te,sampling=(my,mx));pm=int(np.count_nonzero(distance[core][pc]<=threshold))
        if tcount and pe.any():
            distance=distance_transform_edt(~pe,sampling=(my,mx));tm=int(np.count_nonzero(distance[core][tc]<=threshold))
        npred+=pcount;ntrue+=tcount;matched_pred+=pm;matched_true+=tm
        blocks.append(dict(start_row=start,rows=end-start,predicted_boundary_pixels=pcount,true_boundary_pixels=tcount,matched_predicted=pm,matched_true=tm))
    precision=matched_pred/npred if npred else 0.
    recall=matched_true/ntrue if ntrue else None
    bf1=None if not ntrue else 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(predicted_boundary_pixels=npred,true_boundary_pixels=ntrue,boundary_evaluable=ntrue>0,bf1_um2=bf1,
                matched_predicted_boundary_pixels=matched_pred,matched_true_boundary_pixels=matched_true,
                boundary_precision=precision,boundary_recall=recall,tolerance_um=tolerance_um,
                GT_mpp_xy=[mx,my],block_rows=block_rows,halo_rows=halo,blocks=blocks)
