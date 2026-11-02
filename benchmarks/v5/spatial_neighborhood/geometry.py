"""Deterministic kNN on retained cell centers in physical micrometers."""
import numpy as np


def physical_knn(xy_um,instance_ids,k=8):
    xy=np.asarray(xy_um,np.float64);ids=np.asarray(instance_ids)
    assert xy.ndim==2 and xy.shape[1]==2 and ids.shape==(len(xy),)
    assert np.isfinite(xy).all() and len(np.unique(ids))==len(ids) and k==8
    n=len(xy);count=min(k,max(n-1,0))
    if not count:return np.empty((n,0),np.int64),np.empty((n,0),np.float64)
    distances=np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(-1));np.fill_diagonal(distances,np.inf)
    indices=np.lexsort((np.broadcast_to(ids,(n,n)),distances),axis=1)[:,:count]
    values=np.take_along_axis(distances,indices,axis=1)
    assert np.isfinite(values).all() and not (indices==np.arange(n)[:,None]).any()
    return indices,values
