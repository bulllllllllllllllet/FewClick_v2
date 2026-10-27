"""Independent interval-band replay of the original float32 binary WSI contract."""
from collections import defaultdict
import mmap

import numpy as np
from PIL import Image


def require(test,message):
    if not test:raise ValueError(message)


def counts(starts,length,left,right):
    change=np.zeros(right-left+1,np.int64)
    for value in starts:change[value-left]+=1;change[value-left+length]-=1
    return np.cumsum(change)[:-1]


def replay(indices,boxes,decode,outputs,full_bounds,atol=0.,progress=None):
    by_y=defaultdict(list)
    for i in indices:by_y[boxes[i][1]].append(i)
    fw,fh=boxes[indices[0]][2:];xs=sorted({boxes[i][0] for i in indices});ys=sorted(by_y)
    require(len(indices)==len(xs)*len(ys),'complete Cartesian audit window')
    left,top,right,bottom=min(xs),min(ys),max(xs)+fw,max(ys)+fh;full_left,full_top,_,_=full_bounds
    cx=counts(xs,fw,left,right);cy=counts(ys,fh,top,bottom);edges=sorted(set(ys)|{y+fh for y in ys})
    active={};pixels=weighted=decoded=round64=0;maximum_overlap=0;max_error=max64=0.
    for number,(y0,y1) in enumerate(zip(edges,edges[1:])):
        required_y=[y for y in ys if y<y1 and y+fh>y0]
        for y in list(active):
            if y not in required_y:del active[y]
        for y in required_y:
            if y in active:continue
            row=[]
            for i in by_y[y]:
                p=decode(i);require(p.shape==(512,512) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1,'actual512 probability shape/range')
                native=np.asarray(Image.fromarray(p.astype(np.float32)).resize((fw,fh),Image.Resampling.BILINEAR));row.append((boxes[i][0],native));decoded+=1
            active[y]=row
        total32=np.zeros((y1-y0,right-left),np.float32);total64=np.zeros_like(total32,dtype=np.float64)
        for y in sorted(active):
            for x,p in active[y]:
                total32[:,x-left:x-left+fw]+=p[y0-y:y1-y];total64[:,x-left:x-left+fw]+=p[y0-y:y1-y]
        count=cy[y0-top:y1-top,None]*cx[None,:];require((count>0).all() and count.max()<=65535,'independent overlap domain');count=count.astype(np.uint16)
        expected32=total32/count;ideal64=total64/count
        region=(slice(y0-full_top,y1-full_top),slice(left-full_left,right-full_left));actual=outputs['probability'][region];mask=outputs['mask'][region]
        require(np.isfinite(actual).all() and actual.min()>=0 and actual.max()<=1,'saved native probability range')
        error=float(np.abs(actual-expected32).max());require(error<=atol,'independent original float32 probability sum/mean')
        require(np.array_equal(outputs['overlap'][region],count),'every independent native overlap count')
        require(np.array_equal(mask,actual>=.5) and np.array_equal(mask,expected32>=.5),'every actual threshold/mask pixel')
        max_error=max(max_error,error);max64=max(max64,float(np.abs(actual.astype(np.float64)-ideal64).max()));round64+=int(np.count_nonzero(mask!=(ideal64>=.5)))
        pixels+=actual.size;weighted+=int(count.sum(dtype=np.uint64));maximum_overlap=max(maximum_overlap,int(count.max()))
        for array in outputs.values():array._mmap.madvise(mmap.MADV_DONTNEED)
        if progress:progress(dict(band=number+1,total_bands=len(edges)-1,pixels=pixels,decoded_tiles=decoded))
    require(decoded==len(indices) and pixels==(right-left)*(bottom-top) and weighted==len(indices)*fw*fh,'independent tile/pixel/weight conservation')
    return dict(tiles=decoded,pixels=pixels,weighted_pixels=weighted,maximum_overlap=maximum_overlap,maximum_probability_error=max_error,
        float64_diagnostic_maximum_delta=max64,float64_ideal_threshold_disagreements=round64,original_float32_threshold_exact=True,bounds=[left,top,right,bottom])
