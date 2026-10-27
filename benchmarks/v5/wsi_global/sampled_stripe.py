"""Exact native-probability stitching restricted to frozen GT-cell centres.

Geometry uses GT dimensions only, never GT labels. Each 512 probability tile is
first resized by the original native PIL rule. The same row-major float32 sums
and threshold are then evaluated at a predetermined subset of native pixels.
"""
import hashlib
import mmap
from pathlib import Path

import numpy as np
from PIL import Image


class SampledStripe:
    def __init__(self, folder, bounds, fine_wh, native_wh, GT_wh):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False,parents=True)
        self.left,self.top,self.right,self.bottom=map(int,bounds)
        self.fw,self.fh=map(int,fine_wh)
        w,h=map(int,native_wh);gw,gh=map(int,GT_wh)
        if min(w,h,gw,gh,self.fw,self.fh)<=0 or not (0<=self.left<self.right<=w and 0<=self.top<self.bottom<=h):
            raise ValueError('invalid whole-cell native scoring canvas')
        sx,sy=w/gw,h/gh
        # Explicit physical predicates avoid depending on a label array/crop.
        xs=np.flatnonzero((np.arange(gw)*sx>=self.left)&((np.arange(gw)+1)*sx<=self.right))
        ys=np.flatnonzero((np.arange(gh)*sy>=self.top)&((np.arange(gh)+1)*sy<=self.bottom))
        if not len(xs) or not len(ys):
            raise ValueError('no original GT cells fully inside prediction canvas')
        self.GT_bounds=[int(xs[0]),int(ys[0]),int(xs[-1])+1,int(ys[-1])+1]
        self.px=np.floor((xs+.5)*sx).astype(np.int64)
        self.py=np.floor((ys+.5)*sy).astype(np.int64)
        self.capacity=int(np.ceil(self.fh/sy))+2
        self.total=np.zeros((self.capacity,len(xs)),np.float32)
        self.weight=np.zeros_like(self.total,dtype=np.uint16)
        self.prob=np.lib.format.open_memmap(self.folder/'probability.npy',mode='w+',dtype=np.float32,shape=(len(ys),len(xs)))
        self.mask=np.lib.format.open_memmap(self.folder/'mask.npy',mode='w+',dtype=np.bool_,shape=(len(ys),len(xs)))
        self.count=np.lib.format.open_memmap(self.folder/'overlap.npy',mode='w+',dtype=np.uint16,shape=(len(ys),len(xs)))
        self.written=0;self.native_y=self.top;self.tiles=0;self.weighted=0;self.maximum=0;self.closed=False;self.blocks=[]

    def flush_before(self, native_y):
        stop=int(np.searchsorted(self.py,native_y,side='left'))
        rows=stop-self.written
        if rows<0 or rows>self.capacity:
            raise ValueError('sampled stripe exceeds one native tile row')
        if rows:
            count=self.weight[:rows]
            if not np.all(count>0):
                raise ValueError('uncovered original GT-centre prediction pixels')
            value=self.total[:rows]/count
            start=self.written
            self.prob[start:stop]=value;self.mask[start:stop]=value>=.5;self.count[start:stop]=count
            for array in (self.prob,self.mask,self.count):
                byte_start=array.offset % mmap.ALLOCATIONGRANULARITY + start*len(self.px)*array.dtype.itemsize
                byte_end=array.offset % mmap.ALLOCATIONGRANULARITY + stop*len(self.px)*array.dtype.itemsize
                aligned=byte_start//mmap.PAGESIZE*mmap.PAGESIZE
                array._mmap.flush(aligned,byte_end-aligned)
                array._mmap.madvise(mmap.MADV_DONTNEED,aligned,byte_end-aligned)
            self.blocks.append(dict(start_GT_y=self.GT_bounds[1]+start,rows=rows,probability_sha256=hashlib.sha256(value.tobytes()).hexdigest(),mask_sha256=hashlib.sha256((value>=.5).tobytes()).hexdigest()))
            self.maximum=max(self.maximum,int(count.max()))
            remain=self.capacity-rows
            if remain:
                self.total[:remain]=self.total[rows:].copy();self.weight[:remain]=self.weight[rows:].copy()
            self.total[remain:]=0;self.weight[remain:]=0;self.written=stop
        self.native_y=native_y

    def push(self, probability, box):
        if self.closed:
            raise ValueError('sampled writer already finalized')
        x,y,w,h=map(int,box)
        if (w,h)!=(self.fw,self.fh) or x<self.left or x+w>self.right or y<self.native_y or y+h>self.bottom:
            raise ValueError('invalid row-major native tile geometry')
        if y>self.native_y:
            if y-self.native_y>self.fh:
                raise ValueError('native tile rows leave an uncovered gap')
            self.flush_before(y)
        value=np.asarray(probability)
        if value.ndim!=2 or not np.isfinite(value).all() or value.min()<0 or value.max()>1:
            raise ValueError('invalid per-tile probability')
        native=np.asarray(Image.fromarray(value.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
        columns=np.flatnonzero((self.px>=x)&(self.px<x+w))
        rows=np.flatnonzero((self.py>=y)&(self.py<y+h))
        local_rows=rows-self.written
        if len(rows) and (local_rows.min()<0 or local_rows.max()>=self.capacity):
            raise ValueError('sampled rows outside current bounded buffer')
        if len(rows) and len(columns):
            select=np.ix_(local_rows,columns)
            if np.any(self.weight[select]==np.iinfo(np.uint16).max):
                raise ValueError('overlap counter overflow')
            self.total[select]+=native[np.ix_(self.py[rows]-y,self.px[columns]-x)]
            self.weight[select]+=1
            self.weighted+=len(rows)*len(columns)
        self.tiles+=1

    def finish(self):
        if self.closed:
            raise ValueError('sampled writer already finalized')
        self.flush_before(self.bottom)
        if self.written!=len(self.py):
            raise ValueError('sampled scoring domain not fully finalized')
        for array in (self.prob,self.mask,self.count):
            array.flush()
        if int(self.count.sum(dtype=np.uint64))!=self.weighted:
            raise ValueError('sampled overlap weighted-pixel conservation')
        self.closed=True
        return dict(tiles=self.tiles,GT_cell_bounds=self.GT_bounds,GT_cells=len(self.py)*len(self.px),weighted_GT_cells=self.weighted,maximum_overlap=self.maximum,
                    buffer_bytes=self.total.nbytes+self.weight.nbytes,blocks=self.blocks,scope='Exact projection of native PIL/float32 mean/threshold at predetermined complete GT-cell centres; no GT labels or added resolution')
