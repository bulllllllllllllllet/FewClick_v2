"""Row-major probability averaging with a one-tile-high memory buffer."""
from pathlib import Path
import time
import mmap

import numpy as np
from PIL import Image


class StripeWriter:
    def __init__(self,folder,bounds,fine_wh):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.left,self.top,self.right,self.bottom=map(int,bounds);self.fw,self.fh=map(int,fine_wh)
        self.width=self.right-self.left;self.height=self.bottom-self.top;self.y=self.top
        if min(self.width,self.height,self.fw,self.fh)<=0:raise ValueError('invalid canvas')
        self.total=np.zeros((self.fh,self.width),np.float32);self.weight=np.zeros((self.fh,self.width),np.uint16)
        shape=(self.height,self.width)
        self.prob=np.lib.format.open_memmap(self.folder/'probability.npy',mode='w+',dtype=np.float32,shape=shape)
        self.mask=np.lib.format.open_memmap(self.folder/'mask.npy',mode='w+',dtype=np.bool_,shape=shape)
        self.count=np.lib.format.open_memmap(self.folder/'overlap.npy',mode='w+',dtype=np.uint16,shape=shape)
        self.tiles=0;self.covered=0;self.weighted=0;self.maximum_overlap=0;self.flush_seconds=0.;self.closed=False

    def flush_rows(self,number):
        if not 0<number<=self.fh or self.y+number>self.bottom:raise ValueError('invalid finalized stripe')
        counts=self.weight[:number]
        if not (counts>0).all():raise ValueError('uncovered pixels in finalized prediction domain')
        tick=time.perf_counter();value=self.total[:number]/counts
        start=self.y-self.top;end=start+number
        self.prob[start:end]=value;self.mask[start:end]=value>=.5;self.count[start:end]=counts
        # Keep the mapped output pages from accumulating in this process's RSS.
        # Flush only finalized row ranges; shared boundary pages are persisted
        # before releasing the mapping and can be reloaded by the next stripe.
        for array in (self.prob,self.mask,self.count):
            byte_start=array.offset % mmap.ALLOCATIONGRANULARITY + start*self.width*array.dtype.itemsize
            byte_end=array.offset % mmap.ALLOCATIONGRANULARITY + end*self.width*array.dtype.itemsize
            aligned=byte_start//mmap.PAGESIZE*mmap.PAGESIZE
            array._mmap.flush(aligned,byte_end-aligned)
            array._mmap.madvise(mmap.MADV_DONTNEED,aligned,byte_end-aligned)
        self.covered+=number*self.width;self.weighted+=int(counts.sum(dtype=np.uint64));self.maximum_overlap=max(self.maximum_overlap,int(counts.max()))
        # Preserve only the part future overlapping rows can still affect.
        remain=self.fh-number
        if remain:
            self.total[:remain]=self.total[number:].copy();self.weight[:remain]=self.weight[number:].copy()
        self.total[remain:]=0;self.weight[remain:]=0;self.y+=number
        self.flush_seconds+=time.perf_counter()-tick

    def push(self,probability,box):
        if self.closed:raise ValueError('writer already finalized')
        x,y,w,h=map(int,box)
        if (w,h)!=(self.fw,self.fh) or x<self.left or x+w>self.right or y<self.y or y+h>self.bottom:
            raise ValueError('tile geometry/order outside declared domain')
        if y>self.y:
            if y-self.y>self.fh:raise ValueError('gap between tile rows')
            self.flush_rows(y-self.y)
        p=np.asarray(probability)
        if p.ndim!=2 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('invalid probability')
        native=np.asarray(Image.fromarray(p.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
        offset=x-self.left;counts=self.weight[:,offset:offset+w]
        if (counts==np.iinfo(np.uint16).max).any():raise ValueError('overlap count overflow')
        self.total[:,offset:offset+w]+=native;counts+=1;self.tiles+=1

    def finish(self):
        if self.closed:raise ValueError('duplicate finalization')
        self.flush_rows(self.fh)
        if self.y!=self.bottom or self.covered!=self.width*self.height or self.weighted!=self.tiles*self.fw*self.fh:
            raise ValueError('full coverage/weighted pixel conservation failed')
        tick=time.perf_counter()
        for array in (self.prob,self.mask,self.count):array.flush()
        self.flush_seconds+=time.perf_counter()-tick;self.closed=True
        return dict(tiles=self.tiles,covered_pixels=self.covered,weighted_pixels=self.weighted,maximum_overlap=self.maximum_overlap,
            bounds_level0=[self.left,self.top,self.right,self.bottom],sum_count_buffer_bytes=self.total.nbytes+self.weight.nbytes,
            flush_seconds=self.flush_seconds,output_resolution='native level0; continuous probability resized from512, no added spatial detail')
