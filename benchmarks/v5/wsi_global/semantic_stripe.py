"""Native multiclass probability averaging with bounded stripe memory."""
import hashlib
import mmap
from pathlib import Path

import numpy as np
from PIL import Image


class SemanticStripe:
    def __init__(self,folder,bounds,fine_wh,classes=12,chunk=128):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.left,self.top,self.right,self.bottom=map(int,bounds);self.fw,self.fh=map(int,fine_wh);self.classes=classes;self.chunk=chunk
        self.width=self.right-self.left;self.height=self.bottom-self.top;self.y=self.top
        if min(self.width,self.height,self.fw,self.fh,self.classes,self.chunk)<=0:raise ValueError('positive native canvas')
        self.total=np.zeros((self.fh,self.width,classes),np.float64);self.weight=np.zeros((self.fh,self.width),np.uint16)
        shape=(self.height,self.width);self.pred=np.lib.format.open_memmap(self.folder/'class_prediction.npy',mode='w+',dtype=np.uint8,shape=shape)
        self.count=np.lib.format.open_memmap(self.folder/'overlap.npy',mode='w+',dtype=np.uint16,shape=shape)
        self.selected={c:np.lib.format.open_memmap(self.folder/f'probability_class{c}.npy',mode='w+',dtype=np.float32,shape=shape) for c in (0,10)}
        self.blocks=[];self.tiles=0;self.pixels=0;self.weighted=0;self.maximum=0;self.normalization_error=0.;self.closed=False

    def flush(self,number):
        if not 0<number<=self.fh or self.y+number>self.bottom:raise ValueError('flush extent')
        for start in range(0,number,self.chunk):
            end=min(start+self.chunk,number);count=self.weight[start:end]
            if not (count>0).all():raise ValueError('uncovered native probability pixel')
            p=self.total[start:end]/count[:,:,None];err=float(np.max(np.abs(p.sum(2)-1)))
            if not np.isfinite(p).all() or err>1e-5:raise ValueError('invalid categorical probability sum')
            self.normalization_error=max(self.normalization_error,err);offset=self.y-self.top+start
            self.pred[offset:offset+end-start]=p.argmax(2).astype(np.uint8);self.count[offset:offset+end-start]=count
            for c,array in self.selected.items():array[offset:offset+end-start]=p[:,:,c].astype(np.float32)
            self.blocks.append(dict(start_native_y=self.y+start,rows=end-start,sha256_float64_HWC=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()))
            self.pixels+=(end-start)*self.width;self.weighted+=int(count.sum(dtype=np.uint64));self.maximum=max(self.maximum,int(count.max()))
        start=self.y-self.top;end=start+number
        for array in [self.pred,self.count,*self.selected.values()]:
            first=array.offset % mmap.ALLOCATIONGRANULARITY+start*self.width*array.dtype.itemsize
            last=array.offset % mmap.ALLOCATIONGRANULARITY+end*self.width*array.dtype.itemsize
            aligned=first//mmap.PAGESIZE*mmap.PAGESIZE;array._mmap.flush(aligned,last-aligned);array._mmap.madvise(mmap.MADV_DONTNEED,aligned,last-aligned)
        remain=self.fh-number
        if remain:self.total[:remain]=self.total[number:].copy();self.weight[:remain]=self.weight[number:].copy()
        self.total[remain:]=0;self.weight[remain:]=0;self.y+=number

    def push(self,probability,box):
        if self.closed:raise ValueError('already closed')
        x,y,w,h=map(int,box);p=np.asarray(probability)
        if (w,h)!=(self.fw,self.fh) or p.shape!=(self.classes,512,512) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('invalid native tile')
        if x<self.left or x+w>self.right or y<self.y or y+h>self.bottom:raise ValueError('outside canvas or row-major order')
        if y>self.y:
            if y-self.y>self.fh:raise ValueError('row gap')
            self.flush(y-self.y)
        offset=x-self.left
        if (self.weight[:,offset:offset+w]==65535).any():raise ValueError('overlap overflow')
        for c in range(self.classes):
            native=np.asarray(Image.fromarray(p[c].astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
            self.total[:,offset:offset+w,c]+=native
        self.weight[:,offset:offset+w]+=1;self.tiles+=1

    def finish(self):
        if self.closed:raise ValueError('duplicate finish')
        self.flush(self.fh)
        if self.y!=self.bottom or self.pixels!=self.width*self.height or self.weighted!=self.tiles*self.fw*self.fh:raise ValueError('coverage/count conservation')
        for array in [self.pred,self.count,*self.selected.values()]:array.flush()
        self.closed=True
        return dict(bounds_native=[self.left,self.top,self.right,self.bottom],tiles=self.tiles,pixels=self.pixels,weighted_pixels=self.weighted,maximum_overlap=self.maximum,probability_sum_error=self.normalization_error,blocks=self.blocks,buffer_bytes=self.total.nbytes+self.weight.nbytes)
