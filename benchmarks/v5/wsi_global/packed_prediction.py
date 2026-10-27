"""Lossless prediction masks and exact probability digests for full-budget WSI.

Inherits the accepted native-PIL/float32 overlap arithmetic. Only the output
sink changes: one bit per scored pixel, plus hashes of every finalized float32
probability and uint16 overlap block. Labels are never read during prediction.
"""
import hashlib
import mmap
from pathlib import Path

import numpy as np

from .sampled_stripe import SampledStripe


class PackedSampledStripe(SampledStripe):
    def __init__(self,folder,bounds,fine_wh,native_wh,GT_wh):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False,parents=True)
        self.left,self.top,self.right,self.bottom=map(int,bounds)
        self.fw,self.fh=map(int,fine_wh)
        w,h=map(int,native_wh);gw,gh=map(int,GT_wh)
        if min(w,h,gw,gh,self.fw,self.fh)<=0 or not (0<=self.left<self.right<=w and 0<=self.top<self.bottom<=h):
            raise ValueError('invalid native prediction domain')
        sx,sy=w/gw,h/gh
        xs=np.flatnonzero((np.arange(gw)*sx>=self.left)&((np.arange(gw)+1)*sx<=self.right))
        ys=np.flatnonzero((np.arange(gh)*sy>=self.top)&((np.arange(gh)+1)*sy<=self.bottom))
        if not len(xs) or not len(ys):
            raise ValueError('no complete scoring cells')
        self.GT_bounds=[int(xs[0]),int(ys[0]),int(xs[-1])+1,int(ys[-1])+1]
        self.px=np.floor((xs+.5)*sx).astype(np.int64)
        self.py=np.floor((ys+.5)*sy).astype(np.int64)
        self.capacity=int(np.ceil(self.fh/sy))+2
        self.total=np.zeros((self.capacity,len(xs)),np.float32)
        self.weight=np.zeros_like(self.total,dtype=np.uint16)
        self.row_bytes=(len(xs)+7)//8
        self.packed=np.lib.format.open_memmap(self.folder/'mask_bits.npy',mode='w+',dtype=np.uint8,shape=(len(ys),self.row_bytes))
        self.written=0;self.native_y=self.top;self.tiles=0;self.weighted=0
        self.maximum=0;self.finalized_weighted=0;self.closed=False;self.blocks=[]

    def flush_before(self,native_y):
        stop=int(np.searchsorted(self.py,native_y,side='left'));rows=stop-self.written
        if rows<0 or rows>self.capacity:
            raise ValueError('invalid packed prediction stripe')
        if rows:
            counts=self.weight[:rows]
            if not np.all(counts>0):
                raise ValueError('uncovered prediction cells')
            probability=self.total[:rows]/counts
            binary=probability>=.5
            bits=np.packbits(binary,axis=1,bitorder='big')
            if bits.shape!=(rows,self.row_bytes):
                raise ValueError('packed prediction dimensions')
            start=self.written;self.packed[start:stop]=bits
            self.blocks.append(dict(start_GT_y=self.GT_bounds[1]+start,rows=rows,
                probability_float32_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                overlap_uint16_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
                binary_bool_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
                packed_uint8_sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
            byte_start=self.packed.offset % mmap.ALLOCATIONGRANULARITY+start*self.row_bytes
            byte_end=self.packed.offset % mmap.ALLOCATIONGRANULARITY+stop*self.row_bytes
            aligned=byte_start//mmap.PAGESIZE*mmap.PAGESIZE
            self.packed._mmap.flush(aligned,byte_end-aligned)
            self.packed._mmap.madvise(mmap.MADV_DONTNEED,aligned,byte_end-aligned)
            self.maximum=max(self.maximum,int(counts.max()))
            self.finalized_weighted+=int(counts.sum(dtype=np.uint64))
            remain=self.capacity-rows
            if remain:
                self.total[:remain]=self.total[rows:].copy();self.weight[:remain]=self.weight[rows:].copy()
            self.total[remain:]=0;self.weight[remain:]=0;self.written=stop
        self.native_y=native_y

    def finish(self):
        if self.closed:
            raise ValueError('prediction already finalized')
        self.flush_before(self.bottom)
        if self.written!=len(self.py) or self.weighted!=self.finalized_weighted:
            raise ValueError('packed prediction/overlap coverage incomplete')
        self.packed.flush();self.closed=True
        return dict(tiles=self.tiles,GT_cell_bounds=self.GT_bounds,GT_shape=[len(self.py),len(self.px)],
                    GT_cells=len(self.py)*len(self.px),weighted_GT_cells=self.weighted,maximum_overlap=self.maximum,
                    packed_shape=[len(self.py),self.row_bytes],bitorder='big',padding_bits_per_row=self.row_bytes*8-len(self.px),
                    mask_path=str(self.folder/'mask_bits.npy'),blocks=self.blocks,
                    buffer_bytes=self.total.nbytes+self.weight.nbytes,
                    scope='Lossless thresholded native prediction at fixed GT-cell centres; every pre-threshold probability/overlap block hashed; no GT labels read or probabilities quantized')


def unpack_rows(path,rows,width):
    """Read original Boolean predictions; padding bits never enter metrics."""
    packed=np.load(path,mmap_mode='r')
    if packed.dtype!=np.uint8 or packed.ndim!=2 or packed.shape[1]!=(width+7)//8:
        raise ValueError('invalid packed prediction schema')
    values=np.asarray(packed[rows])
    return np.unpackbits(values,axis=-1,count=width,bitorder='big').astype(bool)
