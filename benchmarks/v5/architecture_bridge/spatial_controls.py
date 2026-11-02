"""P3 six-arm controls: keep true sampling geometry separate from MLP inputs."""
import hashlib
import numpy as np
import torch
from .repaired_model import RepairedData, RepairedContinuous
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention

ARMS=('I','S00','S10','S01','S11','H11')


class SpatialData(RepairedData):
    def __init__(self,cache,arm,seed=20261103):
        super().__init__(cache);assert arm in ARMS;self.arm=arm;self.seed=seed

    def inputs(self,identity,cell_arm):
        original=super().inputs(identity,cell_arm)
        if self.arm!='H11':return original
        result=dict(original)
        for scale in range(3):
            suffix='' if scale==0 else '_'+str(scale);key='cells'+suffix;x=original[key];n=x.shape[1]
            seed=int(hashlib.sha256(f'P3H11v1|{identity}|{scale}|{self.seed}'.encode()).hexdigest()[:16],16)
            order=np.random.default_rng(seed).permutation(n)
            out=x.clone();out[...,10:]=x[:,order,10:];result[key]=out
            assert torch.equal(out[...,:10],x[...,:10])
        return result


class SpatialCell(CellToRegionAttention):
    def __init__(self,original,arm):
        super().__init__(256,74,128);self.load_state_dict(original.state_dict());self.arm=arm

    def forward(self,region_tokens,assignment_at_cells,cells,cell_valid,total_cell_count=None):
        if self.arm=='I':
            context=torch.zeros_like(region_tokens);counts=region_tokens.new_zeros(region_tokens.shape[:2])
            return dict(fused_tokens=self.fuse(torch.cat([region_tokens,context,counts[...,None]],-1)),
                        cell_context=context,region_cell_count=counts)
        mass=assignment_at_cells
        if self.arm in ['S00','S10']:mass=torch.ones_like(mass)/region_tokens.shape[1]
        value=cells
        if self.arm in ['S00','S01']:
            value=cells.clone();value[...,:2]=.5
        return super().forward(region_tokens,mass,value,cell_valid,total_cell_count)


class SpatialModel(RepairedContinuous):
    def __init__(self,arm,seed=20261103,device='cuda:0'):
        assert arm in ARMS;super().__init__(seed,device);self.arm=arm
        self.core.cell=SpatialCell(self.core.cell,arm).to(device)
        self.identity.update(spatial_arm=arm)

    def transform(self,inputs):
        if self.arm!='I':return super().transform(inputs)
        result=dict(inputs)
        for suffix in ['', '_1','_2']:
            x=inputs['cells'+suffix]
            result['cells'+suffix]=x.new_zeros(x.shape[0],0,74)
            result['cell_valid'+suffix]=inputs['cell_valid'+suffix].new_zeros(x.shape[0],0)
            count='total_cell_count' if not suffix else 'total'+suffix
            result[count]=torch.zeros_like(inputs[count])
        return result
