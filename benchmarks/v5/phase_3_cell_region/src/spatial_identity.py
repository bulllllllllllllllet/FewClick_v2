"""H&E-only cell identity probe with explicit local and region context."""
from __future__ import annotations
import torch
from torch import nn

class SpatialContextIdentity(nn.Module):
    def __init__(self, classes, hidden=128, neighbors=12):
        super().__init__()
        if classes < 2 or neighbors < 1: raise ValueError("classes>=2 and neighbors>=1 required")
        self.neighbors=neighbors
        self.cell=nn.Sequential(nn.LayerNorm(67),nn.Linear(67,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.region=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,hidden),nn.GELU())
        self.message=nn.Sequential(nn.Linear(hidden+3,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.head=nn.Sequential(nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,classes))

    def forward(self, region_tokens, cells, cell_valid, assignment_at_cells):
        if cells.ndim!=3 or cells.shape[-1]!=74 or cell_valid.dtype!=torch.bool: raise ValueError("cells [B,N,74] with boolean validity required")
        if assignment_at_cells.shape[:2]!=region_tokens.shape[:2] or assignment_at_cells.shape[-1]!=cells.shape[1]: raise ValueError("region/cell assignment mismatch")
        if not (torch.isfinite(region_tokens).all() and torch.isfinite(cells).all() and torch.isfinite(assignment_at_cells).all()): raise ValueError("non-finite spatial identity input")
        xy=cells[...,:2]; n=cells.shape[1]
        if ((xy[cell_valid]<0)|(xy[cell_valid]>1)).any(): raise ValueError("valid xy must be normalized")
        base=self.cell(torch.cat((cells[...,10:74],cells[...,:3]),-1))
        region_at_cell=torch.einsum("bkn,bkd->bnd",assignment_at_cells,region_tokens)
        base=base+self.region(region_at_cell)
        if n==1:
            aggregate=torch.zeros_like(base)
        else:
            distance=torch.cdist(xy,xy)
            invalid=(~cell_valid)[:,None,:] | (~cell_valid)[:,:,None] | torch.eye(n,dtype=torch.bool,device=xy.device)[None]
            distance=distance.masked_fill(invalid,float("inf"))
            k=min(self.neighbors,n-1); values,index=distance.topk(k,dim=-1,largest=False)
            gather=index[...,None].expand(-1,-1,-1,base.shape[-1])
            neighbor=torch.gather(base[:,None].expand(-1,n,-1,-1),2,gather)
            nxy=torch.gather(xy[:,None].expand(-1,n,-1,-1),2,index[...,None].expand(-1,-1,-1,2))
            delta=nxy-xy[:,:,None]
            usable=torch.isfinite(values)[...,None]
            # Padding yields inf distances.  Masking *after* the MLP would retain
            # NaNs (`NaN * 0` is NaN), so make the invalid distance finite first.
            safe_values=values.masked_fill(~torch.isfinite(values),0.)
            message=self.message(torch.cat((neighbor,delta,safe_values[...,None]),-1))
            aggregate=(message*usable).sum(2)/usable.sum(2).clamp_min(1)
        logits=self.head(torch.cat((base,aggregate),-1))
        return {"identity_logits":logits,"identity_probabilities":logits.float().softmax(-1),"neighbor_context":aggregate}
