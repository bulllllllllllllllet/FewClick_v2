"""Border bilinear assignment sampling using deterministic dense matrix products."""
import torch


def sample_assignment_at_cells(assignment,cells):
    if assignment.ndim!=4 or cells.ndim!=3 or cells.shape[0]!=assignment.shape[0] or cells.shape[-1]<2:
        raise ValueError('expected B,K,H,W assignments and B,N,>=2 cells')
    if cells.shape[1]==0:return assignment.new_zeros(assignment.shape[0],assignment.shape[1],0)
    h,w=assignment.shape[-2:];grid=cells[...,:2]*2-1
    x=(((grid[...,0]+1)*w-1)/2).clamp(0,w-1)
    y=(((grid[...,1]+1)*h-1)/2).clamp(0,h-1)
    wx=(1-(x[...,None]-torch.arange(w,device=x.device,dtype=x.dtype)).abs()).clamp(min=0)
    wy=(1-(y[...,None]-torch.arange(h,device=y.device,dtype=y.dtype)).abs()).clamp(min=0)
    weights=(wy[...,None]*wx[...,None,:]).flatten(2)
    return torch.bmm(assignment.flatten(2),weights.transpose(1,2))
