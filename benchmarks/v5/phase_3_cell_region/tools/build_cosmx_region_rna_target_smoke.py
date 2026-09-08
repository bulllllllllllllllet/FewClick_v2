#!/usr/bin/env python3
"""Build an H&E-only cell-feature / semantic-RNA region-target smoke from COAD CosMx."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyvips
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, HERE, Path('/home/zhaoyh/FewClick')):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from benchmarks.v4.phase_1_multiscale.src.common import load_config  # noqa: E402
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder  # noqa: E402
from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells  # noqa: E402
from build_region_rna_soft_target_smoke import image_tensor, phase2_model  # noqa: E402
from run_xenium_xcell_baseline import extract_xcell, load_ctranspath, load_xcell  # noqa: E402


PROGRAMS = {
    'epithelial': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT20', 'MUC1', 'CEACAM5'],
    'stromal_ecm': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'COL6A1', 'TAGLN'],
    't_cell': ['CD3D', 'CD3E', 'TRBC1', 'TRBC2', 'IL7R', 'LTB'],
    'b_plasma_tls': ['MS4A1', 'CD79A', 'CD74', 'CD37', 'JCHAIN', 'MZB1', 'CXCL13', 'CCL19', 'CCL21'],
    'myeloid': ['LYZ', 'LST1', 'TYROBP', 'C1QA', 'C1QB', 'FCER1G'],
    'muscle': ['ACTA2', 'MYH11', 'TAGLN', 'DES'],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/nfs-medical3/fyx/PKU_DATA/CosMx/COAD'))
    p.add_argument('--phase2-config', type=Path, default=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json'))
    p.add_argument('--phase2-checkpoint', type=Path, default=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth'))
    p.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke'))
    p.add_argument('--timestamp', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--x', type=int, default=25500); p.add_argument('--y', type=int, default=34750)
    p.add_argument('--size', type=int, default=1536); p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--max-cells-per-quadrant', type=int, default=255); return p.parse_args()


def he_from_global(x, y, width, height):
    return x / (60215 / width), (y + 4254.99983869582) / (85107 / height)


def global_from_he(x, y, width, height):
    return x * (60215 / width), y * (85107 / height) - 4254.99983869582


def cells_and_polygons(path, x0, y0, size, width, height):
    gx0, gy0 = global_from_he(x0 - 8, y0 - 8, width, height); gx1, gy1 = global_from_he(x0 + size + 8, y0 + size + 8, width, height)
    values = defaultdict(list)
    for chunk in pd.read_csv(path, usecols=['fov', 'cellID', 'x_global_px', 'y_global_px'], chunksize=500_000):
        hit = chunk[(chunk.x_global_px >= gx0) & (chunk.x_global_px < gx1) & (chunk.y_global_px >= gy0) & (chunk.y_global_px < gy1)]
        for row in hit.itertuples(index=False): values[f'{int(row.cellID)}_{int(row.fov)}'].append((float(row.x_global_px), float(row.y_global_px)))
    records=[]; polygons={}
    for cell_id, points in values.items():
        points=np.asarray(points); hx,hy=he_from_global(points[:,0],points[:,1],width,height); cx,cy=hx.mean(),hy.mean()
        if x0 <= cx < x0+size and y0 <= cy < y0+size:
            polygons[cell_id]=np.round(np.column_stack((hx-x0,hy-y0))).astype(np.int32)
            records.append({'cell_id':cell_id,'center_x':cx,'center_y':cy,'bbox_width':float(hx.max()-hx.min()),'bbox_height':float(hy.max()-hy.min())})
    cells=pd.DataFrame(records)
    if cells.empty: raise RuntimeError('no CosMx polygons in requested H&E window')
    return cells, polygons


def choose_xcell_cells(cells, x0, y0, size, maximum):
    value=cells.copy(); value['region']=(value.center_x>=x0+size/2).astype(int)+2*(value.center_y>=y0+size/2).astype(int)
    selected=[]
    for region in range(4):
        part=value[value.region.eq(region)].sort_values(['center_y','center_x','cell_id']).head(maximum)
        if len(part)<20: raise RuntimeError(f'quadrant {region} has only {len(part)} cells')
        selected.append(part)
    return pd.concat(selected,ignore_index=True)


def rna_programs(path, cells, x0, y0, size, width, height):
    gx0,gy0=global_from_he(x0,y0,width,height); gx1,gy1=global_from_he(x0+size,y0+size,width,height)
    source=ds.dataset(path,format='parquet')
    table=source.to_table(columns=['fov','cell_ID','x_global_px','y_global_px','target'], filter=(ds.field('x_global_px')>=gx0)&(ds.field('x_global_px')<gx1)&(ds.field('y_global_px')>=gy0)&(ds.field('y_global_px')<gy1))
    tx=table.to_pandas(); tx['cell_id']=tx.cell_ID.astype(str)+'_'+tx.fov.astype(str); tx=tx[tx.cell_id.isin(set(cells.cell_id))]
    if tx.empty: raise RuntimeError('no transcripts matched cells in H&E window')
    total=tx.groupby('cell_id').size(); result=np.zeros((len(cells),len(PROGRAMS)),np.float32); coverage=[]
    for j, markers in enumerate(PROGRAMS.values()):
        count=tx[tx.target.isin(markers)].groupby('cell_id').size().reindex(cells.cell_id,fill_value=0).to_numpy()
        result[:,j]=np.log1p(count/total.reindex(cells.cell_id,fill_value=1).to_numpy()*1e4); coverage.append(sorted(set(markers)&set(tx.target.unique())))
    return result, tx, coverage


def main():
    a=parse_args(); out=a.output_root/f'cosmx_region_rna_target_smoke_{a.timestamp}'
    if out.exists(): raise FileExistsError(out)
    device=torch.device(a.device); image_vips=pyvips.Image.new_from_file(str(a.data_root/'HE.tif'),access='sequential'); width,height=image_vips.width,image_vips.height
    if a.x<0 or a.y<0 or a.x+a.size>width or a.y+a.size>height: raise ValueError('requested H&E window outside raster')
    all_cells, all_polygons=cells_and_polygons(a.data_root/'segmentation_mask/polygons.csv',a.x,a.y,a.size,width,height)
    program, tx, marker_coverage=rna_programs(a.data_root/'transcripts/tx_file.parquet',all_cells,a.x,a.y,a.size,width,height)
    keep=np.flatnonzero(np.isin(all_cells.cell_id,tx.cell_id.unique())); all_cells=all_cells.iloc[keep].reset_index(drop=True); program=program[keep]
    xcells=choose_xcell_cells(all_cells,a.x,a.y,a.size,a.max_cells_per_quadrant); pos=pd.Index(all_cells.cell_id).get_indexer(xcells.cell_id)
    image=np.ndarray(buffer=image_vips.crop(a.x,a.y,a.size,a.size).write_to_memory(),dtype=np.uint8,shape=(a.size,a.size,image_vips.bands))[:,:,:3]
    xpolys={key:all_polygons[key] for key in xcells.cell_id}; ctp=load_ctranspath(Path('/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth'),device); xcell=load_xcell(Path('/home/zhaoyh/FewClick/module/checkpoint/he_model_best.pth'),device)
    reg=extract_xcell(image,xcells,xpolys,a.x,a.y,ctp,xcell,device,64,a.max_cells_per_quadrant)
    model=phase2_model(a,device); examples=[]; starts=range(0,a.size,a.patch_size)
    with torch.inference_mode():
      for top in starts:
       for left in starts:
        all_idx=np.flatnonzero((all_cells.center_x>=a.x+left)&(all_cells.center_x<a.x+left+a.patch_size)&(all_cells.center_y>=a.y+top)&(all_cells.center_y<a.y+top+a.patch_size))
        in_idx=np.flatnonzero((xcells.center_x>=a.x+left)&(xcells.center_x<a.x+left+a.patch_size)&(xcells.center_y>=a.y+top)&(xcells.center_y<a.y+top+a.patch_size))
        if not len(all_idx) or not len(in_idx): raise RuntimeError(f'empty RNA/input cells in patch {left},{top}')
        token_out=model(image_tensor(image[top:top+a.patch_size,left:left+a.patch_size]).unsqueeze(0).to(device),return_full_assignment=False)
        local=xcells.iloc[in_idx]; xy=np.column_stack(((local.center_x-a.x-left)/a.patch_size,(local.center_y-a.y-top)/a.patch_size)).astype(np.float32)
        mass=sample_assignment_at_cells(token_out['assignment_low'],torch.from_numpy(xy).unsqueeze(0).to(device))[0].float().cpu().numpy()
        rxy=np.column_stack(((all_cells.iloc[all_idx].center_x-a.x-left)/a.patch_size,(all_cells.iloc[all_idx].center_y-a.y-top)/a.patch_size)).astype(np.float32)
        rmass=sample_assignment_at_cells(token_out['assignment_low'],torch.from_numpy(rxy).unsqueeze(0).to(device))[0].float().cpu().numpy()
        target=(rmass@program[all_idx])/rmass.sum(1,keepdims=True).clip(1e-8)
        area=np.log1p((local.bbox_width*local.bbox_height).to_numpy()).astype(np.float32); feature=np.concatenate((xy,area[:,None],np.zeros((len(xy),7),np.float32),reg[in_idx]),1)
        examples.append((left,top,feature,mass,token_out['region_tokens'][0].float().cpu().numpy(),target.astype(np.float32),rmass.sum(1).astype(np.float32),local.cell_id.to_numpy(),len(all_idx)))
    maximum=max(len(e[2]) for e in examples); n=len(examples); cells=np.zeros((n,maximum,74),np.float32); valid=np.zeros((n,maximum),bool); assignment=np.zeros((n,64,maximum),np.float32)
    for i,e in enumerate(examples): cells[i,:len(e[2])]=e[2];valid[i,:len(e[2])]=True;assignment[i,:,:len(e[2])]=e[3]
    out.mkdir(parents=True); np.savez_compressed(out/'region_rna_contract.npz',region_tokens=np.stack([e[4] for e in examples]),cells=cells,cell_valid=valid,assignment_at_cells=assignment,rna_targets=np.stack([e[5] for e in examples]),rna_coverage=np.stack([e[6] for e in examples]))
    pd.DataFrame([{'patch_index':i,'left_px':e[0],'top_px':e[1],'input_he_cells':len(e[2]),'rna_target_cells':e[8],'cell_id_sha256':__import__('hashlib').sha256('|'.join(e[7]).encode()).hexdigest()} for i,e in enumerate(examples)]).to_csv(out/'patch_manifest.csv',index=False)
    meta={'purpose':'CosMx semantic region RNA target; RNA target only and input reg64 is H&E-only frozen XCellFormer','window_he':[a.x,a.y,a.size],'patches':n,'all_rna_cells':len(all_cells),'xcell_input_cells':len(xcells),'transcript_rows':len(tx),'program_names':list(PROGRAMS),'marker_coverage':marker_coverage,'cell_feature_shape':list(cells.shape),'assignment_shape':list(assignment.shape),'rna_target_shape':[n,64,len(PROGRAMS)],'acceptance':{'finite':bool(np.isfinite(cells).all() and np.isfinite(assignment).all()),'assignment_sums_one':bool(np.allclose(assignment.sum(1)[valid],1,atol=1e-4))}}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'event':'complete','output':str(out),**meta},ensure_ascii=False))


if __name__=='__main__': main()
