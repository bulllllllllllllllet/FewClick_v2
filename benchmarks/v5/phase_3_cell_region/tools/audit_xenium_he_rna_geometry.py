#!/usr/bin/env python3
"""Audit COAD H&E/cell-boundary/transcript coordinate consistency by cell ID."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[4]; SCRIPT=Path(__file__).resolve().parent
sys.path[:0]=[str(SCRIPT),str(ROOT)]
from run_xenium_rna_pilot import load_transcripts, normalized_counts
from run_xenium_xcell_baseline import read_polygons

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=Path('/nfs-medical3/fyx/PKU_DATA/Xenium/COAD'));p.add_argument('--summary',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806/cell_summary.csv'));p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/rna_geometry_audit'));p.add_argument('--timestamp');p.add_argument('--sample-cells',type=int,default=50);p.add_argument('--seed',type=int,default=20260908);p.add_argument('--x',type=int,default=1000);p.add_argument('--y',type=int,default=2000);p.add_argument('--size',type=int,default=1500);return p.parse_args()
def main():
 a=args();stamp=a.timestamp or datetime.now().strftime('%m%d_%H%M%S');out=a.output_root/f'rna_geometry_audit_{stamp}'
 if out.exists():raise FileExistsError(out)
 cells=pd.read_csv(a.summary,dtype={'cell_id':str});tx=load_transcripts(a.data_root/'transcripts/transcripts.parquet',set(cells.cell_id),a.x,a.y,a.size,20.0);counts=normalized_counts(tx,cells.cell_id.tolist(),30);keep=cells.set_index('cell_id').loc[counts.index].reset_index();sample=keep.sample(min(a.sample_cells,len(keep)),random_state=a.seed).sort_values('cell_id').reset_index(drop=True);poly=read_polygons(a.data_root/'segmentation_mask/cell_boundaries.csv',set(sample.cell_id),a.x,a.y)
 rows=[];signed=[]
 for row in sample.itertuples(index=False):
  points=tx[tx.cell_id.astype(str)==row.cell_id][['x_location','y_location']].to_numpy(dtype=np.float32)-np.array([a.x,a.y],np.float32);distance=np.asarray([cv2.pointPolygonTest(poly[row.cell_id].astype(np.float32),tuple(p),True) for p in points]);signed.extend(distance.tolist());inside=distance>=0;rows.append({'cell_id':row.cell_id,'transcripts':len(points),'inside_boundary':int(inside.sum()),'inside_fraction':float(inside.mean()) if len(inside) else float('nan'),'center_inside_boundary':bool(cv2.pointPolygonTest(poly[row.cell_id].astype(np.float32),(float(row.center_x-a.x),float(row.center_y-a.y)),False)>=0)})
 detail=pd.DataFrame(rows);out.mkdir(parents=True);detail.to_csv(out/'cell_audit.csv',index=False)
 image_v=pyvips.Image.new_from_file(str(a.data_root/'HE.tif'),access='sequential').crop(a.x,a.y,a.size,a.size);image=np.ndarray(buffer=image_v.write_to_memory(),dtype=np.uint8,shape=(a.size,a.size,image_v.bands))[:,:,:3];canvas=Image.fromarray(image).resize((750,750));draw=ImageDraw.Draw(canvas)
 for row in sample.head(12).itertuples(index=False):
  p=poly[row.cell_id]/2;draw.line([tuple(x) for x in np.vstack((p,p[0]))],fill=(0,255,0),width=1);pts=tx[tx.cell_id.astype(str)==row.cell_id][['x_location','y_location']].to_numpy()-np.array([a.x,a.y]);
  for x,y in pts[::max(1,len(pts)//80)]:draw.point((x/2,y/2),fill=(255,0,0))
 canvas.save(out/'he_boundary_transcript_overlay.png')
 signed=np.asarray(signed);metrics={'purpose':'independent H&E/cell-boundary/transcript coordinate audit','sampled_cells':len(detail),'sampled_transcripts':int(detail.transcripts.sum()),'transcript_inside_fraction_mean':float(detail.inside_boundary.sum()/detail.transcripts.sum()),'transcript_within_2px_fraction':float((signed>=-2).mean()),'transcript_within_5px_fraction':float((signed>=-5).mean()),'signed_distance_median_px':float(np.median(signed)),'cell_center_inside_fraction':float(detail.center_inside_boundary.mean()),'minimum_cell_inside_fraction':float(detail.inside_fraction.min()),'acceptance':'>=0.95 transcript points inside or within 2px of assigned cell boundary; visual overlay reviewed separately'};(out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps({'event':'complete','output':str(out),**metrics},ensure_ascii=False))
if __name__=='__main__':main()
