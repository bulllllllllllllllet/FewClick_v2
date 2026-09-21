#!/usr/bin/env python3
"""Map accepted CRC local H&E patches into panel-2 regional evidence grids."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
CASES=("2017-23800D","2019-00538F","2022-39941E")
def corners(x,y,side,matrix):
 p=np.array([[x,y,1],[x+side,y,1],[x,y+side,1],[x+side,y+side,1]],float);q=p@np.asarray(matrix,float).T;return q[:,:2]
def intersecting_regions(inventory,lo,hi):
 g=float(inventory.grid_um.iloc[0])
 return inventory[(inventory.grid_x*g<hi[0])&(inventory.grid_x*g+g>lo[0])&(inventory.grid_y*g<hi[1])&(inventory.grid_y*g+g>lo[1])]
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--timestamp',required=True);a.add_argument('--cache-dir',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600'));a.add_argument('--inventory-dir',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_region_supervision_inventory_0921_004107'));a.add_argument('--root',type=Path,default=Path('/nfs-medical6/data/SRRSH/crc'));a.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));a.add_argument('--patch-side-px',type=float,default=512.);a.add_argument('--mpp',type=float,default=.2457);z=a.parse_args()
 if not re.fullmatch(r'\d{4}_\d{6}',z.timestamp):raise ValueError('timestamp MMDD_HHMMSS required')
 out=z.output_root/f'crc_patch_region_alignment_{z.timestamp}';out.mkdir(parents=True,exist_ok=False);meta=json.loads((z.cache_dir/'metadata.json').read_text()); rows=[]
 inventories={case:pd.read_csv(z.inventory_dir/f'{case}_regions.csv') for case in CASES}
 matrices={case:json.loads((z.root/'panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations'/case/'coordinate_transform_manifest.json').read_text())['panel2_mif_dapi_to_panel1_he_level0_composed'] for case in CASES}
 for r in meta['records']:
  case=r['case_id'];t=json.loads(Path(r['transform_path']).read_text())
  if t.get('status')!='ok' or t.get('he_tissue_fraction',0)<.15:continue
  m=matrices[case]
  hit=re.fullmatch(r'x(\d+)_y(\d+)',r['patch_name']);x,y=map(float,hit.groups());q=corners(x,y,z.patch_side_px,m)*z.mpp;lo=q.min(0);hi=q.max(0)
  selected=intersecting_regions(inventories[case],lo,hi)
  for _,v in selected.iterrows():
   panel2={k:(int(v[k]) if k.endswith('_count') or k.endswith('_cells') else float(v[k])) for k in v.index if k.startswith('panel2_')}
   rows.append({'case_id':case,'patch_name':r['patch_name'],'cache_path':r['path'],'transform_path':r['transform_path'],'he_tissue_fraction':t['he_tissue_fraction'],'footprint_um':[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],'spatial_group_id':v.spatial_group_id,'grid_x':int(v.grid_x),'grid_y':int(v.grid_y),'panel1_coverage':float(v.panel1_coverage),**panel2,'candidate_only':True})
 pd.DataFrame(rows).to_json(out/'alignment_candidates.jsonl',orient='records',lines=True);(out/'metadata.json').write_text(json.dumps({'schema':'crc_patch_region_alignment_candidate_v1','records':len(rows),'candidate_only':True,'reason':'footprint/grid overlap and local QC do not prove region-token correspondence'},indent=2));print(json.dumps({'output':str(out),'records':len(rows)}))
if __name__=='__main__':main()
