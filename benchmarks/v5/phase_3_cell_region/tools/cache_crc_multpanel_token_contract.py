#!/usr/bin/env python3
"""Cache frozen Phase-2 inputs and both panel targets after spatial splitting."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features
from benchmarks.v5.phase_3_cell_region.src.region_footprint_contract import patch_pixel_grid_indices,project_assignment_to_grid,project_observed_targets

P2ROOT=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240')
CACHE=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600')
COMMON=Path('/nfs-medical6/data/SRRSH/crc/panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations')

def image(path):
 a=np.asarray(Image.open(path).convert('RGB'),np.float32).transpose(2,0,1)/255.;t=torch.from_numpy(a)
 return (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def target_map(rows):return {(int(x['grid_x']),int(x['grid_y'])):(x['counts'],x['observed_cells'],x['coverage']) for x in rows}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timestamp',required=True);p.add_argument('--manifest-dir',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_spatial_manifest_0922_195841'));p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);p.add_argument('--min-token-grid-fraction',type=float,default=.8);a=p.parse_args()
 if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp) or not 0<a.min_token_grid_fraction<=1:raise ValueError('valid timestamp and overlap fraction required')
 out=a.output_root/f'crc_multpanel_token_contract_{a.timestamp}';out.mkdir(parents=True,exist_ok=False)
 spec=json.loads((a.manifest_dir/'manifest.json').read_text());rows=spec['samples'];rows=rows[:a.limit] if a.limit else rows
 paths={(x['case_id'],x['patch_name']):x['he_path'] for x in json.loads((CACHE/'metadata.json').read_text())['records']}
 d=torch.device(a.device);cfg=load_config(P2ROOT/'config_snapshot.json');p2=DeepRegionEncoder(cfg,len(cfg['data']['class_map']),64,256).to(d);p2.load_state_dict(torch.load(P2ROOT/'checkpoint_epoch_028.pth',map_location=d,weights_only=False)['model']);p2.eval().requires_grad_(False)
 matrices={case:json.loads((COMMON/case/'coordinate_transform_manifest.json').read_text())['panel2_mif_dapi_to_panel1_he_level0_composed'] for case in {x['case_id'] for x in rows}};index=[]
 for i,row in enumerate(rows):
  case,patch=row['case_id'],row['sample_id'].split(':',1)[1];raw=np.load(CACHE/row['cache_path']);xy=raw['he_xy'].astype(np.float32);h,w=raw['he_mask'].shape;cells=encode_xcell_features(np.c_[xy[:,0]/w,xy[:,1]/h,np.zeros(len(xy)),np.zeros(len(xy))],raw['reg_features']);x=image(paths[(case,patch)]).unsqueeze(0).to(d)
  with torch.no_grad():z=p2(x,False,True)
  assignment=F.grid_sample(z['assignment_low'],(torch.from_numpy(cells[:,:2]).to(d)*2-1).unsqueeze(0).unsqueeze(1),align_corners=False).squeeze(2)[0];ox,oy=map(float,re.fullmatch(r'x(\d+)_y(\d+)',patch).groups());grid=patch_pixel_grid_indices(z['assignment_low'].shape[-2],z['assignment_low'].shape[-1],(ox,oy),matrices[case],.2457,1024.,source_pixel_stride=w/z['assignment_low'].shape[-1]);projection=project_assignment_to_grid(z['assignment_low'][0],grid);one=project_observed_targets(projection,target_map(row['panel1_targets']),min_token_grid_fraction=a.min_token_grid_fraction);two=project_observed_targets(projection,target_map(row['panel2_targets']),min_token_grid_fraction=a.min_token_grid_fraction)
  name=f'{i:05d}_{case}_{patch}.npz';np.savez_compressed(out/name,region_tokens=z['region_tokens'][0].cpu().numpy().astype(np.float32),cells=cells.astype(np.float32),assignment_at_cells=assignment.cpu().numpy().astype(np.float32),total_cell_count=np.array(len(cells),np.float32),panel1_counts=one['immune_counts'].cpu().numpy(),panel1_observed_cells=one['immune_observed_cells'].cpu().numpy(),panel1_coverage=one['immune_coverage'].cpu().numpy(),panel1_valid=one['immune_target_valid'].cpu().numpy(),panel2_counts=two['immune_counts'].cpu().numpy(),panel2_observed_cells=two['immune_observed_cells'].cpu().numpy(),panel2_coverage=two['immune_coverage'].cpu().numpy(),panel2_valid=two['immune_target_valid'].cpu().numpy())
  index.append({'sample_id':row['sample_id'],'split':row['split'],'spatial_group_id':row['spatial_group_id'],'contract_path':name,'panel1_valid_tokens':int(one['immune_target_valid'].sum()),'panel2_valid_tokens':int(two['immune_target_valid'].sum())})
 manifest_hash=hashlib.sha256((a.manifest_dir/'manifest.json').read_bytes()).hexdigest();(out/'index.json').write_text(json.dumps(index,indent=2));(out/'metadata.json').write_text(json.dumps({'status':'complete','manifest_hash':manifest_hash,'manifest_dir':str(a.manifest_dir),'patches':len(index),'min_token_grid_fraction':a.min_token_grid_fraction,'phase2_checkpoint':str(P2ROOT/'checkpoint_epoch_028.pth'),'scope':'frozen token contract; regional observations only'},indent=2));print(json.dumps({'output':str(out),'patches':len(index)}))
if __name__=='__main__':main()
