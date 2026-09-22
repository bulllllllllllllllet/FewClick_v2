#!/usr/bin/env python3
"""Controlled GDPH segmentation-only versus CRC panel-2 auxiliary smoke."""
from __future__ import annotations
import argparse,json,re,sys
from itertools import cycle
from pathlib import Path
import numpy as np, pyvips, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset,collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.panel2_joint_contract import Panel2JointInputs
from benchmarks.v5.phase_3_cell_region.src.region_footprint_contract import patch_pixel_grid_indices,project_assignment_to_grid,project_observed_targets
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss
P2ROOT=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240');V4=Path('/nfs-medical3/zyh/v4');P3=V4/'phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth'
def image(path):
 a=np.asarray(Image.open(path).convert('RGB'),np.float32).transpose(2,0,1)/255.;t=torch.from_numpy(a);return (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def selected(rows,split,n):
 out=[]
 for case in sorted({x['case_id'] for x in rows}):
  q=sorted([x for x in rows if x['case_id']==case and x['split']==split],key=lambda x:x['sample_id'])
  if len(q)<n:raise ValueError(f'{case}/{split} insufficient')
  out+=q[:n]
 return out
def crc_contract(rows,p2,d):
 meta=json.loads(Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600/metadata.json').read_text())['records'];paths={(x['case_id'],x['patch_name']):x for x in meta};common=Path('/nfs-medical6/data/SRRSH/crc/panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations');mats={c:json.loads((common/c/'coordinate_transform_manifest.json').read_text())['panel2_mif_dapi_to_panel1_he_level0_composed'] for c in {x['case_id'] for x in rows}};out=[]
 for row in rows:
  case,patch=row['case_id'],row['sample_id'].split(':',1)[1];raw=np.load(Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600')/row['cache_path']);xy=raw['he_xy'].astype(np.float32);h,w=raw['he_mask'].shape;cells=encode_xcell_features(np.c_[xy[:,0]/w,xy[:,1]/h,np.zeros(len(xy)),np.zeros(len(xy))],raw['reg_features']);x=image(paths[(case,patch)]['he_path']).unsqueeze(0).to(d)
  with torch.no_grad():z=p2(x,return_full_assignment=False,return_tokens=True)
  c=torch.from_numpy(cells).unsqueeze(0).to(d);v=torch.ones((1,len(cells)),dtype=torch.bool,device=d);mass=F.grid_sample(z['assignment_low'],(c[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);ox,oy=map(float,re.fullmatch(r'x(\d+)_y(\d+)',patch).groups());grid=patch_pixel_grid_indices(z['assignment_low'].shape[-2],z['assignment_low'].shape[-1],(ox,oy),mats[case],.2457,1024.,source_pixel_stride=w/z['assignment_low'].shape[-1]);proj=project_assignment_to_grid(z['assignment_low'][0],grid);targets={(int(q['grid_x']),int(q['grid_y'])):(q['counts'],q['observed_cells'],q['coverage']) for q in row['panel2_targets']};target=project_observed_targets(proj,targets,min_token_grid_fraction=.8);inputs=Panel2JointInputs(z['region_tokens'],mass,c,v,torch.tensor([len(cells)],dtype=torch.float32,device=d));inputs.validate();out.append((inputs,target))
 return out
def panel2_validation_loss(cell,immune,items):
 total_loss=0.;total_tokens=0
 with torch.no_grad():
  for inputs,target in items:
   logits=immune(inputs.fuse(cell)['fused_tokens'])['immune_observed_logits'][0]
   loss,count=observed_composition_loss(logits,target['immune_counts'],target['immune_observed_cells'],target['immune_coverage'])
   total_loss+=float(loss)*count;total_tokens+=count
 if total_tokens<1:raise RuntimeError('panel-2 validation has no supervised tokens')
 return total_loss/total_tokens,total_tokens
def main():
 a=argparse.ArgumentParser();a.add_argument('--timestamp',required=True);a.add_argument('--device',default='cuda:1');a.add_argument('--steps',type=int,default=12);a.add_argument('--panel2-weight',type=float,default=.1);a.add_argument('--limit-train',type=int,default=8);a.add_argument('--limit-val',type=int,default=8);a.add_argument('--panel2-train-per-case',type=int,default=4);a.add_argument('--panel2-val-per-case',type=int,default=2);a.add_argument('--seed',type=int,default=20260921);a.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));z=a.parse_args();out=z.output_root/f'phase3_seg_panel2_joint_smoke_{z.timestamp}';out.mkdir(parents=True,exist_ok=False);d=torch.device(z.device);torch.manual_seed(z.seed);cfg=load_config(P2ROOT/'config_snapshot.json');train0=CellRegionDataset(V4/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet',V4/'phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet',V4/'phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet',cfg,'train');val0=CellRegionDataset(V4/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet',V4/'phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet',V4/'phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet',cfg,'val');rng=np.random.default_rng(z.seed);coll=lambda q:collate_cell_region(q,255);tl=DataLoader(Subset(train0,sorted(rng.choice(len(train0),z.limit_train,False))),batch_size=2,collate_fn=coll);vl=DataLoader(Subset(val0,sorted(rng.choice(len(val0),z.limit_val,False))),batch_size=2,collate_fn=coll);p2=DeepRegionEncoder(cfg,len(cfg['data']['class_map']),64,256).to(d);p2.load_state_dict(torch.load(P2ROOT/'checkpoint_epoch_028.pth',map_location=d,weights_only=False)['model']);p2.eval().requires_grad_(False);spec=json.loads(Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_spatial_manifest_0921_173523/manifest.json').read_text());ctr=crc_contract(selected(spec['samples'],'train',z.panel2_train_per_case),p2,d);cva=crc_contract(selected(spec['samples'],'val',z.panel2_val_per_case),p2,d);base=torch.load(P3,map_location=d,weights_only=False);results=[]
 for arm in ('seg_only','seg_plus_panel2'):
  torch.manual_seed(z.seed);cell=CellToRegionAttention(256,74).to(d);cell.load_state_dict(base['cell']);head=torch.nn.Linear(256,12).to(d);head.load_state_dict(base['head']);immune=ObservedRegionEvidenceHeads().to(d);opt=torch.optim.AdamW(list(cell.parameters())+list(head.parameters())+list(immune.parameters()),lr=1e-4);losses=[]
  for step,b in zip(range(z.steps),cycle(tl)):
   opt.zero_grad()
   with torch.no_grad(): q=p2(b['image'].to(d),False,True)
   m=F.grid_sample(q['assignment_low'],(b['cells'].to(d)[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);f=cell(q['region_tokens'],m,b['cells'].to(d),b['cell_valid'].to(d),b['total_cell_count'].to(d))['fused_tokens'];seg=F.cross_entropy(head(f).flatten(0,1),region_labels(q['assignment_low'],b['mask'].to(d),12,255).flatten(),ignore_index=255);aux=seg*0;item=ctr[step%len(ctr)]
   if arm!='seg_only':
    inputs,target=item;fi=inputs.fuse(cell)['fused_tokens'];log=immune(fi)['immune_observed_logits'][0];aux,_=observed_composition_loss(log,target['immune_counts'],target['immune_observed_cells'],target['immune_coverage'])
   (seg+z.panel2_weight*aux).backward();opt.step();losses.append((float(seg.detach()),float(aux.detach())))
  cell.eval();head.eval();correct=tot=0
  with torch.no_grad():
   for b in vl:q=p2(b['image'].to(d),False,True);m=F.grid_sample(q['assignment_low'],(b['cells'].to(d)[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);pred=head(cell(q['region_tokens'],m,b['cells'].to(d),b['cell_valid'].to(d),b['total_cell_count'].to(d))['fused_tokens']).argmax(-1);lab=region_labels(q['assignment_low'],b['mask'].to(d),12,255);keep=lab!=255;correct+=((pred==lab)&keep).sum().item();tot+=keep.sum().item()
  panel2_loss,panel2_tokens=panel2_validation_loss(cell,immune,cva)
  results.append({'arm':arm,'mean_seg_loss':float(np.mean([x[0] for x in losses])),'mean_panel2_loss':float(np.mean([x[1] for x in losses])),'val_region_accuracy':correct/tot,'val_panel2_loss':panel2_loss,'val_panel2_tokens':panel2_tokens})
 (out/'metrics.json').write_text(json.dumps({'status':'complete','scope':'joint diagnostic smoke, not final FewClick evaluation','steps':z.steps,'panel2_weight':z.panel2_weight,'results':results},indent=2));print(json.dumps(results))
if __name__=='__main__':main()
