#!/usr/bin/env python3
"""Small alternating GDPH segmentation + COAD RNA auxiliary-loss Phase-3 smoke."""
from __future__ import annotations
import argparse, json, sys
from itertools import cycle
from pathlib import Path
import numpy as np
import pyvips  # Load libvips/libjpeg before torch; v4 GDPH reader imports the same binding.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels

P2CFG=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json')
P3CFG=Path('/home/zhaoyh/FewClick_v2/benchmarks/v4/phase_3_cell_region/configs/phase3_train_10x.yaml')
PATCH=Path('/nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet')
SLIC=Path('/nfs-medical3/zyh/v4/phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet')
TRAIN_ROUTE=Path('/nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet')
VAL_ROUTE=Path('/nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet')
P3CKPT=Path('/nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth')

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract-dir',type=Path,required=True);p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke'));p.add_argument('--timestamp',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=20);p.add_argument('--rna-weight',type=float,default=.1);p.add_argument('--limit-train',type=int,default=8);p.add_argument('--limit-val',type=int,default=8);return p.parse_args()
def main():
 a=args();out=a.output_root/f'phase3_seg_rna_joint_smoke_{a.timestamp}'
 if out.exists():raise FileExistsError(out)
 d=torch.device(a.device);p2c=load_config(P2CFG);p3c=load_config(P3CFG)
 train=Subset(CellRegionDataset(PATCH,SLIC,TRAIN_ROUTE,p2c,'train'),list(range(a.limit_train)));val=Subset(CellRegionDataset(PATCH,SLIC,VAL_ROUTE,p2c,'val'),list(range(a.limit_val)))
 coll=lambda x:collate_cell_region(x,int(p3c['data']['max_cells_per_patch']))
 tl=DataLoader(train,batch_size=2,shuffle=False,collate_fn=coll,num_workers=0);vl=DataLoader(val,batch_size=2,shuffle=False,collate_fn=coll,num_workers=0)
 phase2=DeepRegionEncoder(p2c,len(p2c['data']['class_map']),64,256).to(d);phase2.load_state_dict(torch.load(p3c['model']['phase2_checkpoint'],map_location=d,weights_only=False)['model']);phase2.eval()
 for x in phase2.parameters():x.requires_grad=False
 base=torch.load(P3CKPT,map_location=d,weights_only=False);raw=np.load(a.contract_dir/'region_rna_contract.npz');tokens=torch.from_numpy(raw['region_tokens']).float().to(d);cells=torch.from_numpy(raw['cells']).float().to(d);valid=torch.from_numpy(raw['cell_valid']).bool().to(d);mass=torch.from_numpy(raw['assignment_at_cells']).float().to(d);target=torch.from_numpy(raw['rna_targets']).float().to(d)
 mean,std=target.reshape(-1,target.shape[-1]).mean(0),target.reshape(-1,target.shape[-1]).std(0).clamp_min(1e-6)
 result=[]
 for arm in ('seg_only','paired_rna','shuffled_rna'):
  torch.manual_seed(20260908);cell=CellToRegionAttention(256,74).to(d);cell.load_state_dict(base['cell']);head=torch.nn.Linear(256,12).to(d);head.load_state_dict(base['head']);rna=torch.nn.Linear(256,target.shape[-1]).to(d);opt=torch.optim.AdamW(list(cell.parameters())+list(head.parameters())+list(rna.parameters()),lr=1e-4)
  y=target.clone()
  if arm=='shuffled_rna': y=y.reshape(-1,y.shape[-1])[torch.randperm(y.numel()//y.shape[-1],device=d)].reshape_as(y)
  losses=[]
  for batch in cycle(tl):
   if len(losses)>=a.steps:break
   opt.zero_grad(set_to_none=True);image=batch['image'].to(d);mask=batch['mask'].to(d)
   with torch.no_grad():z=phase2(image,return_full_assignment=False,return_tokens=True)
   labels=region_labels(z['assignment_low'],mask,12,255);c=batch['cells'].to(d);v=batch['cell_valid'].to(d);m=F.grid_sample(z['assignment_low'],(c[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2)
   fused=cell(z['region_tokens'],m,c,v,batch['total_cell_count'].to(d))['fused_tokens'];seg=F.cross_entropy(head(fused).flatten(0,1),labels.flatten(),ignore_index=255);aux=F.mse_loss((rna(cell(tokens,mass,cells,valid,valid.sum(1).float())['fused_tokens'])-mean)/std,(y-mean)/std);loss=seg+(a.rna_weight*aux if arm!='seg_only' else 0);loss.backward();opt.step();losses.append((float(seg.detach()),float(aux.detach())))
  cell.eval();head.eval();correct=total=0
  with torch.no_grad():
   for batch in vl:
    z=phase2(batch['image'].to(d),return_full_assignment=False,return_tokens=True);c=batch['cells'].to(d);v=batch['cell_valid'].to(d);m=F.grid_sample(z['assignment_low'],(c[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);pred=head(cell(z['region_tokens'],m,c,v,batch['total_cell_count'].to(d))['fused_tokens']).argmax(-1);lab=region_labels(z['assignment_low'],batch['mask'].to(d),12,255);keep=lab!=255;correct+=((pred==lab)&keep).sum().item();total+=keep.sum().item()
  result.append({'arm':arm,'steps':a.steps,'mean_seg_loss':float(np.mean([x[0] for x in losses])),'mean_rna_loss':float(np.mean([x[1] for x in losses])),'val_region_acc':correct/max(total,1)})
 out.mkdir(parents=True);(out/'metrics.json').write_text(json.dumps({'purpose':'alternating GDPH segmentation and COAD RNA auxiliary smoke; not a formal downstream result','steps':a.steps,'rna_weight':a.rna_weight,'results':result,'acceptance':'all finite and paired arm must not materially reduce GDPH val region accuracy vs seg_only'},indent=2),encoding='utf-8');print(json.dumps({'event':'complete','output':str(out),'results':result},ensure_ascii=False))
if __name__=='__main__':main()
