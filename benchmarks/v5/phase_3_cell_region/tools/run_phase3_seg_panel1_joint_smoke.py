#!/usr/bin/env python3
"""Controlled GDPH segmentation versus cached CRC Panel1 auxiliary smoke."""
from __future__ import annotations
import argparse,json,sys
from itertools import cycle
from pathlib import Path
import numpy as np,pyvips,torch,torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset,collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.panel2_joint_contract import Panel2JointInputs
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss
P2ROOT=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240');V4=Path('/nfs-medical3/zyh/v4');P3=V4/'phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth'
def cached(rows,root,d,panel):
 out=[]
 for row in rows:
  x=np.load(root/row['contract_path']);cells=torch.from_numpy(x['cells']).unsqueeze(0).to(d);inputs=Panel2JointInputs(torch.from_numpy(x['region_tokens']).unsqueeze(0).to(d),torch.from_numpy(x['assignment_at_cells']).unsqueeze(0).to(d),cells,torch.ones((1,len(cells[0])),dtype=torch.bool,device=d),torch.from_numpy(x['total_cell_count']).reshape(1).to(d));inputs.validate();out.append((inputs,torch.from_numpy(x[f'{panel}_counts']).to(d),torch.from_numpy(x[f'{panel}_observed_cells']).to(d),torch.from_numpy(x[f'{panel}_coverage']).to(d)))
 return out
def aux_loss(cell,evidence,item,panel):
 inputs,counts,observed,coverage=item;key={'panel1':'structure_observed_logits','panel2':'immune_observed_logits'}[panel];logits=evidence(inputs.fuse(cell)['fused_tokens'])[key][0];return observed_composition_loss(logits,counts,observed,coverage)
def evaluate_aux(cell,evidence,items,panel):
 total=count=0
 with torch.no_grad():
  for item in items:
   value,n=aux_loss(cell,evidence,item,panel);total+=float(value)*n;count+=n
 if not count:raise RuntimeError('CRC validation has no supervised tokens')
 return total/count,count
def take(index,split,n):
 rows=[x for x in index if x['split']==split]
 return sorted(rows,key=lambda x:x['sample_id'])[:n]
def main():
 a=argparse.ArgumentParser();a.add_argument('--timestamp',required=True);a.add_argument('--contract-dir',type=Path,required=True);a.add_argument('--device',default='cuda:1');a.add_argument('--steps',type=int,default=12);a.add_argument('--panel-weight',type=float,default=.1);a.add_argument('--limit-train',type=int,default=8);a.add_argument('--limit-val',type=int,default=8);a.add_argument('--crc-train',type=int,default=12);a.add_argument('--crc-val',type=int,default=6);a.add_argument('--seed',type=int,default=17);a.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));z=a.parse_args();out=z.output_root/f'phase3_seg_panel1_joint_smoke_{z.timestamp}';out.mkdir(parents=True,exist_ok=False);d=torch.device(z.device);torch.manual_seed(z.seed);cfg=load_config(P2ROOT/'config_snapshot.json');phase1=V4/'phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet';slic=V4/'phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet';train0=CellRegionDataset(phase1,slic,V4/'phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet',cfg,'train');val0=CellRegionDataset(phase1,slic,V4/'phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet',cfg,'val');rng=np.random.default_rng(z.seed);coll=lambda q:collate_cell_region(q,255);tl=DataLoader(Subset(train0,sorted(rng.choice(len(train0),z.limit_train,False))),batch_size=2,collate_fn=coll);vl=DataLoader(Subset(val0,sorted(rng.choice(len(val0),z.limit_val,False))),batch_size=2,collate_fn=coll);p2=DeepRegionEncoder(cfg,len(cfg['data']['class_map']),64,256).to(d);p2.load_state_dict(torch.load(P2ROOT/'checkpoint_epoch_028.pth',map_location=d,weights_only=False)['model']);p2.eval().requires_grad_(False);idx=json.loads((z.contract_dir/'index.json').read_text());ctr=cached(take(idx,'train',z.crc_train),z.contract_dir,d,'panel1');cva=cached(take(idx,'val',z.crc_val),z.contract_dir,d,'panel1');base=torch.load(P3,map_location=d,weights_only=False);results=[]
 for arm in ('seg_only','seg_plus_panel1'):
  torch.manual_seed(z.seed);cell=CellToRegionAttention(256,74).to(d);cell.load_state_dict(base['cell']);head=torch.nn.Linear(256,12).to(d);head.load_state_dict(base['head']);evidence=ObservedRegionEvidenceHeads().to(d);opt=torch.optim.AdamW([*cell.parameters(),*head.parameters(),*evidence.parameters()],lr=1e-4);losses=[]
  for step,b in zip(range(z.steps),cycle(tl)):
   opt.zero_grad()
   with torch.no_grad():q=p2(b['image'].to(d),False,True)
   m=F.grid_sample(q['assignment_low'],(b['cells'].to(d)[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);f=cell(q['region_tokens'],m,b['cells'].to(d),b['cell_valid'].to(d),b['total_cell_count'].to(d))['fused_tokens'];seg=F.cross_entropy(head(f).flatten(0,1),region_labels(q['assignment_low'],b['mask'].to(d),12,255).flatten(),ignore_index=255);aux=seg*0
   if arm=='seg_plus_panel1':aux,_=aux_loss(cell,evidence,ctr[step%len(ctr)],'panel1')
   (seg+z.panel_weight*aux).backward();opt.step();losses.append((float(seg.detach()),float(aux.detach())))
  cell.eval();head.eval();correct=total=0
  with torch.no_grad():
   for b in vl:
    q=p2(b['image'].to(d),False,True);m=F.grid_sample(q['assignment_low'],(b['cells'].to(d)[...,:2]*2-1).unsqueeze(1),align_corners=False).squeeze(2);pred=head(cell(q['region_tokens'],m,b['cells'].to(d),b['cell_valid'].to(d),b['total_cell_count'].to(d))['fused_tokens']).argmax(-1);label=region_labels(q['assignment_low'],b['mask'].to(d),12,255);keep=label!=255;correct+=((pred==label)&keep).sum().item();total+=keep.sum().item()
  av,n=evaluate_aux(cell,evidence,cva,'panel1');results.append({'arm':arm,'mean_seg_loss':float(np.mean([x[0] for x in losses])),'mean_panel1_loss':float(np.mean([x[1] for x in losses])),'val_region_accuracy':correct/total,'val_panel1_loss':av,'val_panel1_tokens':n})
 (out/'metrics.json').write_text(json.dumps({'status':'complete','scope':'joint diagnostic smoke, not FewClick click evaluation','steps':z.steps,'panel_weight':z.panel_weight,'results':results},indent=2));print(json.dumps(results))
if __name__=='__main__':main()
