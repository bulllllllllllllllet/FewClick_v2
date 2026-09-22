#!/usr/bin/env python3
"""Panel-specific observed-composition training on frozen CRC contracts."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss

def load(path,device):
 x=np.load(path);return {k:torch.from_numpy(x[k]).to(device) for k in x.files}
def loss(model,head,x,panel):
 cells=x['cells'].unsqueeze(0);valid=torch.ones((1,len(cells[0])),dtype=torch.bool,device=cells.device);f=model(x['region_tokens'].unsqueeze(0),x['assignment_at_cells'].unsqueeze(0),cells,valid,x['total_cell_count'].reshape(1))['fused_tokens'][0];head_key={'panel1':'structure_observed_logits','panel2':'immune_observed_logits'}[panel];log=head(f.unsqueeze(0))[head_key][0];return observed_composition_loss(log,x[f'{panel}_counts'],x[f'{panel}_observed_cells'],x[f'{panel}_coverage'])
def observed_target(x,panel):
 counts=x[f'{panel}_counts'].float();coverage=x[f'{panel}_coverage'].float();observed=x[f'{panel}_observed_cells']
 valid=(observed>=1)&(counts.sum(-1)>0)&(coverage>0)
 return counts[valid]/counts[valid].sum(-1,keepdim=True),coverage[valid]
def train_prior(items,panel):
 # Training samples a patch uniformly; within a patch its token CE is coverage weighted.
 means=[]
 for x in items:
  target,weight=observed_target(x,panel)
  if len(target):means.append((target*weight[:,None]).sum(0)/weight.sum())
 if not means:raise RuntimeError(f'no valid {panel} training tokens')
 return torch.stack(means).mean(0)
def evaluate_prior(prior,items,panel):
 total=count=0
 for x in items:
  target,weight=observed_target(x,panel)
  if not len(target):continue
  ce=-(target*prior.log()).sum(-1)
  total+=float((ce*weight).sum()/weight.sum())*len(target);count+=len(target)
 if not count:raise RuntimeError(f'no valid {panel} evaluation tokens')
 return total/count,count
def evaluate(model,head,items,panel):
 model.eval();head.eval();total=count=0
 with torch.no_grad():
  for x in items:
   v,n=loss(model,head,x,panel);total+=float(v)*n;count+=n
 if not count:raise RuntimeError(f'no valid {panel} validation tokens')
 return total/count,count
def main():
 p=argparse.ArgumentParser();p.add_argument('--contract-dir',type=Path,required=True);p.add_argument('--timestamp',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--panel',choices=('panel1','panel2'),default='panel2');p.add_argument('--steps',type=int,default=1000);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--device',default='cuda:0');p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));a=p.parse_args();out=a.output_root/f'crc_{a.panel}_cached_train_{a.timestamp}';out.mkdir(parents=True,exist_ok=False)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);d=torch.device(a.device);idx=json.loads((a.contract_dir/'index.json').read_text());base=a.contract_dir;train=[load(base/x['contract_path'],d) for x in idx if x['split']=='train'];val=[load(base/x['contract_path'],d) for x in idx if x['split']=='val'];test=[load(base/x['contract_path'],d) for x in idx if x['split']=='test'];model=CellToRegionAttention(256,74).to(d);head=ObservedRegionEvidenceHeads().to(d);opt=torch.optim.AdamW([*model.parameters(),*head.parameters()],lr=a.lr);rng=random.Random(a.seed);prior=train_prior(train,a.panel);prior_val_ce,prior_val_tokens=evaluate_prior(prior,val,a.panel);initial_val_ce,initial_val_tokens=evaluate(model,head,val,a.panel);history=[{'step':0,'train_ce':None,'train_tokens':0,'val_ce':initial_val_ce,'val_tokens':initial_val_tokens}];best=initial_val_ce;best_step=0;torch.save({'model':model.state_dict(),'head':head.state_dict(),'step':0},out/'best.pth')
 for step in range(1,a.steps+1):
  model.train();head.train();opt.zero_grad();v,n=loss(model,head,rng.choice(train),a.panel);v.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*head.parameters()],5.,error_if_nonfinite=True);opt.step()
  if step%a.eval_every==0 or step==a.steps:
   val_ce,val_tokens=evaluate(model,head,val,a.panel);history.append({'step':step,'train_ce':float(v.detach()),'train_tokens':n,'val_ce':val_ce,'val_tokens':val_tokens})
   if val_ce<best:best=val_ce;best_step=step;torch.save({'model':model.state_dict(),'head':head.state_dict(),'step':step},out/'best.pth')
 state=torch.load(out/'best.pth',map_location=d,weights_only=False);model.load_state_dict(state['model']);head.load_state_dict(state['head']);test_ce,test_tokens=evaluate(model,head,test,a.panel);prior_test_ce,prior_test_tokens=evaluate_prior(prior,test,a.panel);(out/'metrics.json').write_text(json.dumps({'status':'complete','scope':f'cached {a.panel} auxiliary representation training; not segmentation evaluation','seed':a.seed,'panel':a.panel,'steps':a.steps,'best_step':best_step,'initial_val_ce':initial_val_ce,'best_val_ce':best,'train_prior':prior.detach().cpu().tolist(),'train_prior_val_ce':prior_val_ce,'train_prior_val_tokens':prior_val_tokens,'test_ce':test_ce,'test_tokens':test_tokens,'train_prior_test_ce':prior_test_ce,'train_prior_test_tokens':prior_test_tokens,'history':history,'contract_dir':str(a.contract_dir)},indent=2));print(json.dumps({'output':str(out),'initial_val_ce':initial_val_ce,'best_val_ce':best,'train_prior_test_ce':prior_test_ce,'test_ce':test_ce}))
if __name__=='__main__':main()
