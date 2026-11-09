"""GT-only diagnostic oracle; never exported for model selection/inference."""
import argparse,json,time,os,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from benchmarks.v5.paper_closeout.run import load,inputs,prompts,SOURCE
from benchmarks.v5.wsi_quality_study.worker import encode,decode
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.fair_comparison.metrics import score

def oracle(assignment,gt,valid):
 # Exact best union of hard argmax regions: diagnostic for this hard partition,
 # NOT an upper bound on the actual soft projection or residual matcher.
 a=F.interpolate(assignment,size=(512,512),mode='bilinear',align_corners=False)[0].detach();a=a/a.sum(0,keepdim=True).clamp_min(1e-6);k=a.shape[0];ids=a.argmax(0);y=torch.tensor(gt,device=a.device,dtype=torch.float32);v=torch.tensor(valid,device=a.device,dtype=torch.bool)
 iv=ids[v].cpu().numpy();yv_np=y[v].cpu().numpy();mass=torch.tensor(np.bincount(iv,minlength=k),device=a.device,dtype=torch.float32);pos=torch.tensor(np.bincount(iv,weights=yv_np,minlength=k),device=a.device,dtype=torch.float32);order=(pos/mass.clamp_min(1)).argsort(descending=True);tp=pos[order].cumsum(0);size=mass[order].cumsum(0);dice=2*tp/(size+y[v].sum()).clamp_min(1);n=int(dice.argmax())+1;chosen=torch.zeros(k,device=a.device,dtype=torch.bool);chosen[order[:n]]=True;hard=chosen[ids].cpu().numpy()
 # Attainable soft-region oracle: optimize only K temporary logits against GT.
 # Frozen model/assignment; no model update, no threshold tuning; not certified optimum.
 av=a[:,v].T.contiguous();yv=y[v];p0=(a[:,v]*yv).sum(1)/a[:,v].sum(1).clamp_min(1e-6);best=-1;bestmask=None
 for init in [torch.zeros_like(p0),torch.logit(p0.clamp(.001,.999))]:
  z=init.detach().requires_grad_(True);opt=torch.optim.Adam([z],lr=.2)
  for step in range(81):
   prob=(av@z.sigmoid()).clamp(1e-6,1-1e-6);mask=prob>=.5;value=float(2*(mask*yv).sum()/(mask.sum()+yv.sum()).clamp_min(1))
   if value>best:best=value;bestmask=(torch.einsum('khw,k->hw',a,z.detach().sigmoid())>=.5).cpu().numpy()
   if step==80:break
   loss=F.binary_cross_entropy(prob,yv)+1-(2*(prob*yv).sum()+1)/(prob.sum()+yv.sum()+1);opt.zero_grad();loss.backward();opt.step()
 return hard,bestmask

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();r=a.root;out=r/f'shard{a.shard}';out.mkdir(exist_ok=False);d,m=load(out);m.requires_grad_(False);plans=json.load(open(r/'plans.json'));plans=[t for i,t in enumerate(plans) if i%2==a.shard];rows=[];start=time.monotonic();(out/'masks').mkdir()
 @lru_cache(maxsize=12)
 def enc(identity):
  with torch.no_grad():return encode(m,inputs(d,identity),'full')
 reference={x['index']:x['metrics'] for s in [0,1] for x in json.load(open(SOURCE/f'full_{s}/rows.json'))}
 for i,t in enumerate(plans):
  with torch.no_grad():
   se=enc(t['support_id']);qe=enc(t['query_id']);p=prompts(t);task=m.base.core.support_task(se,p);sbase=m.base.core.decode(se,task,p,True)[0];qbase=m.base.core.decode(qe,task,p,False)[0];sp=decode(m,se,se,p,'full');qp=decode(m,se,qe,p,'full')
   sx=inputs(d,t['support_id']);qx=inputs(d,t['query_id'])
   if i<2:assert torch.equal(m(support=sx,query=qx,prompts=p,local=False,shared=False,with_support=True)['query']['pixel_probability'],qp)
  sg,smpp=d.labels(t['support_id']);qg,qmpp=d.labels(t['query_id']);valid=qg!=255;truth=qg==t['target'];mask=qp[0].cpu().numpy()>=.5;metrics=score(mask,truth,valid,qmpp);assert metrics==reference[t['index']],('historical replay differs',t['index'])
  hard,soft=oracle(qe['assignment_low'],truth,valid)
  arrays=dict(support=sp[0].cpu().numpy()>=.5,query=mask,hard_oracle=hard,soft_oracle=soft);path=out/'masks'/f'{t["index"]}.npz';np.savez_compressed(path,**arrays)
  with np.load(path) as saved:
   assert score(saved['query'],truth,valid,qmpp)==metrics
   row=dict(index=t['index'],stratum=t['diagnostic_stratum'],target=t['target'],support_metrics=score(saved['support'],sg==t['target'],sg!=255,smpp),support_region_only=score(sbase['pixel_probability'][0].cpu().numpy()>=.5,sg==t['target'],sg!=255,smpp),query_metrics=metrics,query_region_only=score(qbase['pixel_probability'][0].cpu().numpy()>=.5,truth,valid,qmpp),hard_partition_oracle=score(saved['hard_oracle'],truth,valid,qmpp),attained_soft_oracle=score(saved['soft_oracle'],truth,valid,qmpp),query_probability_max=float(qp.max()),query_probability_mean=float(qp.mean()))
  rows.append(row);dump(out/'rows.json',rows);dump(out/'progress.json',dict(pid=os.getpid(),rows=len(rows),total=len(plans),seconds=time.monotonic()-start))
 dump(out/'acceptance.json',dict(status='complete_accepted_diagnostic',rows=len(rows),historical_query_masks_metrics_exact=True,GT_oracle_diagnostic_only=True,not_certified_soft_upper_bound=True))
if __name__=='__main__':main()
