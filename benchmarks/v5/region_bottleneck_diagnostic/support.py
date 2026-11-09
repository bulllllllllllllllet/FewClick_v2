"""Support-only oracle and point-to-region mixing; diagnostic GT use only."""
import argparse,json,os,time
from pathlib import Path
from functools import lru_cache
import torch,numpy as np
from torch.nn import functional as F
from benchmarks.v5.region_bottleneck_diagnostic.run import oracle
from benchmarks.v5.paper_closeout.run import load,inputs,prompts
from benchmarks.v5.wsi_quality_study.worker import encode,decode
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.fair_comparison.metrics import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();r=a.root;out=r/f'shard{a.shard}';out.mkdir();d,m=load(out);m.requires_grad_(False);tasks=json.load(open(r/'plans.json'));old=json.load(open(r/'contract.json'))['previous'];prior={x['index']:x for s in [0,1] for x in json.load(open(Path(old)/f'shard{s}/rows.json'))};rows=[];start=time.monotonic();(out/'masks').mkdir()
 for i,t in enumerate(tasks):
  if i%2!=a.shard:continue
  with torch.no_grad():
   se=encode(m,inputs(d,t['support_id']),'full');p=prompts(t);prob=decode(m,se,se,p,'full');assignment=se['assignment_low'].float();weights={}
   for sign in ['positive','negative']:
    w=sample_assignment_at_cells(assignment,p[sign+'_xy']);weights[sign]=(w/w.sum(1,keepdim=True).clamp_min(1e-8))[0].T
   pos,neg=weights['positive'],weights['negative'];cos=F.normalize(pos,dim=1)@F.normalize(neg,dim=1).T;tok=se['tokens'][0][0];pt=F.normalize(pos@tok,dim=1);nt=F.normalize(neg@tok,dim=1);token_cos=pt@nt.T
  gt,mpp=d.labels(t['support_id']);truth=gt==t['target'];valid=gt!=255;metric=score(prob[0].cpu().numpy()>=.5,truth,valid,mpp);assert metric==prior[t['index']]['support_metrics']
  with torch.no_grad():
   full=F.interpolate(assignment,size=(512,512),mode='bilinear',align_corners=False)[0];full=full/full.sum(0,keepdim=True).clamp_min(1e-6);v=torch.tensor(valid,device='cuda:0');y=torch.tensor(truth,device='cuda:0');purity=(full[:,v]*y[v]).sum(1)/full[:,v].sum(1).clamp_min(1e-6);pp=(pos@purity).cpu().tolist();npurity=(neg@purity).cpu().tolist();area=float(truth[valid].mean())
   # Confirm allowed simulated points really hit the requested GT signs.
   for sign,expected in [('positive',True),('negative',False)]:
    for x,yy in t[sign+'_xy']:
     xi,yi=int(x*512),int(yy*512);assert valid[yi,xi] and bool(truth[yi,xi])==expected
  hard,soft=oracle(assignment,truth,valid);path=out/'masks'/f'{t["index"]}.npz';np.savez_compressed(path,support=prob[0].cpu().numpy()>=.5,hard_oracle=hard,soft_oracle=soft)
  with np.load(path) as masks:
   assert score(masks['support'],truth,valid,mpp)==metric;hm=score(masks['hard_oracle'],truth,valid,mpp);sm=score(masks['soft_oracle'],truth,valid,mpp)
  rows.append(dict(index=t['index'],query_stratum=t['diagnostic_stratum'],support_target_fraction=area,support_Dice=metric['dice'],hard_oracle_Dice=hm['dice'],soft_attained_Dice=sm['dice'],positive_region_purity=pp,negative_region_target_fraction=npurity,positive_purity_mean=float(np.mean(pp)),negative_target_mean=float(np.mean(npurity)),purity_separation=float(np.mean(pp)-np.mean(npurity)),positive_enrichment=float(np.mean(pp)/max(area,1e-8)),region_assignment_cross_cosine_mean=float(cos.mean()),token_cross_cosine_mean=float(token_cos.mean()),same_argmax_region_pair_fraction=float((pos.argmax(1)[:,None]==neg.argmax(1)[None,:]).float().mean())))
  dump(out/'rows.json',rows);dump(out/'progress.json',dict(pid=os.getpid(),rows=len(rows),total=sum(j%2==a.shard for j in range(len(tasks))),seconds=time.monotonic()-start))
 dump(out/'acceptance.json',dict(status='complete_accepted_diagnostic',rows=len(rows),all_point_GT_signs_correct=True,prior_support_metrics_exact=True,oracle_only=True))
if __name__=='__main__':main()
