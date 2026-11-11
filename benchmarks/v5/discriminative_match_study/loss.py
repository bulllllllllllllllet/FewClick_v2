"""Train-only binary metric loss; same episode labels, no extra tasks or clicks."""
import torch
from torch.nn import functional as F
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss as original_loss
WEIGHT=.1
MARGIN=.3
TEMPERATURE=.1
HARD_NEGATIVES=64

def matching_loss(output,data,plan):
 assert plan['split']=='train'
 if 'matching_aux_support' not in output:return output['query']['pixel_logits'].new_zeros(())
 support=output['matching_aux_support'];query=output['matching_aux_query'];device=query.device
 labels=[]
 for side,feat in [('support',support),('query',query)]:
  assert data.rows[plan[side+'_id']]['split']=='train'
  gt,_=data.labels(plan[side+'_id']);target=torch.as_tensor(gt==plan['target'],device=device,dtype=torch.float32)[None,None];valid=torch.as_tensor(gt!=255,device=device,dtype=torch.float32)[None,None]
  labels.append((F.interpolate(target,size=feat.shape[-2:],mode='area'),F.interpolate(valid,size=feat.shape[-2:],mode='area')))
 (st,sv),(qt,qv)=labels;assert st.sum()>0
 prototype=F.normalize((F.normalize(support,dim=1)*st).sum((2,3))/st.sum((2,3)).clamp_min(1e-8),dim=1)
 similarity=(F.normalize(query,dim=1)*prototype[:,:,None,None]).sum(1)
 logits=(similarity-MARGIN)/TEMPERATURE
 pos=-(F.logsigmoid(logits)*qt[:,0]).sum()/qt.sum().clamp_min(1e-8)
 # Hard negatives are pure valid nontarget cells, ranked only within this train episode.
 valid_negative=(qt[:,0]==0)&(qv[:,0]>=1-1e-6)
 ranking=F.interpolate(output['query']['pixel_probability'].detach()[:,None],size=query.shape[-2:],mode='area')[:,0]
 candidates=valid_negative.flatten().nonzero().flatten();count=min(HARD_NEGATIVES,len(candidates))
 if count:
  order=torch.argsort(ranking.flatten()[candidates],descending=True,stable=True)[:count];selected=logits.flatten()[candidates[order]];neg=-F.logsigmoid(-selected).mean()
 else:neg=logits.sum()*0
 return (pos+neg)/(int(qt.sum()>0)+int(count>0))

def episode_loss(output,data,plan,prompts):
 return original_loss(output,data,plan,prompts)+WEIGHT*matching_loss(output,data,plan)
