"""Measure support fit, query precision/recall and sign sensitivity, without fitting."""
import argparse,json,os,time,gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from benchmarks.v5.size_aware_study.model import DecoderData,DecoderModel
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.architecture_bridge.train_monitor import probability_diagnostics

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--accept',action='store_true');a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for path,h in c['sources'].items():assert sha(path)==h,path
 for path,h in c['inputs'].items():assert sha(path)==h,path
 plans=json.load(open(r/'plans.json'));assert len(plans)==24;data=DecoderData(Path(c['cache']),'control',20261103)
 def measure(prob,identity,target):
  gt,mpp=data.labels(identity);mask=prob>=.5;truth=gt==target;valid=gt!=255;metrics=score(mask,truth,valid,mpp);tp=int((mask&truth&valid).sum());fp=int((mask&~truth&valid).sum());fn=int((~mask&truth&valid).sum());return metrics,dict(TP=tp,FP=fp,FN=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,GT_fraction=float(truth[valid].mean()))
 if a.accept:
  rows=json.load(open(r/'rows.json'));assert len(rows)==48
  with np.load(r/'masks.npz') as f:
   for row,q,s in zip(rows,f['query'],f['support']):
    plan=row['plan']
    for packed,identity,key in [(q,plan['query_id'],'query'),(s,plan['support_id'],'support')]:
     mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);metrics,err=measure(mask,identity,plan['target']);assert metrics==row[key] and err==row[key+'_errors']
  groups=defaultdict(list)
  for row in rows:groups[row['endpoint']+'|'+row['plan']['dataset']+'|'+row['plan']['kind']].append(row)
  summary={}
  for name,group in groups.items():
   assert len(group)==4
   def avg(v):v=[x for x in v if x is not None];return sum(v)/len(v) if v else None
   summary[name]=dict(n=4,query_Dice=avg([x['query']['dice'] for x in group if x['query']['positive']>0]),support_Dice=avg([x['support']['dice'] for x in group]),absent_FPA=avg([x['query']['false_positive_fraction'] for x in group if x['query']['positive']==0]),query_precision=avg([x['query_errors']['precision'] for x in group if x['query']['positive']>0]),query_recall=avg([x['query_errors']['recall'] for x in group if x['query']['positive']>0]),swap_mean=avg([x['prompt_response']['swap_valid_mean'] for x in group]))
  (r/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted_readonly_diagnostic',summary=summary,rows_sha256=sha(r/'rows.json'),masks_sha256=sha(r/'masks.npz'),contract_sha256=sha(r/'contract.json'),limitation='4 fixed train episodes per group, one initialization seed; diagnostic only, no new method or causal proof'),indent=2)+'\n');print('accepted48rows96masks');return
 started=time.monotonic();rows=[];qmask=[];smask=[]
 def status(**kw):
  v=dict(worker_pid=os.getpid(),seconds=time.monotonic()-started,**kw);(r/'status.json').write_text(json.dumps(v));print(json.dumps(v),flush=True)
 for endpoint,path in c['checkpoints'].items():
  status(stage='loading',endpoint=endpoint);model=DecoderModel('control',20261103,'cuda:0');state=torch.load(path,map_location='cuda:0',weights_only=True);assert state['updates']==int(endpoint);model.load_state_dict(state['model']);del state;model.eval();before=tree_sha(model.state_dict())
  with torch.no_grad():
   for i,plan in enumerate(plans):
    inputs=data.model_inputs(plan,'xcell','cuda:0');out=model(**inputs);prompts=dict(inputs['prompts'])
    for key in ['xy','mask']:prompts['positive_'+key],prompts['negative_'+key]=prompts['negative_'+key],prompts['positive_'+key]
    other=model(**dict(inputs,prompts=prompts))['query']['pixel_probability'][0];prob=out['query']['pixel_probability'][0];support=out['support']['pixel_probability'][0];gt,_=data.labels(plan['query_id']);diag=probability_diagnostics(prob,torch.from_numpy(gt==plan['target']).cuda(),torch.from_numpy(gt!=255).cuda(),other)
    qm,qe=measure(prob.cpu().numpy(),plan['query_id'],plan['target']);sm,se=measure(support.cpu().numpy(),plan['support_id'],plan['target']);rows.append(dict(endpoint=endpoint,plan=plan,query=qm,query_errors=qe,support=sm,support_errors=se,prompt_response=diag));qmask.append(np.packbits((prob>=.5).cpu().numpy()));smask.append(np.packbits((support>=.5).cpu().numpy()));status(stage='diagnosing',endpoint=endpoint,tasks=i+1,total=24)
  assert before==tree_sha(model.state_dict());del model;gc.collect();torch.cuda.empty_cache()
 (r/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');np.savez_compressed(r/'masks.npz',query=np.stack(qmask),support=np.stack(smask));status(stage='complete_unaccepted',rows=48)
if __name__=='__main__':main()
