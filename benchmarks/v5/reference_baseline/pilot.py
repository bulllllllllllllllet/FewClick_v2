"""Fixed published SANSA Universal same-slide sparse-support inference pilot."""
import argparse,copy,hashlib,json,os,sys,time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.paper_multiscale.data import native_context,intersects
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def choose(tasks):
 selected=[]
 for target in sorted({t['target'] for t in tasks}):
  pool=sorted([t for t in tasks if t['target']==target],key=lambda t:hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest());chosen=[];seen=set()
  for group,n in [('large',4),('other_present',3),('absent',3)]:
   candidates=[t for t in pool if ('large' if t['query_positive_pixels']>=.2*512*512 else 'other_present' if t['query_positive_pixels'] else 'absent')==group];count=0
   for t in candidates:
    if t['query_id'] in seen:continue
    chosen.append(t);seen.add(t['query_id']);count+=1
    if count==n:break
  for t in pool:
   if len(chosen)==10:break
   if t['query_id'] not in seen:chosen.append(t);seen.add(t['query_id'])
  assert len(chosen)==10;selected.extend(chosen)
 assert len(selected)==110;return selected

def summarize(rows,tasks):
 by=defaultdict(list);large=defaultdict(list);absent=defaultdict(list)
 for r,t in zip(rows,tasks):
  m=r['metrics'];assert r['index']==t['index']
  if m['positive']:
   by[t['target']].append(m['dice'])
   if m['positive']/m['valid']>=.2:large[t['target']].append(m['dice'])
  else:absent[t['target']].append(m['false_positive_fraction'])
 macro=lambda d:float(np.mean([np.mean(v) for v in d.values()])) if d else None
 return dict(present_class_macro_Dice=macro(by),large20_class_macro_Dice=macro(large),absent_class_macro_FPA=macro(absent),present_n=sum(map(len,by.values())),large_n=sum(map(len,large.values())),absent_n=sum(map(len,absent.values())),classes={k:dict(present_Dice=float(np.mean(v)),n=len(v)) for k,v in by.items()})

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));w=json.load(open(r/'weights.json'));src=Path(c['source_tasks_root']);assert sha(src/'tasks.json')==c['tasks_sha256'];assert sha(src/'query_rows.json')==c['query_rows_sha256']
 for path,h in c['sources'].items():assert sha(path)==h,path
 for v in w['weights'].values():assert sha(v['path'])==v['sha256']
 import subprocess
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=r/'SANSA',text=True).strip()==w['source_commit']
 torch.set_num_threads(2);torch.manual_seed(20261103);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sys.path[:0]=[str(r/'deps'),str(r/'SANSA')]
 from models.sansa.sansa import build_sansa
 from util.path_utils import SAM2_PATHS_CONFIG
 # Point to the downloaded official SAM2.0 file; do not substitute SAM2.1 weights.
 SAM2_PATHS_CONFIG['large']=(w['weights']['sam2_hiera_large.pt']['path'],SAM2_PATHS_CONFIG['large'][1])
 model=build_sansa(channel_factor=.8,device='cuda').cuda().eval();state=torch.load(w['weights']['sansa_universal.pth']['path'],map_location='cpu',weights_only=False)['model'];missing,unexpected=model.load_state_dict(state,strict=False)
 assert not unexpected,unexpected;assert not any('adapter' in k for k in missing),missing
 assert state and all('adapter' in k for k in state),'unexpected Universal checkpoint schema'
 dump(r/'weight_load_acceptance.json',dict(adapter_keys=len(state),missing_nonadapter=len(missing),unexpected=unexpected,all_adapters_loaded=True,base_source=w['weights']['sam2_hiera_large.pt']))
 before=tree_sha(model.state_dict());data=DecoderData(Path(c['cache']),'match');data.rows.update(json.load(open(src/'query_rows.json')));tasks=json.load(open(r/'plans.json'));assert sha(r/'plans.json')==c['plans_sha256']
 @lru_cache(maxsize=24)
 def image(identity):
  array=np.asarray(Image.fromarray(native_context(data.rows[identity],0)).resize((512,512),Image.Resampling.BILINEAR)).copy();return array
 def predict(task):
  support=image(task['support_id']);query=image(task['query_id']);frames=[]
  for raw in [support,query]:
   resized=np.asarray(Image.fromarray(raw).resize((640,640),Image.Resampling.BILINEAR)).copy();x=torch.from_numpy(resized).permute(2,0,1).float().cuda()/255;frames.append((x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None])
  xy=torch.tensor([task['positive_xy']+task['negative_xy']],device='cuda',dtype=torch.float32)*640;labels=torch.tensor([[1]*len(task['positive_xy'])+[0]*len(task['negative_xy'])],device='cuda',dtype=torch.int32)
  prompt={0:{0:dict(prompt_type='point',prompt=dict(point_coords=xy,point_labels=labels))},'shots':1}
  # Only images and original support point coordinates/labels enter inference.
  output=model(torch.stack(frames)[None],prompt)['pred_masks'];assert output.shape==(2,640,640),output.shape
  logits=F.interpolate(output[1][None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0];assert torch.isfinite(logits).all();return logits
 rows=[];masks=[];fingerprints={};start=time.monotonic()
 with torch.inference_mode():
  for j,t in enumerate(tasks):
   assert t['support_space']==t['query_space'];assert t['support_id']!=t['query_id'];assert not intersects(json.loads(data.rows[t['support_id']]['multiscale_json'])['context_um'],json.loads(data.rows[t['query_id']]['multiscale_json'])['context_um'])
   logits=predict(t)
   if j<2:assert torch.equal(logits,predict(t)),'fresh prompt/memory replay differs'
   pred=(logits>=0).cpu().numpy();gt,mpp=data.labels(t['query_id']);metrics=score(pred,gt==t['target'],gt!=255,mpp);assert metrics['positive']==t['query_positive_pixels'];rows.append(dict(index=t['index'],metrics=metrics));masks.append(np.packbits(pred))
   for identity in [t['support_id'],t['query_id']]:fingerprints[identity]=hashlib.sha256(image(identity).tobytes()).hexdigest()
   if j%10==0:dump(r/'progress.json',dict(status='running',pid=os.getpid(),rows=j+1,total=110,seconds=time.monotonic()-start));print(json.dumps(dict(rows=j+1,total=110)),flush=True)
 assert tree_sha(model.state_dict())==before;np.save(r/'masks.npy',np.stack(masks));dump(r/'rows.json',rows);dump(r/'input_hashes.json',fingerprints)
 for row,t,packed in zip(json.load(open(r/'rows.json')),tasks,np.load(r/'masks.npy',mmap_mode='r')):
  gt,mpp=data.labels(t['query_id']);pred=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);assert score(pred,gt==t['target'],gt!=255,mpp)==row['metrics']
 comparisons={'SANSA_Universal_public':summarize(rows,tasks)}
 for name,base in c['reference_rows'].items():
  indexed={}
  for path in base:
   for item in json.load(open(path)):assert item['index'] not in indexed;indexed[item['index']]=item
  comparisons[name]=summarize([indexed[t['index']] for t in tasks],tasks)
 dump(r/'report.json',dict(status='complete_accepted_readonly_pilot',comparisons=comparisons,target_updates=0,prompts='original support3positive2negative; zero query',limitations='Stratified110 development tasks, not full population or independent generalization. Universal promptable adapter differs from mask-only paper benchmark. Public pretraining and target update budgets differ from adapted baselines. All strata retained.'))
 dump(r/'acceptance.json',dict(status='complete_accepted',rows=110,worker_pid=os.getpid(),weight_state_unchanged=True,first2_repeat_exact=True,plans_sha256=sha(r/'plans.json'),report_sha256=sha(r/'report.json'),rows_sha256=sha(r/'rows.json'),masks_sha256=sha(r/'masks.npy'),seconds=time.monotonic()-start))
if __name__=='__main__':main()
