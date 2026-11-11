import argparse,json,os,time
from functools import lru_cache
from pathlib import Path
import numpy as np,torch
from PIL import Image
def encode(model,x,method):
 if method=='sansa':return model.encode(x)
 return model.encode(model.base.transform(x))
def decode(model,se,qe,prompts,method):
 return model.cached_decode(se,qe,prompts,False,with_support=False)[0]['pixel_probability']
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));assert sha(r/'tasks.json')==c['tasks_sha256']
 for path,h in c['sources'].items():assert sha(path)==h,path
 ck=c['checkpoints'][a.method];assert sha(ck['path'])==ck['sha256'];alltasks=json.load(open(r/'tasks.json'));tasks=[alltasks[i] for i in c.get('gate_indices',[0,700,1000,1700,2000,2700])] if a.gate else [x for x in alltasks if x['index']%2==a.shard]
 out=r/(f'gate_{a.method}' if a.gate else f'{a.method}_{a.shard}');out.mkdir(exist_ok=False);dump(out/'worker.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],status='loading',gate=a.gate))
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=DecoderData(Path(c['cache']),'match');assert sha(r/'query_rows.json')==c['query_rows_sha256'];data.rows.update(json.load(open(r/'query_rows.json')))
 from .model import build
 model=build(a.method,ck['path']);model.eval();hashes={}
 @lru_cache(maxsize=32)
 def inputs(identity):
  x={k:v.to('cuda:0') for k,v in data.inputs(data.rows[identity].get('template_id',identity),'xcell').items()};fingerprints=[]
  for scale in range(3):
   rgb=np.asarray(Image.fromarray(native_context(data.rows[identity],scale)).resize((512,512),Image.Resampling.BILINEAR)).copy();fingerprints.append(__import__('hashlib').sha256(rgb.tobytes()).hexdigest());x['image'+('' if scale==0 else '_'+str(scale))]=he_to_tensor(rgb)[None].to('cuda:0')
  hashes[identity]=fingerprints;return x
 @lru_cache(maxsize=16)
 def encoded(identity):return encode(model,inputs(identity),a.method)
 result=[];masks=[];start=time.monotonic()
 with torch.no_grad():
  for task in tasks:
   assert task['support_space']==task['query_space']
   from benchmarks.v5.paper_multiscale.data import intersects
   assert not intersects(json.loads(data.rows[task['support_id']]['multiscale_json'])['context_um'],json.loads(data.rows[task['query_id']]['multiscale_json'])['context_um'])
   prompts={}
   for sign in ['positive','negative']:
    prompts[sign+'_xy']=torch.tensor([task[sign+'_xy']],device='cuda:0',dtype=torch.float32);prompts[sign+'_mask']=torch.ones(1,len(task[sign+'_xy']),device='cuda:0',dtype=torch.bool)
   prompts['prompt_size_id']=torch.ones(1,device='cuda:0',dtype=torch.long)
   se=encoded(task['support_id']);qe=encoded(task['query_id']);prob=decode(model,se,qe,prompts,a.method);assert torch.isfinite(prob).all()
   if a.gate or len(result)<2:
    full=model(support=inputs(task['support_id']),query=inputs(task['query_id']),prompts=prompts,local=False,shared=False,with_support=False)['query']['pixel_probability'];assert torch.equal(full,prob)
   mask=prob[0].cpu().numpy()>=.5;gt,mpp=data.labels(task['query_id']);metrics=score(mask,gt==task['target'],gt!=255,mpp);assert metrics['positive']==task['query_positive_pixels'];result.append(dict(index=task['index'],metrics=metrics));masks.append(np.packbits(mask))
   if len(result)%25==0 or len(result)==1:dump(out/'progress.json',dict(pid=os.getpid(),rows=len(result),total=len(tasks),seconds=time.monotonic()-start));print(json.dumps(dict(pid=os.getpid(),rows=len(result),total=len(tasks))),flush=True)
 dump(out/'rows.json',result);np.save(out/'masks.npy',np.stack(masks));dump(out/'input_hashes.json',hashes)
 # Independent disk masks replay through GT loader and metric, with task identities bound.
 stored=np.load(out/'masks.npy',mmap_mode='r');assert len(stored)==len(tasks)
 for row,task,packed in zip(json.load(open(out/'rows.json')),tasks,stored):
  assert row['index']==task['index'];gt,mpp=data.labels(task['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);assert score(mask,gt==task['target'],gt!=255,mpp)==row['metrics']
 dump(out/'acceptance.json',dict(status='complete_accepted',rows=len(tasks),tasks_sha256=c['tasks_sha256'],checkpoint_sha256=ck['sha256'],rows_sha256=sha(out/'rows.json'),masks_sha256=sha(out/'masks.npy'),input_sha256=sha(out/'input_hashes.json'),cached_full_exact_count=len(tasks) if a.gate else 2,seconds=time.monotonic()-start))
if __name__=='__main__':main()
