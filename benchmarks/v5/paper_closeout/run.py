"""Frozen inference-only paper closeout; no model selection or training."""
import argparse,copy,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from benchmarks.v5.wsi_quality_study.prepare import dump,labels_for
from benchmarks.v5.wsi_quality_study.worker import encode,decode,timing
from benchmarks.v5.support_match_study.model import DecoderData,DecoderModel
from benchmarks.v5.paper_multiscale.data import native_context,intersects
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score
BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
SOURCE=BASE/'wsi_same_slide_1108_013913'
def load(r):
 c=json.load(open(SOURCE/'contract.json'));ck=c['checkpoints']['full'];assert sha(ck['path'])==ck['sha256']
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 d=DecoderData(Path(c['cache']),'match');d.rows.update(json.load(open(SOURCE/'query_rows.json')))
 m=DecoderModel('match');m.load_state_dict(torch.load(ck['path'],map_location='cuda:0',weights_only=True)['model']);m.eval()
 dump(r/'identity.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),checkpoint=ck,source_tasks_sha256=sha(SOURCE/'tasks.json'),code_sha256=sha(__file__)))
 return d,m

def inputs(d,identity):
 row=d.rows[identity];x={k:v.to('cuda:0') for k,v in d.inputs(row.get('template_id',identity),'xcell').items()}
 for s in range(3):
  rgb=np.asarray(Image.fromarray(native_context(row,s)).resize((512,512),Image.Resampling.BILINEAR)).copy();x['image'+('' if s==0 else '_'+str(s))]=he_to_tensor(rgb)[None].cuda()
 return x

def prompts(t,n=3):
 p={}
 for sign,count in [('positive',n),('negative',2)]:
  p[sign+'_xy']=torch.tensor([t[sign+'_xy'][:count]],device='cuda:0',dtype=torch.float32);p[sign+'_mask']=torch.ones(1,count,device='cuda:0',dtype=torch.bool)
 p['prompt_size_id']=torch.ones(1,device='cuda:0',dtype=torch.long);return p

def evaluate(r,d,m,tasks,budgets):
 rows=[];start=time.monotonic();last=None;qe=None
 (r/'masks').mkdir(exist_ok=True)
 for i,t in enumerate(tasks):
  sx=inputs(d,t['support_id']);se=encode(m,sx,'full')
  if last!=t['query_id']:qx=inputs(d,t['query_id']);qe=encode(m,qx,'full');last=t['query_id']
  gt,mpp=d.labels(t['query_id'])
  for n in budgets:
   p=prompts(t,n);prob=decode(m,se,qe,p,'full');assert torch.isfinite(prob).all()
   if i<2:
    fresh=m(support=sx,query=qx,prompts=p,local=False,shared=False,with_support=False)['query']['pixel_probability'];assert torch.equal(prob,fresh)
   mask=prob[0].cpu().numpy()>=.5;path=r/'masks'/f'{i:06}_{n}.npy';np.save(path,np.packbits(mask))
   metrics=score(mask,gt==t['target'],gt!=255,mpp)
   replay=np.unpackbits(np.load(path),count=512*512).reshape(512,512).astype(bool);assert score(replay,gt==t['target'],gt!=255,mpp)==metrics
   rows.append(dict(task=t,index=i,positive_points=n,metrics=metrics))
  if i%10==0:dump(r/'progress.json',dict(pid=os.getpid(),tasks=i+1,total=len(tasks),seconds=time.monotonic()-start))
 dump(r/'rows.json',rows);dump(r/'acceptance.json',dict(status='complete_accepted',tasks=len(tasks),masks=len(rows),rows_sha256=sha(r/'rows.json'),fresh_cached_exact_tasks=min(2,len(tasks)),independent_disk_mask_score=True))

def robustness(r,d,m):
 from benchmarks.v5.fair_comparison.protocol import resolve_clicks
 tasks=json.load(open(SOURCE/'tasks.json'));selected=[]
 # SHA-fixed 20 tasks/class; three point draws on each existing support image.
 for target in sorted({t['target'] for t in tasks}):
  pool=sorted([t for t in tasks if t['target']==target],key=lambda t:hashlib.sha256(str(t['index']).encode()).hexdigest())[:20]
  for t in pool:
   for draw in range(3):
    try:p=resolve_clicks(dict(dataset='crc',target=target,click_seed=20261109+draw),d.rows[t['support_id']],maximum=(5,2))
    except ValueError as e:selected.append(dict(excluded=True,source_index=t['index'],draw=draw,reason=str(e)));continue
    selected.append(dict(t,positive_xy=p['positive_xy'],negative_xy=p['negative_xy'],draw=draw))
 dump(r/'plans.json',selected);evaluate(r,d,m,[t for t in selected if not t.get('excluded')],[1,3,5])


def confirmation(r,d,m):
 import pandas as pd
 from benchmarks.v5.paper_multiscale.data import CATALOG,SOURCES,context_geometry
 from benchmarks.v5.fair_comparison.protocol import resolve_clicks
 audit=json.load(open(BASE/'paper_four_studies_1108_134925/confirmation_pool_audit.json'))
 # Source test slides are disjoint from current model's full training/development catalog.
 slides=audit.get('slides',audit.get('test_slides',[]));allowed={x['wsi_id'] for x in slides}
 if not allowed:
  allowed={x['wsi_id'] for v in audit.values() if isinstance(v,list) for x in v if isinstance(x,dict) and 'wsi_id' in x}
 assert len(allowed)==30
 existing={v['space_key'].split('|')[-1] for v in d.rows.values()};assert not allowed & existing
 source=pd.read_parquet(SOURCES).set_index('source_id').to_dict('index');catalog=pd.read_parquet(CATALOG/'tiles.parquet');candidates=[];excluded=[]
 for row in catalog.to_dict('records'):
  if row['dataset']!='crc' or row['space_key'].split('|')[-1] not in allowed:continue
  try:g=context_geometry(row,source[row['source_id']])
  except ValueError as e:excluded.append(dict(tile=row['tile_id'],reason=str(e)));continue
  candidates.append(dict(row,multiscale_json=json.dumps(g)))
 # Fixed geometry selection, before labels/predictions: up to 40 patches/slide.
 chosen=[]
 for slide in sorted(allowed):chosen.extend(sorted([x for x in candidates if x['space_key'].split('|')[-1]==slide],key=lambda x:hashlib.sha256(x['tile_id'].encode()).hexdigest())[:40])
 dump(r/'exposure_audit.json',dict(status='current_catalog_disjoint_historical_AAAI_exposed',slides=sorted(allowed),current_catalog_overlap=0,candidates=len(candidates),selected=len(chosen),excluded=excluded,not_never_observed_blind_test=True))
 if not chosen:
  # The high-resolution catalog contains train/val only: construct geometric
  # test windows from the already frozen slide manifest, not model scores.
  import pyvips
  manifest=audit['candidates'];prototype=next(v for v in d.rows.values() if v['dataset']=='crc');pg=json.loads(prototype['multiscale_json']);bw,bh=pg['native_boxes'][0][2:]
  for item in manifest:
   im=pyvips.Image.new_from_file(item['wsi_path']);mx,my=1000/float(im.xres),1000/float(im.yres);assert .05<mx<2 and .05<my<2
   sid='confirmation_source_'+item['wsi_id'];source[sid]=dict(image=item['wsi_path'],annotation=item['gt_path'],available_mpp_x=mx,available_mpp_y=my);assert im.width==item['level0_width'] and im.height==item['level0_height']
   for cy in np.linspace(2*bh,im.height-2*bh,6).round().astype(int):
    for cx in np.linspace(2*bw,im.width-2*bw,6).round().astype(int):
     boxes=[[int(cx-bw*f/2),int(cy-bh*f/2),bw*f,bh*f] for f in (1,2,4)];x,y,w,h=boxes[-1];assert x>=0 and y>=0 and x+w<=im.width and y+h<=im.height
     identity='confirm_'+hashlib.sha256(f"{item['wsi_id']}|{cx}|{cy}".encode()).hexdigest()[:24];g=dict(native_boxes=boxes,source_image=item['wsi_path'],native_mpp=[mx,my],context_um=[x*mx,y*my,(x+w)*mx,(y+h)*my])
     chosen.append(dict(prototype,tile_id=identity,source_id=sid,space_key='crc|'+item['wsi_id'],split='test',multiscale_json=json.dumps(g)))
  dump(r/'geometry_revision.json',dict(reason='original highres catalog has no test rows',selection='fixed6x6 geometry grid per frozen test slide before reading labels or predictions',rows=len(chosen)))
 assert chosen,'No eligible original test geometry'
 template=next(iter(d.rows));(r/'labels').mkdir()
 for row in chosen:
  g=json.loads(row['multiscale_json']);s=source[row['source_id']]
  import pyvips
  im=pyvips.Image.new_from_file(g['source_image']);gt=pyvips.Image.new_from_file(s['annotation']);aff=[gt.width/im.width,gt.height/im.height,0,0]
  lab=labels_for(dict(annotation=s['annotation'],gt_affine=aff),g['native_boxes'][0],512);path=r/'labels'/(row['tile_id']+'.png');Image.fromarray(lab).save(path)
  row.update(labels=str(path),labels_array_sha256=hashlib.sha256(lab.tobytes()).hexdigest(),template_id=template,input_mpp_xy_json=json.dumps([g['native_boxes'][0][2]*g['native_mpp'][0]/512,g['native_boxes'][0][3]*g['native_mpp'][1]/512]));d.rows[row['tile_id']]=row
 plans=[];failures=[]
 for slide in sorted(allowed):
  pool=[x for x in chosen if x['space_key'].split('|')[-1]==slide]
  for target in [0,1,3,4,5,6,7,8,9,10,11]:
   valid=[]
   for s in pool:
    try:p=resolve_clicks(dict(dataset='crc',target=target,click_seed=20261109),s,maximum=(3,2))
    except ValueError:continue
    for q in pool:
     if intersects(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um']):continue
     valid.append(dict(p,target=target,support_id=s['tile_id'],query_id=q['tile_id'],support_space=s['space_key'],query_space=q['space_key'],query_has_human_clicks=False))
   ranked=sorted(valid,key=lambda t:hashlib.sha256((t['support_id']+t['query_id']).encode()).hexdigest())[:10];plans.extend(ranked)
   if not ranked:failures.append(dict(slide=slide,target=target,reason='no_geometric_support_query_pair'))
 dump(r/'query_rows.json',{x['tile_id']:x for x in chosen});dump(r/'plans.json',plans);dump(r/'coverage_failures.json',failures);assert plans
 evaluate(r,d,m,plans,[3])


def tree(x,device):
 if torch.is_tensor(x):return x.to(device)
 if isinstance(x,dict):return {k:tree(v,device) for k,v in x.items()}
 if isinstance(x,list):return [tree(v,device) for v in x]
 if isinstance(x,tuple):return tuple(tree(v,device) for v in x)
 return x

def efficiency(r,d,m):
 inv=json.load(open(BASE/'same_slide_analysis_1108_130423/wsi_efficiency_inventory.json'));dump(r/'inventory.json',inv)
 for si,item in enumerate(inv['slides']):
  out=r/f'slide{si}';out.mkdir();(out/'cache').mkdir();task=item['support_task'];sx=inputs(d,task['support_id']);se=encode(m,sx,'full');template=d.rows[task['support_id']];g=json.loads(template['multiscale_json']);w,h=g['native_boxes'][0][2:];mx,my=g['native_mpp'];start=time.monotonic();times=[];window_records=[]
  for j,window in enumerate(item['windows']):
   cx,cy=window['center_native'];boxes=[[round(cx-w*f/2),round(cy-h*f/2),w*f,h*f] for f in (1,2,4)]
   identity=f'efficiency_{si}_{j}';row=dict(template,tile_id=identity,template_id=task['support_id'],multiscale_json=json.dumps(dict(g,native_boxes=boxes,context_um=[boxes[-1][0]*mx,boxes[-1][1]*my,(boxes[-1][0]+boxes[-1][2])*mx,(boxes[-1][1]+boxes[-1][3])*my])))
   d.rows[identity]=row;t=time.monotonic();qx=inputs(d,identity);qe,enc=timing(lambda:encode(m,qx,'full'));cache=out/'cache'/f'{j:06}.pt';torch.save(tree(qe,'cpu'),cache);loaded=tree(torch.load(cache,weights_only=True),'cuda:0')
   p=prompts(task);prob,dec=timing(lambda:decode(m,se,loaded,p,'full'))
   if j==0:
    fresh=m(support=sx,query=qx,prompts=p,local=False,shared=False,with_support=False)['query']['pixel_probability'];assert torch.equal(fresh,prob)
    assert torch.equal(decode(m,se,qe,p,'full'),prob)
   np.save(out/f'prediction_{j:06}.npy',np.packbits(prob[0].cpu().numpy()>=.5));times.append(dict(total_seconds=time.monotonic()-t,encode_seconds=enc,decode_seconds=dec,cache_bytes=cache.stat().st_size));window_records.append(dict(center_native=[cx,cy],boxes=boxes,engineering_only=True));del d.rows[identity]
   if j%10==0:dump(r/'progress.json',dict(pid=os.getpid(),stage='persistent_cache_first_prediction',slide=si,windows=j+1,total=len(item['windows']),seconds=time.monotonic()-start))
  dump(out/'windows.json',window_records);dump(out/'preparation.json',dict(windows=len(times),seconds=time.monotonic()-start,times=times,scope='full fixed interior domain; boundaries excluded; OS cache uncontrolled; no quality claims'))
  update=time.monotonic();p=prompts(task,1)
  for j in range(len(times)):
   qe=tree(torch.load(out/'cache'/f'{j:06}.pt',weights_only=True),'cuda:0');prob=decode(m,se,qe,p,'full');np.save(out/f'updated_{j:06}.npy',np.packbits(prob[0].cpu().numpy()>=.5))
  torch.cuda.synchronize();dump(out/'completion.json',dict(status='complete_engineering_measurement',update_including_cache_IO_and_output_seconds=time.monotonic()-update,peak_GPU_MiB=torch.cuda.max_memory_allocated()/1024**2,cache_bytes=sum(t['cache_bytes'] for t in times),not_full_boundary_coverage=True,spatial_fusion_pending=True))
 dump(r/'completion.json',dict(status='measurements_complete_spatial_fusion_acceptance_pending',slides=len(inv['slides'])))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['robustness','confirmation','efficiency'],required=True);a=p.parse_args();r=a.root;r.mkdir(exist_ok=True);dump(r/'status.json',dict(status='running',pid=os.getpid(),mode=a.mode))
 try:
  with torch.no_grad():
   d,m=load(r);globals()[a.mode](r,d,m)
  dump(r/'status.json',dict(status='complete' if a.mode!='efficiency' else 'measurement_complete_fusion_pending',pid=os.getpid()))
 except Exception as e:
  dump(r/'status.json',dict(status='failed',pid=os.getpid(),error=repr(e)));raise
if __name__=='__main__':main()
