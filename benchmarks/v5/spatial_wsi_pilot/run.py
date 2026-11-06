"""Three real WSI thumbnails, footprint audit, one support per slide and indexed val queries."""
import argparse,json,hashlib,os,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw
import torch,pyvips
from benchmarks.v5.size_aware_study.model import DecoderModel,DecoderData
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--accept',action='store_true');a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for path,h in c['sources'].items():assert sha(path)==h
 for path,h in c['inputs'].items():assert sha(path)==h
 data=DecoderData(Path(c['cache']),'control');catalog=data.rows
 if a.accept:
  rows=json.load(open(r/'rows.json'))
  with np.load(r/'masks.npz') as f:
   for row,bits in zip(rows,f['masks']):
    gt,mpp=data.labels(row['plan']['query_id']);mask=np.unpackbits(bits,count=512*512).reshape(512,512).astype(bool);assert score(mask,gt==row['plan']['target'],gt!=255,mpp)==row['metrics']
  assert len(rows)==len(json.load(open(r/'plans.json')))
  (r/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted_indexed_WSI_pilot',rows=len(rows),rows_sha256=sha(r/'rows.json'),masks_sha256=sha(r/'masks.npz'),limitation='three WSI thumbnails with indexed spatial-block validation only; not dense whole-slide scanning or independent cohort'))+'\n');return
 def box(row):return json.loads(row['multiscale_json'])['context_um']
 def overlap(a,b):return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
 by=defaultdict(list)
 for row in catalog.values():
  if row['dataset']=='crc' and row['split']=='val':by[row['space_key']].append(row)
 episodes=[json.loads(x)['plan'] for x in (Path(c['protocol'])/'evaluation.jsonl').read_text().splitlines() if json.loads(x)['budget']=='3+2' and json.loads(x)['scope']=='remote'];plans=[];inventory=[]
 eligible_slides={catalog[e['query_id']]['space_key'] for e in episodes if e['dataset']=='crc'}
 slides=sorted([k for k,v in by.items() if len(v)>=2 and k in eligible_slides],key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:3];assert len(slides)==3

 for i,slide in enumerate(slides):
  queries=sorted(by[slide],key=lambda x:x['tile_id']);source=json.loads(queries[0]['multiscale_json'])['source_image'];assert Path(source).is_file()
  candidates=[e for e in episodes if catalog[e['query_id']]['space_key']==slide and e['dataset']=='crc'];assert candidates
  task=sorted(candidates,key=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest())[0]
  train=[x for x in catalog.values() if x['split']=='train' and x['space_key']==slide];queryplans=[]
  for q in queries:
   assert not any(overlap(box(q),box(t)) for t in train),'max-context train/val overlap'
   support=catalog[task['support_id']]
   if support['space_key']==slide and overlap(box(q),box(support)):continue
   queryplans.append(dict(task,query_id=q['tile_id']))
  assert queryplans;plans+=queryplans
  thumb=pyvips.Image.thumbnail(source,1024,height=1024,size='down');thumb=thumb[:3];arr=np.ndarray(buffer=thumb.write_to_memory(),dtype=np.uint8,shape=(thumb.height,thumb.width,thumb.bands));Image.fromarray(arr).save(r/f'slide_{i}.png')
  inventory.append(dict(slide=slide,source=source,support_id=task['support_id'],target=task['target'],query_ids=[x['query_id'] for x in queryplans],thumbnail=f'slide_{i}.png',max_input_footprint_checked=True))
 (r/'plans.json').write_text(json.dumps(plans,indent=2)+'\n');(r/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 model=DecoderModel('control');model.load_state_dict(torch.load(c['checkpoint'],map_location='cuda:0',weights_only=True)['model']);model.eval();rows=[];masks=[];start=time.monotonic()
 with torch.no_grad():
  for plan in plans:
   x=data.model_inputs(plan,'xcell','cuda:0');assert not x['local'];mask=(model(**x)['query']['pixel_probability'][0]>=.5).cpu().numpy();gt,mpp=data.labels(plan['query_id']);rows.append(dict(plan=plan,metrics=score(mask,gt==plan['target'],gt!=255,mpp)));masks.append(np.packbits(mask));print(json.dumps(dict(stage='indexed_wsi_scoring',worker_pid=os.getpid(),rows=len(rows),total=len(plans))),flush=True)
 for j,(row,bits) in enumerate(zip(rows,masks)):
  plan=row['plan'];rgb=lambda t:((t[0].permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255).round().clip(0,255).astype('uint8')
  query_rgb=rgb(data.inputs(plan['query_id'],'xcell')['image']);support_rgb=rgb(data.inputs(plan['support_id'],'xcell')['image']);gt,_=data.labels(plan['query_id']);pred=np.unpackbits(bits,count=512*512).reshape(512,512).astype(bool)
  support_im=Image.fromarray(support_rgb);draw=ImageDraw.Draw(support_im)
  for sign,color in [('positive','lime'),('negative','magenta')]:
   for px,py in plan[sign+'_xy']:draw.ellipse((px*512-5,py*512-5,px*512+5,py*512+5),outline=color,width=3)
  panels=[np.asarray(support_im),query_rgb]
  for mask,color in [(gt==plan['target'],np.array([255,220,0])),(pred,np.array([255,0,0]))]:
   overlay=query_rgb.copy();overlay[mask]=(.55*overlay[mask]+.45*color).astype('uint8');overlay[gt==255]=128;panels.append(overlay)
  Image.fromarray(np.concatenate(panels,axis=1)).save(r/f'prediction_{j}.png')
 (r/'rows.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(r/'masks.npz',masks=np.stack(masks));(r/'report.json').write_text(json.dumps(dict(status='complete_unaccepted',slides=3,queries=len(plans),seconds=time.monotonic()-start,scope='indexed val spatial blocks; full-slide thumbnails only'))+'\n')
 html='<meta charset="utf-8"><h1>WSI空间块验证试验</h1><p>仅评分已审计验证空间块；缩略图是全片，预测不是连续全片覆盖。每片固定一个支持任务，查询人工点为0。</p>'
 for i,item in enumerate(inventory):html+=f'<h2>{item["slide"]}</h2><p>目标ID {item["target"]}；验证位置数 {len(item["query_ids"])}</p><img width="640" src="slide_{i}.png">'
 html+='<h2>支持区（绿正点/紫负点）｜查询原图｜查询GT（黄）｜预测（红）</h2>'
 for j,row in enumerate(rows):html+=f'<p>查询 {row["plan"]["query_id"]}，Dice {row["metrics"]["dice"]:.4f}</p><img width=1200 src="prediction_{j}.png">'
 html+='<p>逐任务指标及空间来源见 inventory.json、plans.json、rows.json。</p>';(r/'index.html').write_text(html)
if __name__=='__main__':main()
