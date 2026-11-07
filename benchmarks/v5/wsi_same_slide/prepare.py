import argparse,json,hashlib,copy
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,pandas as pd,pyvips
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.paper_multiscale.data import SOURCES,intersects
from benchmarks.v5.fair_comparison.protocol import resolve_clicks
from benchmarks.v5.wsi_quality_study.prepare import dump,CHECKPOINTS,BASE,labels_for

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
 cat,rows=load_cell_catalog(TASKS);vals=[v for v in rows.values() if v['dataset']=='crc' and v['split']=='val'];geoms={k:json.loads(v['multiscale_json']) for k,v in rows.items()};train=defaultdict(list);sources=pd.read_parquet(SOURCES).set_index('source_id').to_dict('index')
 for k,v in rows.items():
  if v['split']=='train':train[v['space_key']].append(geoms[k]['context_um'])
 candidates={};rejections=Counter()
 for v in sorted(vals,key=lambda x:x['tile_id']):
  g=geoms[v['tile_id']];im=pyvips.Image.new_from_file(g['source_image'],access='random');s=sources[v['source_id']];gt=pyvips.Image.new_from_file(s['annotation'],access='random');affine=[gt.width/im.width,gt.height/im.height,0,0]
  for dy in [-8,-4,0,4,8]:
   for dx in [-8,-4,0,4,8]:
    ox=dx*g['native_boxes'][0][2]//2;oy=dy*g['native_boxes'][0][3]//2;boxes=[[x+ox,y+oy,w,h] for x,y,w,h in g['native_boxes']];x,y,w,h=boxes[-1];mx,my=g['native_mpp'];foot=[x*mx,y*my,(x+w)*mx,(y+h)*my]
    if x<0 or y<0 or x+w>im.width or y+h>im.height:rejections['source_boundary']+=1;continue
    if any(intersects(foot,b) for b in train[v['space_key']]):rejections['train_4x_overlap']+=1;continue
    key=v['space_key']+'|'+str(boxes[0]);identity='samewsi_'+hashlib.sha256(key.encode()).hexdigest()[:24]
    if identity in candidates:continue
    row=dict(v,tile_id=identity,template_id=v['tile_id'],multiscale_json=json.dumps(dict(g,native_boxes=boxes,context_um=foot)),source_gt_annotation=s['annotation'],gt_affine=affine);candidates[identity]=row
 dump(r/'geometry_candidates.json',dict(status='fixed_before_query_labels_and_predictions',rows=list(candidates.values()),rejections=dict(rejections),source_catalog_sha256=sha(cat/'tiles.parquet')))
 # Labels only determine declared present/absent strata, never inference candidates/prompts.
 labeldir=r/'query_labels';labeldir.mkdir();counts={}
 for i,(identity,v) in enumerate(candidates.items()):
  geom=json.loads(v['multiscale_json']);lab=labels_for(dict(annotation=v['source_gt_annotation'],gt_affine=v['gt_affine']),geom['native_boxes'][0],512);path=labeldir/(identity+'.png');Image.fromarray(lab).save(path);v['labels']=str(path);v['labels_array_sha256']=hashlib.sha256(lab.tobytes()).hexdigest();w,h=geom['native_boxes'][0][2:];mx,my=geom['native_mpp'];v['input_mpp_xy_json']=json.dumps([w*mx/512,h*my/512]);counts[identity]={int(k):int(n) for k,n in zip(*np.unique(lab,return_counts=True))}
  if i%100==0:print(json.dumps(dict(stage='query_labels',rows=i+1,total=len(candidates))),flush=True)
 byslide=defaultdict(list)
 for identity,v in candidates.items():byslide[v['space_key']].append((identity,v))
 tasks=[];coverage={};rejected=[]
 for target in [0,1,3,4,5,6,7,8,9,10,11]:
  buckets=defaultdict(list)
  for support in vals:
   if target not in json.loads(support['eligible_json']):continue
   sg=geoms[support['tile_id']]
   assert not any(intersects(sg['context_um'],b) for b in train[support['space_key']])
   try:prompt=resolve_clicks(dict(dataset='crc',target=target,click_seed=20261108+target),support,maximum=(3,2))
   except ValueError as e:rejected.append(dict(support=support['tile_id'],target=target,reason=str(e)));continue
   truth=np.asarray(Image.open(support['labels']));assert all(truth[int(y*512),int(x*512)]==target for x,y in prompt['positive_xy']);assert all(truth[int(y*512),int(x*512)] not in [target,255] for x,y in prompt['negative_xy'])
   for identity,query in byslide[support['space_key']]:
    qg=json.loads(query['multiscale_json'])
    if intersects(sg['context_um'],qg['context_um']):continue
    plan=dict(prompt,dataset='crc',target=target,support_id=support['tile_id'],query_id=identity,split='val',local=False,query_has_human_clicks=False,gt_simulated_clicks=True,query_positive_pixels=counts[identity].get(target,0),support_space=support['space_key'],query_space=query['space_key'])
    rank=hashlib.sha256(f"{target}|{support['tile_id']}|{identity}".encode()).hexdigest();buckets[identity].append((rank,plan))
  pools={}
  for present in [True,False]:
   bs=[sorted(v) for q,v in sorted(buckets.items()) if bool(counts[q].get(target,0))==present];pool=[]
   for depth in range(max(map(len,bs),default=0)):
    for b in bs:
     if depth<len(b):pool.append(b[depth][1])
   pools[present]=pool
  assert sum(map(len,pools.values()))>=1000,('insufficient SAME-WSI unique pairs',target,{k:len(v) for k,v in pools.items()})
  npos=min(700,len(pools[True]));nneg=min(300,len(pools[False]));need=1000-npos-nneg;extra=min(need,len(pools[True])-npos);npos+=extra;nneg+=need-extra
  chosen=pools[True][:npos]+pools[False][:nneg];assert len(chosen)==1000;tasks.extend(chosen);coverage[target]=dict(available_present=len(pools[True]),available_absent=len(pools[False]),present=npos,absent=nneg,unique_queries=len({x['query_id'] for x in chosen}),query_slides=len({x['query_space'] for x in chosen}));print(json.dumps(dict(stage='same_slide_tasks',target=target,total=len(tasks),coverage=coverage[target])),flush=True)
 assert len(tasks)==11000 and len({(v['target'],v['support_id'],v['query_id']) for v in tasks})==11000
 for i,t in enumerate(tasks):t['index']=i;assert t['support_space']==t['query_space']
 selected={t['query_id'] for t in tasks};dump(r/'query_rows.json',{k:v for k,v in candidates.items() if k in selected});dump(r/'tasks.json',tasks);dump(r/'coverage.json',dict(classes=coverage,total_tasks=11000,unique_queries=len(selected),query_slides=len({t['query_space'] for t in tasks}),same_slide=11000,cross_slide=0,overlapping_4x=0,rejected_supports=rejected))
 sources_py=list(Path('benchmarks/v5/wsi_same_slide').glob('*.py'))+[Path('benchmarks/v5/wsi_quality_study/worker.py'),Path('benchmarks/v5/support_match_study/model.py'),Path('benchmarks/v5/paper_completion/model.py'),Path('benchmarks/v5/sam_prior_repair/model.py')]
 dump(r/'contract.json',dict(tasks_sha256=sha(r/'tasks.json'),query_rows_sha256=sha(r/'query_rows.json'),catalog_sha256=sha(cat/'tiles.parquet'),checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in CHECKPOINTS.items()},sources={str(p):sha(p) for p in sources_py},cache=str(BASE/'cell_repair_rebalance_1102_210552/normalized'),threshold=.5,input='native三尺度bilinear512, support original label / query canonical whole-slide label; known index rounding differences disclosed',query_human_clicks=0))
if __name__=='__main__':main()
