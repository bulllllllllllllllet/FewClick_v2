import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.paper_multiscale.data import intersects
from benchmarks.v5.fair_comparison.protocol import resolve_clicks
from benchmarks.v5.wsi_quality_study.prepare import dump,CHECKPOINTS,BASE

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
 cat,rows=load_cell_catalog(TASKS);vals=[v for v in rows.values() if v['dataset']=='crc' and v['split']=='val'];geom={k:json.loads(v['multiscale_json']) for k,v in rows.items()};train=defaultdict(list)
 for k,v in rows.items():
  if v['split']=='train':train[v['space_key']].append(geom[k]['context_um'])
 for v in vals:assert not any(intersects(geom[v['tile_id']]['context_um'],b) for b in train[v['space_key']])
 counts={};labels={}
 for v in vals:
  gt=np.asarray(Image.open(v['labels']));assert hashlib.sha256(gt.tobytes()).hexdigest()==v['labels_array_sha256'];labels[v['tile_id']]=gt;counts[v['tile_id']]={int(k):int(n) for k,n in zip(*np.unique(gt,return_counts=True))}
 tasks=[];coverage={};rejected=[]
 for target in [0,1,3,4,5,6,7,8,9,10,11]:
  supports=[]
  for v in vals:
   if target not in json.loads(v['eligible_json']):continue
   try:plan=resolve_clicks(dict(dataset='crc',target=target,click_seed=20261108+target),v,maximum=(3,2))
   except ValueError as e:rejected.append(dict(support=v['tile_id'],target=target,reason=str(e)));continue
   gt=labels[v['tile_id']]
   assert all(gt[int(y*512),int(x*512)]==target for x,y in plan['positive_xy'])
   assert all(gt[int(y*512),int(x*512)] not in [target,255] for x,y in plan['negative_xy'])
   supports.append((v,plan))
  perquery=defaultdict(list)
  for q in vals:
   for s,prompt in supports:
    if q['tile_id']==s['tile_id']:continue
    if q['space_key']==s['space_key'] and intersects(geom[q['tile_id']]['context_um'],geom[s['tile_id']]['context_um']):continue
    key=f"{target}|{s['tile_id']}|{q['tile_id']}";rank=hashlib.sha256(key.encode()).hexdigest()
    plan=dict(prompt,dataset='crc',target=target,support_id=s['tile_id'],query_id=q['tile_id'],split='val',local=False,query_has_human_clicks=False,gt_simulated_clicks=True,query_positive_pixels=counts[q['tile_id']].get(target,0),support_space=s['space_key'],query_space=q['space_key'])
    perquery[q['tile_id']].append((rank,plan))
  coverage[target]={}
  for present,quota in [(True,700),(False,300)]:
   buckets=[sorted(v) for q,v in sorted(perquery.items()) if bool(counts[q].get(target,0))==present];pool=[]
   for depth in range(max(map(len,buckets),default=0)):
    for bucket in buckets:
     if depth<len(bucket):pool.append(bucket[depth][1])
   assert len(pool)>=quota,('insufficient unique pairs',target,present,len(pool),quota)
   chosen=pool[:quota];coverage[target]['present' if present else 'absent']=dict(available=len(pool),selected=len(chosen),unique_queries=len({x['query_id'] for x in chosen}),query_slides=len({x['query_space'] for x in chosen}))
   tasks.extend(chosen)
  print(json.dumps(dict(stage='freeze_tasks',target=target,total=len(tasks))),flush=True)
 assert len(tasks)==11000 and len({(x['target'],x['support_id'],x['query_id']) for x in tasks})==11000
 for i,t in enumerate(tasks):t['index']=i
 dump(r/'tasks.json',tasks);dump(r/'coverage.json',dict(classes=coverage,rejected_supports=rejected,total_tasks=11000,unique_queries=len({x['query_id'] for x in tasks}),query_slides=len({x['query_space'] for x in tasks}),rule='700present+300absent per class; query-round-robin unique pairs; no repeated prompt groups; same cohort spatial validation not independent11000cases'))
 sources=list(Path('benchmarks/v5/wsi_taskscale').glob('*.py'))+[Path('benchmarks/v5/wsi_quality_study/worker.py'),Path('benchmarks/v5/support_match_study/model.py'),Path('benchmarks/v5/paper_completion/model.py'),Path('benchmarks/v5/sam_prior_repair/model.py')]
 dump(r/'contract.json',dict(tasks_sha256=sha(r/'tasks.json'),catalog_sha256=sha(cat/'tiles.parquet'),checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in CHECKPOINTS.items()},sources={str(p):sha(p) for p in sources},cache=str(BASE/'cell_repair_rebalance_1102_210552/normalized'),threshold=.5,input='native_context three scales PIL bilinear512 on BOTH support and query; historical labels retained and coordinate rounding differences disclosed',query_human_clicks=0))
if __name__=='__main__':main()
