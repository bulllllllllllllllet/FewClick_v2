"""Audit class coverage and candidate continuous WSI grids without changing splits."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
 _,rows=load_cell_catalog(TASKS);counts={}
 for split in ['train','val']:
  c=Counter()
  for row in rows.values():
   if row['dataset']=='crc' and row['split']==split:
    for k in json.loads(row['eligible_json']):c[int(k)]+=1
  counts[split]=dict(c)
 candidates=[3,7,9] # necrosis, muscle, mucus; frozen before model results
 (r/'generalization_inventory.json').write_text(json.dumps(dict(counts=counts,candidates=candidates,selection='morphological diversity; no score selection; fat remains supervised',status='coverage_audit_only',strict_holdout='must mask chosen class in all training losses including support, boundary, assignment purity and all domains; filter positive episodes; audit negative clicks; fresh public initialization required'),indent=2))
 def overlap(a,b):return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
 train={}
 for row in rows.values():
  if row['split']=='train':train.setdefault(row['space_key'],[]).append(json.loads(row['multiscale_json'])['context_um'])
 evaluation=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p3_task_protocol_1103_003729/evaluation.jsonl')
 tasks=[json.loads(x)['plan'] for x in evaluation.read_text().splitlines() if json.loads(x)['scope']=='remote' and json.loads(x)['budget']=='3+2' and json.loads(x)['plan']['dataset']=='crc']
 plans=[];seen=set()
 for task in sorted(tasks,key=lambda t:hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest()):
  row=rows[task['query_id']];slide=row['space_key']
  if slide in seen:continue
  g=json.loads(row['multiscale_json']);s=rows[task['support_id']];sg=json.loads(s['multiscale_json']);tiles=[]
  for dy in [-1,0,1]:
   for dx in [-1,0,1]:
    ox=dx*g['native_boxes'][0][2]//2;oy=dy*g['native_boxes'][0][3]//2
    boxes=[[x+ox,y+oy,w,h] for x,y,w,h in g['native_boxes']]
    x,y,w,h=boxes[-1];mx,my=g['native_mpp'];foot=[x*mx,y*my,(x+w)*mx,(y+h)*my]
    if x<0 or y<0 or any(overlap(foot,b) for b in train.get(slide,[])) or (s['space_key']==slide and overlap(foot,sg['context_um'])):break
    tiles.append(dict(dx=dx,dy=dy,boxes=boxes,footprint_um=foot))
   if len(tiles)!=(dy+2)*3:break
  if len(tiles)==9:
   plans.append(dict(slide=slide,task=task,geometry=g,tiles=tiles));seen.add(slide)
  if len(plans)==3:break
 (r/'wsi_grid_plan.json').write_text(json.dumps(dict(status='preparing_continuous_ROI',plans=plans,tiles=len(plans)*9,annotation_scope='center indexed annotation only; full ROI quality requires source annotation registration',split_audit='all shifted 4x footprints disjoint from current training inputs and support footprint'),indent=2))
 assert len(plans)==3,'insufficient audited grids; do not relax footprint rules'
 print(json.dumps(dict(status='accepted_preparation_only',slides=3,tiles=27,candidates=candidates,counts=counts)))
if __name__=='__main__':main()
