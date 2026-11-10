"""Freeze a same-slide remote training queue without changing historical artifacts."""
import argparse,json,hashlib,os,time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.multidomain_training.click_policy import representative_clicks,POLICY

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
TASKS=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def overlap(a,b):return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
def initialize():
 global ROWS
 _,ROWS=load_cell_catalog(TASKS)
@lru_cache(maxsize=48)
def labels(identity):
 r=ROWS[identity];assert r['split']=='train'
 with Image.open(r['labels']) as f:gt=np.asarray(f).copy()
 assert hashlib.sha256(gt.tobytes()).hexdigest()==r['labels_array_sha256']
 return gt

def realize(t):
 s=ROWS[t['support_id']];q=ROWS[t['query_id']];sgt=labels(t['support_id']);qgt=labels(t['query_id'])
 assert s['split']==q['split']=='train' and s['space_key']==q['space_key'] and t['support_id']!=t['query_id']
 assert not overlap(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um'])
 assert bool(np.any(qgt==t['target']))==t['query_target_present']
 clicks,audit=representative_clicks(sgt,t['target'],json.loads(s['input_mpp_xy_json']),t['positive_count'],t['negative_count'],np.random.default_rng(t['click_seed']),dataset='crc')
 for sign in ['positive','negative']:
  for x,y in clicks[sign+'_xy']:
   code=sgt[int(y*sgt.shape[0]),int(x*sgt.shape[1])];assert code!=255 and ((code==t['target'])==(sign=='positive'))
 return t|clicks|dict(click_audit=audit,prompt_policy=POLICY,gt_simulated_clicks=True,support_quality='original',query_quality='original',support_organ=s['organ'],query_organ=q['organ'],query_positive_pixels=int((qgt==t['target']).sum()),support_positive_pixels=int((sgt==t['target']).sum()),support_labels_sha256=s['labels_array_sha256'],query_labels_sha256=q['labels_array_sha256'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(exist_ok=False);start=time.monotonic();initialize();train={k:v for k,v in ROWS.items() if v['split']=='train'};boxes={k:json.loads(v['multiscale_json'])['context_um'] for k,v in ROWS.items()};by_slide=defaultdict(list)
 for k,v in train.items():by_slide[v['space_key']].append(k)
 pools=defaultdict(list)
 for ids in by_slide.values():
  for s in ids:
   for q in ids:
    if overlap(boxes[s],boxes[q]):continue
    counts=json.loads(train[q]['counts_json']);eligible=json.loads(train[q]['eligible_json'])
    for target in json.loads(train[s]['eligible_json']):
     positive=counts.get(str(target),0)>0
     if not positive or target in eligible:pools[train[s]['dataset'],target,positive].append((s,q))
 dump(r/'availability.json',{str(k):len(v) for k,v in sorted(pools.items())})
 source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';lines=source.read_text().splitlines(keepends=True);old=list(map(json.loads,lines));assert len(old)==4000
 classes=sorted({t['target'] for t in old if t['dataset']=='crc'});used=set();class_use=Counter();slide_use=Counter();recipes=[]
 templates=[t for t in old if t['support_id']!=t['query_id']]
 for i in range(5500):
  t=templates[i%len(templates)];positive=i<5000;available=[k for k in classes if class_use[k,positive]<len(pools['crc',k,positive])];assert available
  target=min(available,key=lambda k:(class_use[k,positive],k));candidates=[pair for pair in pools['crc',target,positive] if (pair[0],pair[1],target) not in used];assert candidates
  s,q=min(candidates,key=lambda pair:(slide_use[train[pair[0]]['space_key'],positive],hashlib.sha256(f'{20261110}|{i}|{pair}'.encode()).hexdigest()))
  used.add((s,q,target));class_use[target,positive]+=1;slide_use[train[s]['space_key'],positive]+=1
  recipes.append(dict(number=i,dataset='crc',target=target,support_id=s,query_id=q,source_space=train[s]['space_key'],support_space=train[s]['space_key'],query_space=train[q]['space_key'],kind='same_present' if positive else 'same_absent',protocol='same_slide_remote',local=False,split='train',query_has_human_clicks=False,query_target_present=positive,positive_count=t['positive_count'],negative_count=t['negative_count'],click_seed=20261110+500000000+i,source_training_line=i,training_recipe='same_slide_crc_remote_v1',replaces_kind=t['kind'],replaces_dataset=t['dataset']))
 assert len(recipes)==5500 and Counter(t['query_target_present'] for t in recipes)=={True:5000,False:500}
 # Check all used training footprints against validation inputs, including sameWSI evaluation query windows.
 selected=set(k for t in old if t['support_id']==t['query_id'] for k in [t['support_id']])|set(k for t in recipes for k in [t['support_id'],t['query_id']]);evalrows=json.load(open(BASE/'wsi_same_slide_1108_013913/query_rows.json'));excluded=defaultdict(list)
 for v in list(ROWS.values())+list(evalrows.values()):
  if v['split']!='train':excluded[v['space_key']].append(json.loads(v['multiscale_json'])['context_um'])
 for k in selected:
  assert train[k]['training_allowed'] and train[k]['slide_verified']
  assert not any(overlap(boxes[k],bb) for bb in excluded[train[k]['space_key']]),k
 dump(r/'recipe.json',recipes);dump(r/'contract.json',dict(status='preparing',source=str(source),source_sha256=sha(source),source_code_sha256=sha(__file__),seed=20261110,consumption=9500,updates=2375,batch=4,local_occurrences=4000,remote_occurrences=5500,remote_dataset='crc',present_absent_remote=[5000,500],scope='same-slide different physical patches and disjoint full4x',training_models_started=False,domain_change='TIGER local retained; all remote CRC; newly balanced remote classes with availability caps',class_unique_pair_counts={str(k):v for k,v in class_use.items()},historical_evaluation_unchanged=True))
 resolved={}
 with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
  for t in pool.map(realize,recipes,chunksize=4):
   resolved[t['number']]=t
   if len(resolved)%50==0:dump(r/'progress.json',dict(pid=os.getpid(),resolved=len(resolved),total=5500,seconds=time.monotonic()-start))
 local_lines=[line for line,t in zip(lines,old) if t['support_id']==t['query_id']]*2
 assert len(local_lines)==4000
 newlines=local_lines+[json.dumps(resolved[i],sort_keys=True)+'\n' for i in range(5500)]
 order=np.random.default_rng(20261110).permutation(len(newlines));newlines=[newlines[i] for i in order]
 (r/'training9500.jsonl').write_text(''.join(newlines));(r/'training_consumption.jsonl').write_text(''.join(newlines))
 new=list(map(json.loads,newlines));assert len(new)==9500
 assert Counter(line for line,t in zip(newlines,new) if t['support_id']==t['query_id'])==Counter(local_lines)
 for t in new:
  if t['support_id']!=t['query_id']:assert t['source_space']==t['query_space'] and t['kind'] in ['same_present','same_absent']
 fixed=[]
 for dataset,kind,count in [('crc','local',8),('tiger','local',8),('crc','same_present',8),('crc','same_absent',8)]:
  seen=set();group=[]
  for t in new:
   key=(t['support_id'],t['query_id'],t['target'])
   if t['dataset']==dataset and t['kind']==kind and key not in seen:group.append(t);seen.add(key)
   if len(group)==count:break
  assert len(group)==count;fixed+=group
 dump(r/'functional_tasks.json',fixed);dump(r/'monitor_tasks.json',fixed)
 # Independent resolver replay on fixed entries.
 from benchmarks.v5.fair_comparison.protocol import resolve_clicks
 for index in [0,2499,4999,5000,5499]:
  t=resolved[recipes[index]['number']];check=resolve_clicks(t,ROWS[t['support_id']],(t['positive_count'],t['negative_count']))
  for key in ['positive_xy','negative_xy','click_audit','prompt_policy']:assert check[key]==t[key]
 dump(r/'input_manifest.json',{k:{f:train[k][f] for f in ['space_key','patient_id','split','labels_array_sha256','multiscale_json','multiscale_sha256']} for k in sorted(selected)})
 dump(r/'acceptance.json',dict(status='accepted_same_slide_training_queue',training9500_sha256=sha(r/'training9500.jsonl'),training_consumption_sha256=sha(r/'training_consumption.jsonl'),source_sha256=sha(source),source_code_sha256=sha(__file__),source_tasks=9500,consumption=9500,unique_remote_pairs_target=5500,remote_cross_slide=0,local_lines_exact=True,all_remote_full4x_disjoint=True,train_eval_full4x_disjoint=True,local_prompt_budgets_exact=True,remote_prompt_budget_templates_preserved=True,all_remote_GT_signs_checked=True,independent_prompt_replay=5,actual_consumption_domain_kind_counts={str(k):v for k,v in Counter((t['dataset'],t['kind']) for t in new).items()},classes=classes,training_started=False,required_before_training='Use this source for functional gates and monitors too; never keep historical cross-slide gate tasks. Fresh matched FewClick/SANSA recipe must consume the identical new hash.'))
 print(json.dumps(json.load(open(r/'acceptance.json')),indent=2))
if __name__=='__main__':main()
