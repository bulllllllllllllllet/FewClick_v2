"""Read-only paired error review of frozen repair arms; no model selection."""
import json,hashlib,argparse
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root;root.mkdir(exist_ok=False)
 b=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');src=b/'wsi_same_slide_1108_013913';tasks=json.load(open(src/'tasks.json'));models={};sources={}
 locations=[('original',src,'full'),('detail',b/'support_token_repair_1110_005747/point_detail/evaluation','point_detail'),('aux',b/'region_auxiliary_1110_135144/aux_original/evaluation','aux_original'),('detail_aux',b/'region_auxiliary_1110_135144/aux_detail/evaluation','aux_detail'),('sansa',b/'sansa_adaptation_1109_043250/sansa_ft/evaluation','sansa_ft')]
 for name,folder,prefix in locations:
  assert sha(folder/'tasks.json')==sha(src/'tasks.json');rr={}
  for shard in [0,1]:
   out=folder/f'{prefix}_{shard}';acc=json.load(open(out/'acceptance.json'));assert acc['status']=='complete_accepted' and sha(out/'rows.json')==acc['rows_sha256'];sources[str(out/'rows.json')]=sha(out/'rows.json')
   for row in json.load(open(out/'rows.json')):assert row['index'] not in rr;rr[row['index']]=row['metrics']
  assert sorted(rr)==list(range(11000));models[name]=rr
 base=models['original'];present=[i for i in base if base[i]['positive']];large=[i for i in present if base[i]['positive']/base[i]['valid']>=.2];small=[i for i in present if base[i]['positive']/base[i]['valid']<.05];absent=[i for i in base if not base[i]['positive']]
 for rr in models.values():
  for i,m in rr.items():assert (m['positive'],m['valid'])==(base[i]['positive'],base[i]['valid'])
 def summary(rr,ids):
  groups=defaultdict(list)
  for i in ids:
   m=rr[i];pred=m['tp']+m['fp'];pos=m['positive'];groups[tasks[i]['target']].append(dict(dice=m['dice'],recall=m['tp']/pos if pos else None,precision=m['tp']/max(pred,1) if pos else None,area_ratio=pred/pos if pos else None,bf1=m['bf1_um2'],FPA=m['false_positive_fraction'] if not pos else None))
  cls={k:{key:float(np.mean([v[key] for v in vals if v[key] is not None])) if any(v[key] is not None for v in vals) else None for key in vals[0]} for k,vals in groups.items()}
  macro={key:float(np.mean([v[key] for v in cls.values() if v[key] is not None])) if any(v[key] is not None for v in cls.values()) else None for key in next(iter(cls.values()))}
  return dict(n=len(ids),macro=macro,classes=cls,empty_count=sum(rr[i]['tp']+rr[i]['fp']==0 for i in ids))
 strata={k:{name:summary(rr,ids) for name,rr in models.items()} for k,ids in [('present',present),('large',large),('small',small),('absent',absent)]}
 pairs={};rng=np.random.default_rng(20261110);slides=sorted({t['query_space'] for t in tasks});sid={s:i for i,s in enumerate(slides)};weights=rng.multinomial(len(slides),np.ones(len(slides))/len(slides),size=1000)
 for a,c in [('detail','original'),('aux','original'),('detail_aux','detail'),('sansa','detail')]:
  result={}
  for key,ids in [('present',present),('large',large)]:
   deltas=[]
   for cls in sorted({tasks[i]['target'] for i in ids}):
    ix=[i for i in ids if tasks[i]['target']==cls];counts=np.zeros(len(slides));sums=np.zeros(len(slides))
    for i in ix:j=sid[tasks[i]['query_space']];counts[j]+=1;sums[j]+=models[a][i]['dice']-models[c][i]['dice']
    den=weights@counts;deltas.append(np.divide(weights@sums,den,out=np.full(len(weights),np.nan),where=den>0))
   ci=np.nanquantile(np.nanmean(deltas,axis=0),[.025,.975]);oldempty=[i for i in ids if models[c][i]['tp']+models[c][i]['fp']==0];newempty=[i for i in ids if models[a][i]['tp']+models[a][i]['fp']==0];rescued=[i for i in oldempty if i not in set(newempty)]
   result[key]=dict(delta_macro=strata[key][a]['macro']['dice']-strata[key][c]['macro']['dice'],CI95=ci.tolist(),old_empty=len(oldempty),new_empty=len(newempty),empty_to_nonempty=len(rescued),nonempty_to_empty=len(set(newempty)-set(oldempty)),rescued_Dice_mean=float(np.mean([models[a][i]['dice'] for i in rescued])) if rescued else None)
  pairs[a+'_vs_'+c]=result
 consumption=b/'p3_task_protocol_1103_003729/training_consumption.jsonl';training=[json.loads(l) for l in consumption.read_text().splitlines()];counts=Counter((t['dataset'],'local' if t['support_id']==t['query_id'] else 'same_slide' if t['source_space']==t['query_space'] else 'cross_slide') for t in training)
 monitors={};reference_plans=None
 for name,folder,prefix in [('original',b/'support_region_study_1106_162117/control/control','train_monitor_'),('detail',b/'support_token_repair_1110_005747/point_detail/formal','monitor_'),('aux',b/'region_auxiliary_1110_135144/aux_original/formal','monitor_'),('detail_aux',b/'region_auxiliary_1110_135144/aux_detail/formal','monitor_')]:
  monitors[name]={}
  for step in [0,250,500,1000,2000]:
   f=folder/f'{prefix}{step}.json';x=json.load(open(f));plans=[v['plan'] for v in x['rows']]
   if reference_plans is None:reference_plans=plans
   assert plans==reference_plans;monitors[name][str(step)]=x['groups'];sources[str(f)]=sha(f)
 result=dict(status='complete_accepted_readonly',strata=strata,pairs=pairs,train_task_counts={str(k):v for k,v in counts.items()},evaluation_dataset_counts=dict(Counter(t['dataset'] for t in tasks)),training_consumption_sha256=sha(consumption),train_monitors=monitors,sources=sources,task_sha256=sha(src/'tasks.json'),limitations=['Exploratory exposed development set, single training seed; bootstrap clustered by slide and conditional on fitted checkpoints, no multiplicity correction','Train monitor only32 tasks, CRC remote-present only4; functional gate repeatedly fits32 whereas formal training uses8000 mixed consumptions','Training-task mismatch observed, causality and effect on SANSA gap not established','Auxiliary final2000 pre-matcher masks not measured here; gate branch learning does not establish formal branch health'])
 (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');(root/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted',report_sha256=sha(root/'report.json'),tasks=11000,methods=5)))
 print('PAIRS',json.dumps(pairs));print('LARGE',json.dumps({n:v['macro']|dict(empty=v['empty_count']) for n,v in strata['large'].items()}));print('SMALL',json.dumps({n:v['macro']['dice'] for n,v in strata['small'].items()}));print('TRAIN',json.dumps({n:{k:v['Dice'] for k,v in x['2000'].items()} for n,x in monitors.items()}))
if __name__=='__main__':main()
