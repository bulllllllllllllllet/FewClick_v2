"""Paired saved-prediction diagnosis; no thresholds/models selected."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;r.mkdir(exist_ok=False)
 b=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');src=b/'wsi_same_slide_1108_013913';tasks=json.load(open(src/'tasks.json'));sources={};methods={}
 for name,folder,prefix in [('fewclick',src,'full'),('sansa',b/'sansa_adaptation_1109_043250/sansa_ft/evaluation','sansa_ft'),('sam2_fewclick',b/'sam2_encoder_probe_1109_162523/sam2_encoder/evaluation','sam2_encoder')]:
  rows={}
  for shard in [0,1]:
   out=folder/f'{prefix}_{shard}';a=json.load(open(out/'acceptance.json'));path=out/'rows.json';assert digest(path)==a['rows_sha256'];sources[str(path)]=digest(path)
   for x in json.load(open(path)):assert x['index'] not in rows;rows[x['index']]=x['metrics']
  assert sorted(rows)==list(range(11000));methods[name]=rows
 def summarize(ids):
  result={}
  for method,rows in methods.items():
   groups=defaultdict(list)
   for i in ids:groups[tasks[i]['target']].append(rows[i])
   cls={}
   for k,rr in groups.items():
    def mean(f):
     v=[f(x) for x in rr];v=[x for x in v if x is not None];return float(np.mean(v)) if v else None
    cls[k]=dict(n=len(rr),dice=mean(lambda m:m['dice']),recall=mean(lambda m:m['tp']/m['positive'] if m['positive'] else None),precision=mean(lambda m:m['tp']/(m['tp']+m['fp']) if m['positive'] else None) if all(m['tp']+m['fp'] for m in rr if m['positive']) else mean(lambda m:m['tp']/max(1,m['tp']+m['fp']) if m['positive'] else None),bf1=mean(lambda m:m['bf1_um2']),prediction_to_GT_area=mean(lambda m:(m['tp']+m['fp'])/m['positive'] if m['positive'] else None),empty_rate=mean(lambda m:float(m['tp']+m['fp']==0)),absent_FPA=mean(lambda m:m['false_positive_fraction'] if not m['positive'] else None))
   keys=list(next(iter(cls.values()))) if cls else [];macro={k:float(np.mean([v[k] for v in cls.values() if v[k] is not None])) for k in keys if k!='n' and any(v[k] is not None for v in cls.values())};result[method]=dict(n=len(ids),classes=cls,macro=macro)
  return result
 strata={'present':[],'absent':[],'tiny_lt1pct':[],'small_1to5pct':[],'medium_5to20pct':[],'large_ge20pct':[],'both_area_ratio_half_to_two_large':[]}
 for i,t in enumerate(tasks):
  a=methods['fewclick'][i];s=methods['sansa'][i];assert a['positive']==s['positive'] and a['valid']==s['valid'];pos=a['positive'];area=pos/a['valid']
  if pos:
   strata['present'].append(i);strata['tiny_lt1pct' if area<.01 else 'small_1to5pct' if area<.05 else 'medium_5to20pct' if area<.2 else 'large_ge20pct'].append(i)
   if area>=.2 and all(.5<=(m['tp']+m['fp'])/pos<=2 for m in [a,s]):strata['both_area_ratio_half_to_two_large'].append(i)
  else:strata['absent'].append(i)
 report={k:summarize(v) for k,v in strata.items()};paired={}
 for key,ids in strata.items():
  if key=='absent':continue
  dif=[methods['sansa'][i]['dice']-methods['fewclick'][i]['dice'] for i in ids];paired[key]=dict(n=len(ids),sansa_win_fraction=float(np.mean(np.asarray(dif)>0)),mean_dice_delta=float(np.mean(dif)))
 # Paired slide-cluster bootstrap with task strata and class-macro recomputation.
 ids=strata['present'];slides=sorted({tasks[i]['query_space'] for i in ids});rng=np.random.default_rng(20261109);values=[]
 for _ in range(1000):
  count=dict(zip(slides,np.bincount(rng.integers(len(slides),size=len(slides)),minlength=len(slides))));groups=defaultdict(list);weights=defaultdict(list)
  for i in ids:
   w=count[tasks[i]['query_space']]
   if w:groups[tasks[i]['target']].append(methods['sansa'][i]['dice']-methods['fewclick'][i]['dice']);weights[tasks[i]['target']].append(w)
  values.append(np.mean([np.average(v,weights=weights[k]) for k,v in groups.items()]))
 result=dict(status='complete_accepted_readonly',strata=report,paired=paired,slide_cluster_bootstrap_macro_delta_CI95=np.quantile(values,[.025,.975]).tolist(),sources=sources,tasks_sha256=digest(src/'tasks.json'),limitations=['Error localization, not causal module attribution','Precision=0 for empty predictions; absent separate','Area-ratio subset selected from predictions for diagnosis only','Same-slide clustered bootstrap of exposed development set, not independent confirmation','Image encoder, adapters, memory and decoder differ jointly across systems'])
 (r/'report.json').write_text(json.dumps(result,indent=2));(r/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted',report_sha256=digest(r/'report.json'),tasks=11000,methods=3)))
 for k in ['present','tiny_lt1pct','small_1to5pct','medium_5to20pct','large_ge20pct','both_area_ratio_half_to_two_large','absent']:
  print(k,{m:dict(n=v['n'],**v['macro']) for m,v in report[k].items()})
 print('CI',result['slide_cluster_bootstrap_macro_delta_CI95'])
if __name__=='__main__':main()
