import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));method=c['method'];tasks=json.load(open(r/'tasks.json'));assert len(tasks)==11000 and sha(r/'tasks.json')==c['tasks_sha256'];rows={}
 for shard in [0,1]:
  out=r/f'{method}_{shard}';a=json.load(open(out/'acceptance.json'));assert a['status']=='complete_accepted' and a['rows']==5500 and a['tasks_sha256']==c['tasks_sha256'] and a['checkpoint_sha256']==c['checkpoints'][method]['sha256'];assert a['rows_sha256']==sha(out/'rows.json') and a['masks_sha256']==sha(out/'masks.npy')
  for row in json.load(open(out/'rows.json')):assert row['index'] not in rows;rows[row['index']]=row['metrics']
 assert sorted(rows)==list(range(11000));by=defaultdict(list)
 for i,m in rows.items():assert tasks[i]['query_space']==tasks[i]['support_space'];by[tasks[i]['target']].append(m)
 classes={}
 for k,rr in by.items():
  pos=[m for m in rr if m['positive']];neg=[m for m in rr if not m['positive']];classes[k]=dict(present=len(pos),absent=len(neg),dice=float(np.mean([m['dice'] for m in pos])),FPA=float(np.mean([m['false_positive_fraction'] for m in neg])))
 strata={}
 for cut in [.01,.05,.2]:
  groups=defaultdict(list)
  for i,m in rows.items():
   if m['positive'] and m['positive']/m['valid']>=cut:groups[tasks[i]['target']].append(m['dice'])
  strata[str(cut)]=dict(n=sum(map(len,groups.values())),task_mean_Dice=float(np.mean([x for v in groups.values() for x in v])),class_macro_Dice=float(np.mean([np.mean(v) for v in groups.values()])))
 dump(r/'report.json',dict(status='complete_accepted',method=method,track=c['comparison_track'],target_updates=2375,classes=classes,macro_Dice=float(np.mean([m['dice'] for m in classes.values()])),macro_absent_FPA=float(np.mean([m['FPA'] for m in classes.values()])),area_strata=strata,limitations='Matched9500new tasks/2375updates, public initialization; SAM-Med2D native256 three-view fixed automatic transfer bridge; SANSA/HG-mix current9500 reference. Different priors/views/capacity; exposed sameWSI development cohort; queryGT scoring only. Fixed final checkpoint/0.5 threshold. No automatic promotion.'))
 dump(r/'acceptance.json',dict(status='complete_accepted',report_sha256=sha(r/'report.json'),tasks_sha256=c['tasks_sha256'],rows=11000))
if __name__=='__main__':main()
