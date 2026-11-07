import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));tasks=json.load(open(r/'tasks.json'));assert all(t['support_space']==t['query_space'] for t in tasks);assert len(tasks)==11000 and sha(r/'tasks.json')==c['tasks_sha256'];results={};identities={}
 for method in ['full','no_match','sam']:
  combined={};images={}
  for shard in [0,1]:
   out=r/f'{method}_{shard}';a=json.load(open(out/'acceptance.json'));assert a['status']=='complete_accepted' and a['rows']==5500 and a['tasks_sha256']==c['tasks_sha256'];assert a['rows_sha256']==sha(out/'rows.json') and a['masks_sha256']==sha(out/'masks.npy') and a['checkpoint_sha256']==c['checkpoints'][method]['sha256'];assert a['input_sha256']==sha(out/'input_hashes.json')
   for k,v in json.load(open(out/'input_hashes.json')).items():
    if k in images:assert images[k]==v
    images[k]=v
   for x in json.load(open(out/'rows.json')):assert x['index'] not in combined;combined[x['index']]=x
  assert sorted(combined)==list(range(11000));identities[method]=images;by=defaultdict(list)
  for i,row in combined.items():by[tasks[i]['target']].append(row['metrics'])
  table={}
  for k,rows in by.items():
   pos=[v for v in rows if v['positive']];neg=[v for v in rows if not v['positive']];expected=json.load(open(r/'coverage.json'))['classes'][str(k)];assert len(pos)==expected['present'] and len(neg)==expected['absent']
   table[k]=dict(present=len(pos),absent=len(neg),dice=float(np.mean([v['dice'] for v in pos])) if pos else None,FPA=float(np.mean([v['false_positive_fraction'] for v in neg])) if neg else None,bf1=float(np.mean([v['bf1_um2'] for v in pos if v['bf1_um2'] is not None])))
  results[method]=dict(classes=table,macro_Dice=float(np.mean([v['dice'] for v in table.values() if v['dice'] is not None])),macro_absent_FPA=float(np.mean([v['FPA'] for v in table.values() if v['FPA'] is not None])))
 assert identities['full']==identities['no_match']==identities['sam']
 dump(r/'report.json',dict(status='complete_accepted_11000_tasks_per_method',results=results,coverage=json.load(open(r/'coverage.json')),limitations='SAME-WSI remote only; repeated queries/supports and correlated spatial windows; not11000independentcases; not continuous fullWSI'))
 dump(r/'acceptance.json',dict(status='complete_accepted',report_sha256=sha(r/'report.json'),tasks_sha256=c['tasks_sha256'],total_method_tasks=33000))
if __name__=='__main__':main()
