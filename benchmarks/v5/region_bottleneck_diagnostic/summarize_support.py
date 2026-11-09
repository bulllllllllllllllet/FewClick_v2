import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;rows=[]
 for s in [0,1]:
  a=json.load(open(r/f'shard{s}/acceptance.json'));assert a['status']=='complete_accepted_diagnostic';rows+=json.load(open(r/f'shard{s}/rows.json'))
 plans=json.load(open(r/'plans.json'));assert len(rows)==len(plans) and {x['index'] for x in rows}=={x['index'] for x in plans};groups=defaultdict(list)
 for row in rows:
  area=row['support_target_fraction'];groups['support_small_lt5pct' if area<.05 else 'support_medium_5to20pct' if area<.2 else 'support_large_ge20pct'].append(row);groups['query_'+row['query_stratum']].append(row)
 keys=['support_target_fraction','support_Dice','hard_oracle_Dice','soft_attained_Dice','positive_purity_mean','negative_target_mean','purity_separation','positive_enrichment','region_assignment_cross_cosine_mean','token_cross_cosine_mean','same_argmax_region_pair_fraction'];report={}
 for name,rr in groups.items():
  report[name]=dict(n=len(rr),means={k:float(np.mean([x[k] for x in rr])) for k in keys},positive_purity_lt_half=sum(x['positive_purity_mean']<.5 for x in rr),soft_oracle_ge08=sum(x['soft_attained_Dice']>=.8 for x in rr),individuals=rr)
 out=dict(status='complete_accepted_diagnostic',tasks=len(rows),groups=report,limitations=['failure-stratified diagnostic sample','region purity is assignment-based proxy not proof of token semantic content','soft oracle attained fit not certified upper bound','no model parameters or inference thresholds changed'])
 (r/'report.json').write_text(json.dumps(out,indent=2));(r/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted_diagnostic',report_sha256=hashlib.sha256((r/'report.json').read_bytes()).hexdigest())))
 for k,v in report.items():print(k,{key:value for key,value in v.items() if key!='individuals'})
if __name__=='__main__':main()
