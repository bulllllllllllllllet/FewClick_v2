import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;plans=json.load(open(r/'plans.json'));rows=[]
 for i in [0,1]:
  a=json.load(open(r/f'shard{i}/acceptance.json'));assert a['status']=='complete_accepted_diagnostic';rows+=json.load(open(r/f'shard{i}/rows.json'))
 assert {x['index'] for x in rows}=={x['index'] for x in plans} and len(rows)==len(plans)
 groups=defaultdict(list)
 for x in rows:groups[x['stratum']].append(x)
 result={}
 for name,rr in groups.items():
  fields=['support_metrics','support_region_only','query_metrics','query_region_only','hard_partition_oracle','attained_soft_oracle'];result[name]=dict(n=len(rr),mean_Dice={k:float(np.mean([x[k]['dice'] for x in rr])) for k in fields},support_Dice_ge_half=sum(x['support_metrics']['dice']>=.5 for x in rr),soft_oracle_Dice_ge_08=sum(x['attained_soft_oracle']['dice']>=.8 for x in rr),max_probability_mean=float(np.mean([x['query_probability_max'] for x in rr])))
 report=dict(status='complete_accepted_diagnostic',tasks=len(rows),groups=result,limitations='Failure-stratified diagnostic sample; support local not remote main result; oracle uses GT; attained soft fit not certified upper bound; no module causality')
 (r/'report.json').write_text(json.dumps(report,indent=2));(r/'acceptance.json').write_text(json.dumps(dict(status='complete_accepted_diagnostic',report_sha256=hashlib.sha256((r/'report.json').read_bytes()).hexdigest(),tasks=len(rows))))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
