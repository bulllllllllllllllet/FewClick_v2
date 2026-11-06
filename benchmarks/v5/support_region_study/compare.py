"""Bind matched2000 arms and original baseline reproduction before quality promotion."""
import argparse,json,time
from pathlib import Path
import torch
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;torch.set_num_threads(2);c=json.load(open(r/'study.json'))
 while True:
  statuses=[]
  for arm in ['control','region']:
   path=r/arm/'stage.json';stage=json.load(open(path)) if path.exists() else {};assert stage.get('status')!='failed',(arm,stage);statuses.append(stage.get('status'))
  if statuses==['complete_accepted']*2:break
  time.sleep(10)
 raw={}
 for arm in ['control','region']:
  folder=r/arm/'control';proof=json.load(open(folder/'independent_acceptance.json'))
  for name,key in [('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256'),('report.json','report_sha256')]:assert sha(folder/name)==proof[key]
  raw[arm]=json.load(open(folder/'rows.json'));ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert ck['updates']==2000 and all(int(v['step'])==2000 for v in ck['optimizer']['state'].values());assert all(torch.isfinite(v).all() for v in ck['model'].values());del ck
 assert (r/'control/control/training_consumed.jsonl').read_bytes()==(r/'region/control/training_consumed.jsonl').read_bytes()
 ref=Path(c['baseline']);assert raw['control']==json.load(open(ref/'rows.json'))
 one=torch.load(ref/'checkpoint.pt',map_location='cpu',weights_only=True);two=torch.load(r/'control/control/checkpoint.pt',map_location='cpu',weights_only=True);assert tree_sha(one['model'])==tree_sha(two['model']);del one,two
 guards=compare_rows(raw['control'],raw['region']);values={k:summarize_arm(v)[2] for k,v in raw.items()}
 (r/'comparison.json').write_text(json.dumps(dict(status='complete_accepted',values=values,guards=guards,candidate=guards['pass_all'],baseline_reproduction_exact=True,limitation='one development seed; conditional second seed required; no automatic next experiment'),indent=2)+'\n')
if __name__=='__main__':main()
