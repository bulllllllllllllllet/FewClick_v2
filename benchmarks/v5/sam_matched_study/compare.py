"""Pair native-SAM bridge with accepted FewClick under identical episode supervision."""
import argparse,json
from pathlib import Path
import torch
from benchmarks.v5.encoder_research.continuous_data import sha,TASKS
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.compare_continuous_pilots import paired_interval

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;torch.set_num_threads(2);s=json.loads((r/'settings.json').read_text());c=json.loads((r/'contract.json').read_text());allrows={};evidence={}
 for name,folder in [('sam_native',r/'sam_native'),('fewclick',Path(s['fewclick_reference']))]:
  proof=json.loads((folder/'independent_acceptance.json').read_text());report=json.loads((folder/'report.json').read_text());assert proof['status']=='accepted_P3_spatial_evaluation' and not proof['smoke'];assert report['updates']==1000 and report['consumed_tasks']==4000 and report['seed']==s['seed']
  for f,k in [('report.json','report_sha256'),('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256')]:assert sha(folder/f)==proof[k]
  assert sha(folder/'training_consumed.jsonl')==c['consumption_sha256'];assert report['protocol_sha256']==c['protocol_sha256']
  ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in ck['model'].values());assert all(int(v['step'])==1000 for v in ck['optimizer']['state'].values());del ck
  allrows[name]=json.loads((folder/'rows.json').read_text());evidence[name]={f:sha(folder/f) for f in ['independent_acceptance.json','report.json','rows.json','masks.npz','training_consumed.jsonl']}
 flat={};values={}
 for name,rows in allrows.items():flat[name],_,values[name]=summarize_arm(rows)
 guards=compare_rows(allrows['sam_native'],allrows['fewclick']);_,inventory=load_cell_catalog(TASKS)
 intervals={scope:paired_interval(flat['sam_native'],flat['fewclick'],inventory,scope,'dice',True,True) for scope in ['local','remote']}
 result=dict(status='accepted_episode_matched_SAM_bridge_comparison',values=values,FewClick_minus_SAM_guards=guards,conditional_intervals=intervals,evidence=evidence,scope='one seed; exposed development; same support clicks, three input views, 1000x4 episodes, frozen encoders, common data-fit losses; distinct public pretraining/capacity/native preprocessing/model-specific regularization; not equal compute or universally optimized SAM; single mask token, no GT candidate selection')
 out=r/'comparison';out.mkdir();(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
