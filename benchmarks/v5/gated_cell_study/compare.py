"""Bind final cell-on and reusable matched cell-off independent receipts."""
import argparse,json
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));paths={'cell_off':Path(c['baseline']),'cell_on':r/'control'};rows={}
 for arm,path in paths.items():
  proof=json.load(open(path/'independent_acceptance.json'))
  for name,key in [('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256'),('report.json','report_sha256')]:assert sha(path/name)==proof[key]
  report=json.load(open(path/'report.json'));assert report['seed']==c['seed'] and report['updates']==2000
  rows[arm]=json.load(open(path/'rows.json'));assert len(rows[arm])==3711
 assert (paths['cell_on']/'training_consumed.jsonl').read_bytes()==(paths['cell_off']/'training_consumed.jsonl').read_bytes()
 assert all({k:a[k] for k in ['scope','budget','plan']}=={k:b[k] for k in ['scope','budget','plan']} for a,b in zip(rows['cell_off'],rows['cell_on']))
 guards=compare_rows(rows['cell_off'],rows['cell_on'])
 (r/'comparison.json').write_text(json.dumps(dict(status='complete_accepted_matched_cell_comparison',values={k:summarize_arm(v)[2] for k,v in rows.items()},guards=guards,candidate=guards['pass_all'],limitation='single existing seed, exposed spatial development; complete original cell input versus none, not embedding-only or molecular objective causality'),indent=2))
if __name__=='__main__':main()
