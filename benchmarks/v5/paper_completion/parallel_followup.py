"""Resume registered negative arm only after accepted no-match arm; no duplicate training."""
import argparse,json,os,signal,time,hashlib
from pathlib import Path
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;study=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_completion_dirfix_1106_193345')
 def compare(arm):
  full=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/support_region_study_1106_162117/control/control');abl=study/arm/'control';rows={}
  for k,f in [('full',full),('ablated',abl)]:
   proof=json.load(open(f/'independent_acceptance.json'))
   for name,key in [('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256'),('report.json','report_sha256')]:
    h=hashlib.sha256()
    with open(f/name,'rb') as s:
     for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==proof[key]
   rows[k]=json.load(open(f/'rows.json'));assert len(rows[k])==3711
  assert (full/'training_consumed.jsonl').read_bytes()==(abl/'training_consumed.jsonl').read_bytes()
  assert all({k:a[k] for k in ['scope','budget','plan']}=={k:b[k] for k in ['scope','budget','plan']} for a,b in zip(rows['full'],rows['ablated']))
  (r/(arm+'_comparison.json')).write_text(json.dumps(dict(status='complete_accepted_matched_ablation',values={k:summarize_arm(v)[2] for k,v in rows.items()},full_minus_ablated_guards=compare_rows(rows['ablated'],rows['full']),limitation='one fixed seed; no architecture claim from unequal budgets'),indent=2))
 for arm in ['no_match','no_negative']:
  while True:
   s=json.load(open(study/arm/'stage.json'));assert s['status']!='failed',s
   if s['status']=='complete_accepted':break
   (r/'stage.json').write_text(json.dumps(dict(status='waiting_registered_arm',arm=arm,controller_pid=os.getpid())));time.sleep(15)
  compare(arm)
  if arm=='no_match':
   for pid,module in [(1716208,'pipeline'),(1753449,'train')]:
    cmd=Path('/proc')/str(pid)/'cmdline';assert cmd.exists() and ('benchmarks.v5.paper_completion.'+module).encode() in cmd.read_bytes()
   for pid in [1753449,1716208]:os.kill(pid,signal.SIGCONT)
 (r/'stage.json').write_text(json.dumps(dict(status='complete_accepted_both_ablation_comparisons')))
if __name__=='__main__':main()
