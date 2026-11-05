"""Score saved seed2 step1000; verify both endpoints before budget comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from benchmarks.v5.size_aware_study.model import DecoderData,DecoderModel
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--accept',action='store_true');a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for path,h in c['sources'].items():assert sha(path)==h
 data=DecoderData(Path(c['cache']),'control',c['seed']);tasks=[json.loads(x) for x in (Path(c['protocol'])/'evaluation.jsonl').read_text().splitlines()];out=r/'endpoint1000'
 if a.accept:
  proof=json.load(open(r/'control/independent_acceptance.json'));assert proof['rows_sha256']==sha(r/'control/rows.json')
  rows=json.load(open(out/'rows.json'));assert len(rows)==len(tasks)==3711
  with np.load(out/'masks.npz') as f:
   for task,row,packed in zip(tasks,rows,f['masks']):
    assert all(task[k]==row[k] for k in ['scope','budget','plan']);gt,mpp=data.labels(task['plan']['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);assert score(mask,gt==task['plan']['target'],gt!=255,mpp)==row['metrics']
  report=json.load(open(out/'report.json'));assert report['fresh_replay']==12 and report['checkpoint_sha256']==sha(r/'control/checkpoint1000.pt')
  final=json.load(open(r/'control/rows.json'));guards=compare_rows(rows,final);first=json.load(open(Path(c['seed1'])/'comparison.json'))
  result=dict(status='complete_accepted',seed=c['seed'],values={'1000':summarize_arm(rows)[2],'2000':summarize_arm(final)[2]},guards=guards,all_guards_pass=guards['pass_all'],positive_primary_direction=summarize_arm(final)[2]['remote_macro_Dice']>summarize_arm(rows)[2]['remote_macro_Dice'],seed1_values=first['values'],limitation='two initialization seeds, identical task order, exposed development set; not independent data or method superiority',bindings={str(x):sha(x) for x in [r/'contract.json',out/'rows.json',out/'masks.npz',r/'control/checkpoint1000.pt',r/'control/checkpoint.pt',r/'control/independent_acceptance.json']})
  (r/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete_accepted','guards':guards['pass_all']}));return
 out.mkdir(exist_ok=False);m=DecoderModel('control',c['seed'],'cuda:0');state=torch.load(r/'control/checkpoint1000.pt',map_location='cuda:0',weights_only=True);assert state['updates']==1000;m.load_state_dict(state['model']);m.eval();del state;rows=[];masks=[];replay=[]
 with torch.no_grad():
  for task in tasks:
   logits=m(**data.model_inputs(task['plan'],'xcell','cuda:0'))['query']['pixel_logits'];gt,mpp=data.labels(task['plan']['query_id']);mask=(logits[0]>=0).cpu().numpy();rows.append(task|dict(metrics=score(mask,gt==task['plan']['target'],gt!=255,mpp)));masks.append(np.packbits(mask))
   if len(replay)<12:replay.append(logits.cpu())
   if len(rows)%25==0:print(json.dumps({'stage':'endpoint1000_scoring','rows':len(rows),'total':3711}),flush=True)
 del m;torch.cuda.empty_cache();m=DecoderModel('control',c['seed'],'cuda:0');m.load_state_dict(torch.load(r/'control/checkpoint1000.pt',map_location='cuda:0',weights_only=True)['model']);m.eval()
 with torch.no_grad():
  for task,expected in zip(tasks,replay):assert torch.equal(m(**data.model_inputs(task['plan'],'xcell','cuda:0'))['query']['pixel_logits'].cpu(),expected)
 (out/'rows.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(out/'masks.npz',masks=np.stack(masks));(out/'report.json').write_text(json.dumps(dict(checkpoint_sha256=sha(r/'control/checkpoint1000.pt'),fresh_replay=12))+'\n')
if __name__=='__main__':main()
