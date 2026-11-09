"""Read-only spatial/data-lineage audit of completed SANSA adaptation."""
import json,hashlib,datetime
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS
B=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');R=B/'sansa_adaptation_1109_043250';E=R/'sansa_ft/evaluation'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def overlap(a,b):return (np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])>0).all(1)
def main():
 _,rows=load_cell_catalog(TASKS);orig=dict(rows);rows.update(json.load(open(E/'query_rows.json')));c=json.load(open(R/'contract.json'));formal=R/'sansa_ft/formal';consumed=[json.loads(s) for s in (formal/'training_consumed.jsonl').read_text().splitlines()];functional=json.load(open(R/'sansa_ft/functional/contract.json'))['tasks'];plans=consumed+functional;train_ids={t[k] for t in plans for k in ['support_id','query_id']};assert all(rows[k]['split']=='train' for k in train_ids)
 tasks=json.load(open(E/'tasks.json'));eval_ids={t[k] for t in tasks for k in ['support_id','query_id']};assert not train_ids & eval_ids
 train=defaultdict(list)
 for k,v in orig.items():
  if v['split']=='train':train[v['space_key']].append(json.loads(v['multiscale_json'])['context_um'])
 train={k:np.array(v,float) for k,v in train.items()};violations=[];fine_violations=[];geometry_errors=[]
 def geom(k):
  row=rows[k];g=json.loads(row['multiscale_json']);mx,my=g['native_mpp'];x,y,w,h=g['native_boxes'][-1];actual=np.array([x*mx,y*my,(x+w)*mx,(y+h)*my]);declared=np.array(g['context_um']);
  if row['dataset']=='crc' and not np.allclose(actual,declared,atol=1e-6):geometry_errors.append(k)
  return row,g,np.array(g['context_um'])
 for k in eval_ids:
  row,g,box=geom(k);tr=train.get(row['space_key'])
  if tr is not None and overlap(box,tr).any():violations.append(k)
 for t in tasks:
  sr,sg,sb=geom(t['support_id']);qr,qg,qb=geom(t['query_id']);assert sr['space_key']==qr['space_key'];assert not overlap(sb,qb[None]).any()
  assert len(t['positive_xy'])==3 and len(t['negative_xy'])==2 and not t['query_has_human_clicks']
 assert not violations and not geometry_errors
 # Both train-only gates and actual formal consumption are included in above union.
 assert sha(formal/'training_consumed.jsonl')==c['training_consumption_sha256']==sha(c['training_consumption'])
 evalaccept=json.load(open(E/'acceptance.json'));assert sha(E/'report.json')==evalaccept['report_sha256'];assert sha(E/'tasks.json')==evalaccept['tasks_sha256']
 weightsha=sha(formal/'checkpoint.pt');assert weightsha==json.load(open(formal/'acceptance.json'))['checkpoint_sha256']==json.load(open(E/'contract.json'))['checkpoints']['sansa_ft']['sha256']
 changed={p:dict(expected=h,current=sha(p)) for p,h in c['sources'].items() if sha(p)!=h}
 assert not changed
 output=dict(status='accepted_no_detected_target_data_spatial_leakage',formal_tasks=len(consumed),functional_tasks=len(functional),unique_training_patches=len(train_ids),evaluation_tasks=len(tasks),unique_evaluation_input_patches=len(eval_ids),training_evaluation_ID_overlap=0,evaluation_inputs_overlapping_any_train_full4x=len(violations),same_slide_support_query_full4x_overlap=0,CRC_declared_vs_native_full4x_geometry_errors=0,training_catalog_patch_count=sum(map(len,train.values())),same_slide_train_eval_shared_slides=len({rows[k]['space_key'] for k in train_ids}&{rows[k]['space_key'] for k in eval_ids}),consumption_sha256=sha(formal/'training_consumed.jsonl'),checkpoint_sha256=weightsha,sources_unchanged=True,threshold=.5,model_selection='fixed final2000, train-only monitors; code inspected',query_GT='scoring after probability; wrapper receives only RGB and support points; code inspected',episode_memory='official SANSA resets memory_bank in every batch episode forward; code inspected',limits=['spatial-block validation; same-slide residual dependence permitted','public SAM2/Universal pretraining image-level overlap not exhaustively provable from local weights','evaluation exposed development data; not a blind new-organ test','current report limitations template describes SignedMatcher/decoder training incorrectly; actual adapter-only training verified separately'])
 out=B/('sansa_leakage_audit_'+datetime.datetime.now().strftime('%m%d_%H%M%S'));out.mkdir();(out/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print(out);print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
