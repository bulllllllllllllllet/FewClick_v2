"""Readonly receipt after the already-authorized final WSI-SAM arm completes."""
import argparse,hashlib,json,os,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt-root',required=True,type=Path);r=p.parse_args().receipt_root;base=r.parent;active=base/'wsi_sparse_bridge_1109_035845'
 while True:
  state=json.load(open(active/'stage.json'))
  if state['status'] in ['complete_accepted','partial_or_failed_stop']:break
  time.sleep(15)
 files={}
 for root in [base/'sam_matched_adaptation_1108_142826',base/'wsi_author_freeze_1108_204150',active]:
  arm=root/'wsi_sam_ft';evaluation=arm/'evaluation'
  if (evaluation/'acceptance.json').exists():
   acc=json.load(open(evaluation/'acceptance.json'));assert acc['status']=='complete_accepted';assert sha(evaluation/'report.json')==acc['report_sha256']
  for path in [root/'contract.json',arm/'formal/checkpoint.pt',arm/'formal/acceptance.json',evaluation/'report.json',evaluation/'acceptance.json']:
   if path.exists():files[str(path)]=sha(path)
 receipt=dict(status='frozen_accepted' if state['status']=='complete_accepted' else 'frozen_attempt_with_failure',worker_pid=os.getpid(),final_candidate_status=state['status'],files_sha256=files,no_further_WSI_training_or_bridge_candidates=True,retained_all_variants=True)
 with (r/'wsi_freeze_manifest.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=receipt['status'],receipt=str(r/'wsi_freeze_manifest.json'))),flush=True)
if __name__=='__main__':main()
