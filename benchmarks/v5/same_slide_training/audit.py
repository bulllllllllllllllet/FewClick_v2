"""Independent manifest audit; no model training or inference."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;a=json.load(open(r/'acceptance.json'));c=json.load(open(r/'contract.json'));assert a['status']=='accepted_same_slide_training_queue';assert sha(r/'training_consumption.jsonl')==a['training_consumption_sha256'];lines=(r/'training_consumption.jsonl').read_text().splitlines(keepends=True);rows=list(map(json.loads,lines));assert len(rows)==9500;local=[i for i,t in enumerate(rows) if t['support_id']==t['query_id']];remote=[t for t in rows if t['support_id']!=t['query_id']];assert len(local)==4000 and len(remote)==5500
 oldlines=Path(c['source']).read_text().splitlines(keepends=True);old=list(map(json.loads,oldlines));assert Counter(lines[i] for i in local)==Counter([l for l,t in zip(oldlines,old) if t['support_id']==t['query_id']]*2)
 assert Counter(t['kind'] for t in remote)=={'same_present':5000,'same_absent':500};assert len({(t['support_id'],t['query_id'],t['target']) for t in remote})==5500
 inputs=json.load(open(r/'input_manifest.json'))
 for t in remote:
  s,q=inputs[t['support_id']],inputs[t['query_id']];assert s['split']==q['split']=='train';assert s['space_key']==q['space_key']==t['query_space']==t['source_space'];assert not overlap(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um']);assert t['query_has_human_clicks'] is False
  assert (t['query_positive_pixels']>0)==(t['kind']=='same_present');assert len(t['positive_xy'])==t['positive_count'] and len(t['negative_xy'])==t['negative_count']
 gate=json.load(open(r/'functional_tasks.json'));monitor=json.load(open(r/'monitor_tasks.json'));assert gate==monitor and len(gate)==32;assert Counter((t['dataset'],t['kind']) for t in gate)=={('crc','local'):8,('tiger','local'):8,('crc','same_present'):8,('crc','same_absent'):8}
 encoded={json.dumps(t,sort_keys=True) for t in rows};assert all(json.dumps(t,sort_keys=True) in encoded for t in gate)
 result=dict(status='complete_accepted_manifest_audit',training_consumption_sha256=sha(r/'training_consumption.jsonl'),functional_tasks_sha256=sha(r/'functional_tasks.json'),monitor_tasks_sha256=sha(r/'monitor_tasks.json'),tasks=9500,local=4000,remote_present=5000,remote_absent=500,remote_unique_slide_count=len({t['query_space'] for t in remote}),unique_input_patches=len(inputs),remote_class_counts={str(k):v for k,v in Counter((t['target'],t['kind']) for t in remote).items()},training_updates=2375,batch=4,training_started=False,source_code_sha256=sha(__file__))
 (r/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
