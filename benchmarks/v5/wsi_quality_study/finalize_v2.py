"""Accept completed shards with explicit empty-component scoring fix; no inference rerun."""
import argparse,json,time,subprocess,sys,hashlib
from pathlib import Path
from .prepare import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    while True:
        stage=json.loads((r/'stage.json').read_text())
        if stage['status'] in ['failed','complete_accepted']:break
        time.sleep(5)
    launches=json.loads((r/'launch.json').read_text());assert all(x.get('exit_code')==0 for x in launches), 'inference failed: no scoring repair allowed'
    if stage['status']=='failed':
        log=(r/'collect.log').read_text();assert 'TypeError' in log and 'bitwise_and' in log, 'unexpected failure must be investigated'
    amendment=json.loads((r/'collector_amendment.json').read_text());assert hashlib.sha256(Path(amendment['new_path']).read_bytes()).hexdigest()==amendment['new_sha256']
    dump(r/'finalize_v2.json',dict(status='independent_acceptance_running',amendment=amendment))
    with (r/'collect_v2.log').open('x') as f:
        rc=subprocess.call([sys.executable,'-m','benchmarks.v5.wsi_quality_study.collect_v2','--root',str(r)],stdout=f,stderr=subprocess.STDOUT);f.write(f'\nEXIT_CODE={rc}\n')
    dump(r/'finalize_v2.json',dict(status='complete_accepted' if rc==0 else 'failed',exit_code=rc,acceptance=str(r/'acceptance_v2.json'),amendment=amendment));assert rc==0
if __name__=='__main__':main()
