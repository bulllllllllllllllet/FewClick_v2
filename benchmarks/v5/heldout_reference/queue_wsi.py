"""Sequential all-label WSI-SAM feature reference, each subset gated first."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .wsi_remote import reference_sources
from benchmarks.v5.fair_comparison.audit import digest,write


def read(path):return json.loads(Path(path).read_text())


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);snapshot=reference_sources()|{str(Path(__file__)):digest(__file__)}
    manifest=read(a.protocols/'manifest.json');done={};failed=[];active=None;handle=None
    def locked():
        for p,h in snapshot.items():
            if digest(p)!=h:raise ValueError('locked reference source changed '+p)
    write(a.output/'contract.json',dict(coordinator_pid=os.getpid(),source_sha256=snapshot,protocols=str(a.protocols),
        manifest_sha256=digest(a.protocols/'manifest.json'),development=str(a.development),training=str(a.training),
        realized=str(a.realized),first_gate=str(a.first_gate),CPU_workers=1,GPU_allocations=0,
        scope='All-label G3 WSISAM custom two-view prototype reference; not heldout-trained accuracy or native official remote inference'))
    try:
        prior=read(a.first_gate/'summary.json');meta=read(a.first_gate/'run.json')
        if (prior['status']!='complete' or not prior['gate'] or not prior['all_label_reference'] or
            prior['checkpoint_gate'] or any(prior['native_cached_errors']) or not prior['saved_count_replay'] or
            meta['source_sha256']!=reference_sources() or meta['protocol_sha256']!=digest(Path(manifest['necrosis']['heldout_remote'])/'contract.json')):
            raise ValueError('accepted same-source all-label necrosis reference gate required')
        coverage=read(a.first_gate/'query_annotation_coverage.json')
        if not coverage or not all(isinstance(v,dict) and set(v)=={'valid_pixels','total_pixels'} for v in coverage.values()):
            raise ValueError('query coverage map schema invalid')
        done['necrosis|gate']=str(a.first_gate);stamp='_'.join(a.output.name.rsplit('_',2)[-2:])
        for fold,item in manifest.items():
            for stage in (('score',) if fold=='necrosis' else ('gate','score')):
                locked();destination=a.output/f'{stage}_{fold}_{stamp}';log=Path(str(destination)+'.log')
                cmd=[sys.executable,'-u','-m','benchmarks.v5.heldout_reference.wsi_remote','--tasks',item['heldout_remote'],
                    '--development',str(a.development),'--training',str(a.training),'--realized',str(a.realized),'--output',str(destination)]
                if stage=='gate':cmd.append('--gate')
                else:cmd+=['--verified-gate',done[fold+'|gate']]
                handle=log.open('x');env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                active=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
                while active.poll() is None:
                    locked();write(a.output/'status.json',dict(status='running',coordinator_pid=os.getpid(),worker_pid=active.pid,
                        fold=fold,stage=stage,command=cmd,output=str(destination),log=str(log),complete=done,failed=failed))
                    time.sleep(5)
                code=active.returncode;handle.write(f'\nEXIT_CODE={code}\n');handle.close();handle=None;active=None
                if code:
                    failed.append(dict(fold=fold,stage=stage,exit_code=code,log=str(log)));break
                result=read(destination/'summary.json')
                expected=4*len(read(Path(item['heldout_remote'])/'validation_maximum.json')) if stage=='score' else 4*len({(p['dataset'],p['kind']) for p in read(Path(item['heldout_remote'])/'validation_maximum.json')})
                if result['status']!='complete' or not result['all_label_reference'] or result['rows_per_method']!=expected:
                    raise ValueError('all-label reference scope/coverage mismatch')
                for folder in result['outputs'].values():
                    if sum(1 for _ in (Path(folder)/'rows.jsonl').open())!=expected:raise ValueError('reference row coverage')
                done[fold+'|'+stage]=str(destination)
        if failed:raise RuntimeError('reference subset failure; other subsets preserved; no retries')
        if len(done)!=6:raise ValueError('all three reference gate/full stages required')
        write(a.output/'summary.json',dict(status='complete_all_label_WSI_remote_execution',complete=done,
            pending='Independent full rows/GT/counts/point/weight audit; descriptor only, not new heldout counterpart performance'))
        write(a.output/'status.json',dict(status='complete_all_label_WSI_remote_execution',coordinator_pid=os.getpid(),complete=done))
    except Exception:
        if active is not None and active.poll() is None:active.terminate();active.wait()
        if handle is not None and not handle.closed:handle.write(f'\nEXIT_CODE={active.returncode if active else -1}\n');handle.close()
        write(a.output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),complete=done,failed=failed,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('protocols','development','training','realized','first-gate','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
