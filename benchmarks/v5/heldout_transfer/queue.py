"""CPU remote gates/scores as each frozen local heldout training completes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .evaluate import METHODS,sources
from benchmarks.v5.fair_comparison.audit import digest,write


def read(path):return json.loads(Path(path).read_text())


def observe(path):
    try:return read(path)
    except json.JSONDecodeError:return None


def prepare(a):
    integration=read(a.integration_gate/'summary.json');meta=read(a.integration_gate/'run.json');feature=read(a.feature_gate/'summary.json')
    if (integration['status']!='complete' or not integration['gate'] or not integration['checkpoint_gate'] or
        not integration['GT_forbidden_input_API'] or not integration['query_maps_zero'] or
        not integration['saved_count_replay'] or any(integration['native_cached_errors'])):
        raise ValueError('actual GT-free remote checkpoint interface gate required')
    if feature['status']!='passed_feature_wrapper_gate' or {v['method'] for v in feature['results']}!={'sam_ft','wsi_sam_ft'}:
        raise ValueError('both actual encoder feature gates required')
    for evidence in (meta,feature):
        for path,sha in evidence['source_sha256'].items():
            if digest(path)!=sha:raise ValueError('gated remote source changed '+path)
    original=read(a.producer/'jobs.json');jobs=[]
    for job in original:
        if job['method'] not in METHODS:raise ValueError('undeclared method')
        tasks=Path(job['heldout_remote']);contract=read(tasks/'contract.json');plans=read(tasks/'validation_maximum.json')
        if contract['training_protocol_sha256']!=job['protocol_sha256'] or digest(tasks/'validation_maximum.json')!=contract['tasks_sha256']:
            raise ValueError('frozen remote/training identity')
        jobs.append(job|dict(remote_expected_rows=4*len(plans),remote_gate_rows=4*len({(p['dataset'],p['kind']) for p in plans}),
            remote_protocol_sha256=digest(tasks/'contract.json')))
    if len(jobs)!=9 or len({j['key'] for j in jobs})!=9:raise ValueError('all9 remote conditions required')
    write(a.output/'jobs.json',jobs);return jobs


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);active={};done={};failed=[];started=time.time()
    snapshot=sources();snapshot[str(Path(__file__))]=digest(__file__)
    write(a.output/'contract.json',dict(coordinator_pid=os.getpid(),producer=str(a.producer),producer_pid=a.producer_pid,
        producer_jobs_sha256=digest(a.producer/'jobs.json'),source_sha256=snapshot,max_workers=a.max_workers,GPU_allocations=0,
        policy='For each completed seen-selected local training: same-weight CPU remote gate then all fixed tasks, no R-supervised updates or retries',
        initialization_scope='Current excluded-label adaptation heldout; public priors/image morphology remain qualified',
        integration_gate=str(a.integration_gate),feature_gate=str(a.feature_gate),formal_results_require_new_weight_gate=True))
    def locked():
        for path,sha in snapshot.items():
            if digest(path)!=sha:raise ValueError('locked remote source changed '+path)
    def state(status,**extra):
        write(a.output/'status.json',dict(status=status,coordinator_pid=os.getpid(),workers=[v[5] for v in active.values()],
            complete=done,failed=failed,seconds=time.time()-started,**extra))
    try:
        jobs=prepare(a);locked()
        if a.prepare_only:
            write(a.output/'summary.json',dict(status='prepared_remote_queue',jobs=len(jobs),GPU_allocations=0,jobs_sha256=digest(a.output/'jobs.json')))
            state('prepared_remote_queue');return
        pending={j['key']:j for j in jobs};waiting=[];stamp='_'.join(a.output.name.rsplit('_',2)[-2:])
        def launch(job,training,stage):
            locked();destination=a.output/f'{stage}_{job["fold"]}_{job["method"]}_{stamp}';log=Path(str(destination)+'.log')
            cmd=[sys.executable,'-u','-m','benchmarks.v5.heldout_transfer.evaluate','--tasks',job['heldout_remote'],
                '--training-tasks',job['tasks'],'--local-test',job['heldout_local'],'--training',training,'--realized',job['realized'],
                '--method',job['method'],'--output',str(destination)]
            if stage=='gate':cmd.append('--gate')
            else:cmd+=['--verified-gate',done[job['key']+'|gate']]
            handle=log.open('x');env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
            meta=dict(job=job['key'],stage=stage,worker_pid=proc.pid,output=str(destination),log=str(log),command=cmd)
            active[job['key']]=(proc,handle,job,training,stage,meta);print(json.dumps(meta),flush=True)
        while pending or waiting or active:
            locked();producer=observe(a.producer/'status.json')
            if producer is None:
                proc=Path(f'/proc/{a.producer_pid}/cmdline')
                if not proc.exists() or str(a.producer) not in proc.read_bytes().replace(b'\0',b' ').decode():
                    raise ValueError('producer missing during transient snapshot read')
                state('waiting_live_producer_snapshot',producer_pid=a.producer_pid);time.sleep(1);continue
            proc=Path(f'/proc/{a.producer_pid}/cmdline')
            cmd=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
            alive='benchmarks.v5.heldout_comparison.pipeline' in cmd and str(a.producer) in cmd
            if not alive and producer['status'] not in ('complete_local_execution','failed'):
                raise ValueError('local training coordinator missing before terminal evidence')
            for key,job in list(pending.items()):
                train=producer.get('complete',{}).get(key+'|train')
                if train:
                    waiting.append((job,train,'gate'));del pending[key]
                elif producer['status']=='failed' or any(v.get('job')==key and v.get('stage') in ('gate','gate_score','train') for v in producer.get('failed',[])):
                    failed.append(dict(job=key,stage='dependency',reason='local condition did not complete training'));del pending[key]
            if pending and producer['status']=='complete_local_execution':raise ValueError('completed producer missing declared training conditions')
            for key,(child,handle,job,training,stage,meta) in list(active.items()):
                code=child.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del active[key]
                if code:
                    failed.append(meta|dict(exit_code=code));continue
                destination=Path(meta['output']);result=read(destination/'summary.json');run_meta=read(destination/'run.json')
                expected=job['remote_gate_rows'] if stage=='gate' else job['remote_expected_rows']
                if (result['status']!='complete' or result['checkpoint_gate'] or result['rows_per_method']!=expected or
                    run_meta['protocol_sha256']!=job['remote_protocol_sha256']):raise ValueError('remote child scope/coverage mismatch')
                if stage=='gate' and (any(result['native_cached_errors']) or not result['saved_count_replay']):raise ValueError('new-weight remote gate failed')
                for folder in result['outputs'].values():
                    if sum(1 for _ in (Path(folder)/'rows.jsonl').open())!=expected:raise ValueError('remote output row count')
                done[key+'|'+stage]=str(destination)
                if stage=='gate':waiting.insert(0,(job,training,'score'))
            while waiting and len(active)<a.max_workers:launch(*waiting.pop(0))
            state('running_or_waiting_local_training',pending_conditions=len(pending),ready_conditions=len(waiting),producer_pid=a.producer_pid)
            time.sleep(5)
        if failed:raise RuntimeError('some remote conditions failed; independent outputs preserved; no retry')
        if len(done)!=18:raise ValueError('nine remote gate/full condition coverage incomplete')
        write(a.output/'summary.json',dict(status='complete_remote_execution',complete=done,pending='Independent full GT/point/selection/mask audit and paired statistics'))
        state('complete_remote_execution')
    except Exception:
        for child,handle,job,training,stage,meta in active.values():
            if child.poll() is None:child.terminate()
            code=child.wait();handle.write(f'\nEXIT_CODE={code}\n');handle.close()
        state('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('producer','integration-gate','feature-gate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--producer-pid',type=int,required=True);p.add_argument('--max-workers',type=int,default=2)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.max_workers not in (1,2):raise SystemExit('CPU workers limited to1/2')
    run(a)
