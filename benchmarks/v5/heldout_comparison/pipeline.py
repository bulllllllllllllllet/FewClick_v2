"""Run nine heldout conditions after full coordinates and owned GPU dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .audit import FOLDS
from .training import METHODS
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.experiment_orchestration.mechanism_queue import occupied_gpus

STAGES=('gate','gate_score','train','score')
RAW={4:('raw_history02_1025_234700',3914383),5:('raw_history03_1025_234700',3914488)}


def read(path):return json.loads(Path(path).read_text())


def prepare(a):
    final=read(a.points/'summary.json');audit=read(a.points/'full_audit/summary.json')
    if final['status']!='complete_coordinates_and_protocol_audits' or digest(a.points/'full_audit/summary.json')!=final['full_audit_sha256']:
        raise ValueError('complete independently audited full coordinates required')
    if audit['status']!='passed_prepared_protocol_audit' or set(audit['folds'])!=set(FOLDS):raise ValueError('all folds required')
    gate=read(a.cpu_gate/'gate_acceptance.json')
    if gate['status']!='passed_checkpoint_binding_gate':raise ValueError('actual checkpoint CPU gate required')
    for path,sha in gate['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('CPU-gated source changed '+path)
    manifest=read(a.protocols/'manifest.json');jobs=[];stamp='_'.join(a.output.name.rsplit('_',2)[-2:])
    for fold in FOLDS:
        item=manifest[fold];tasks=Path(item['tasks']);actual=a.points/'full'/fold
        contract=read(tasks/'contract.json');coordinates=read(actual/'summary.json')
        if contract['heldout_fold']!=fold or contract['heldout_excluded_labels']!=FOLDS[fold]['exclude']:raise ValueError('fold definition changed')
        if not coordinates['complete_protocol'] or coordinates['tasks']!=40000 or not audit['folds'][fold]['complete_actual_protocol']:
            raise ValueError('incomplete actual fold tasks')
        sha=digest(actual/'training_plans.jsonl')
        if sha!=coordinates['training_plans_sha256'] or sha!=audit['folds'][fold]['actual_plans_sha256']:
            raise ValueError('actual coordinate SHA changed')
        if coordinates['protocol_sha256']!=digest(tasks/'contract.json'):raise ValueError('actual fold protocol binding')
        by_domain={d:[] for d in ('crc','tiger')}
        for line in (actual/'training_plans.jsonl').open():
            plan=json.loads(line);domain=plan['dataset']
            if len(by_domain[domain])<4:by_domain[domain].append(line)
            if all(len(v)==4 for v in by_domain.values()):break
        if not all(len(v)==4 for v in by_domain.values()):raise ValueError('two-domain gate coverage')
        target=a.output/f'gate_realized_{fold}_{stamp}';target.mkdir()
        lines=[by_domain[d][i] for i in range(4) for d in ('crc','tiger')]
        (target/'training_plans.jsonl').write_text(''.join(lines))
        write(target/'summary.json',coordinates|dict(tasks=8,counts=dict(crc=4,tiger=4),gate=True,complete_protocol=False,
            training_plans_sha256=digest(target/'training_plans.jsonl'),selection='First4 per domain, interleaved; unchanged actual task bytes'))
        for method in METHODS:
            jobs.append(dict(key=fold+'|'+method,fold=fold,method=method,tasks=str(tasks),realized=str(actual),gate_realized=str(target),
                heldout_local=item['heldout_local'],heldout_remote=item['heldout_remote'],training_plans_sha256=sha,
                protocol_sha256=digest(tasks/'contract.json'),expected_rows=4*len(read(Path(item['heldout_local'])/'validation_maximum.json'))))
    write(a.output/'jobs.json',jobs)
    return jobs


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();active={};slots={};done={};failed=[]
    files=list(Path(__file__).parent.glob('*.py'))+list(Path('benchmarks/v5/fair_comparison').glob('*.py'))
    files += [Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/multidomain_training/losses.py'),
        Path('benchmarks/v5/experiment_orchestration/mechanism_queue.py')]
    snapshot={str(p):digest(p) for p in files}
    write(a.output/'contract.json',dict(coordinator_pid=os.getpid(),source_sha256=snapshot,protocols=str(a.protocols),points=str(a.points),
        point_worker=a.point_worker,CPU_gate=str(a.cpu_gate),CPU_gate_sha256=digest(a.cpu_gate/'gate_acceptance.json'),
        GPU_slots=[4,5],GPU_dependencies={str(g):dict(output=str(a.generic_root/f),worker_pid=p) for g,(f,p) in RAW.items()},
        methods=METHODS,folds=FOLDS,stages=STAGES,seed=20261025,steps=10000,warmup=1000,accumulate=4,
        policy='Native4 updates and gate-checkpoint terminal scorer -> NEW initialization full train -> seen-best full local test; no retries',
        scope='Single-seed no-excluded-label adaptation; public/image priors may expose morphology; remote scores/independent final statistics pending'))
    def locked():
        for path,sha in snapshot.items():
            if digest(path)!=sha:raise ValueError('heldout queue source changed '+path)
    def state(status,**extra):
        write(a.output/'status.json',dict(status=status,coordinator_pid=os.getpid(),workers=list(slots.values()),complete=done,failed=failed,**extra))
    try:
        while not (a.points/'summary.json').exists():
            locked();status=read(a.points/'status.json')
            if status['status']=='failed':raise ValueError('full actual point dependency failed')
            proc=Path(f'/proc/{a.point_worker}/cmdline')
            if not proc.exists():raise ValueError('point coordinator missing before completion')
            cmd=proc.read_bytes().replace(b'\0',b' ').decode()
            if 'benchmarks.v5.heldout_comparison.point_pipeline' not in cmd or str(a.points) not in cmd:
                raise ValueError('owned point PID identity mismatch')
            state('waiting_live_full_coordinates',point_worker=a.point_worker);time.sleep(5)
        while not Path(str(a.points)+'.log').read_text().rstrip().endswith('EXIT_CODE=0'):
            if 'EXIT_CODE=' in Path(str(a.points)+'.log').read_text():raise ValueError('point coordinator failed')
            locked();state('waiting_point_exit_evidence');time.sleep(2)
        jobs=prepare(a);locked()
        if a.prepare_only:
            write(a.output/'summary.json',dict(status='prepared_execution_plan',jobs=len(jobs),GPU_allocations=0,
                jobs_sha256=digest(a.output/'jobs.json'),scope='Actual native gate manifests/9job plan; no GPU or model training'))
            state('prepared_execution_plan');return
        pending=list(jobs);slots={g:dict(gpu=g,status='waiting_raw_and_GPU') for g in RAW}
        def ready(gpu):
            name,pid=RAW[gpu];folder=a.generic_root/name;summary=folder/'summary.json'
            if not summary.exists():
                proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists():raise ValueError('raw dependency ended without summary')
                cmd=proc.read_bytes().replace(b'\0',b' ').decode()
                if 'benchmarks.v5.generic_cell_control.history_extract' not in cmd or str(folder) not in cmd:
                    raise ValueError('raw dependency PID identity mismatch')
                return False
            result=read(summary)
            if result['status']!='complete' or result['gate']:raise ValueError('full raw dependency not complete')
            log=Path(str(folder)+'.log').read_text().rstrip()
            if 'EXIT_CODE=' in log and not log.endswith('EXIT_CODE=0'):raise ValueError('raw dependency nonzero exit')
            if not log.endswith('EXIT_CODE=0'):return False
            proc=Path(f'/proc/{pid}/cmdline')
            if not proc.exists():return True
            cmd=proc.read_bytes().replace(b'\0',b' ').decode()
            return not ('benchmarks.v5.generic_cell_control.history_extract' in cmd and str(folder) in cmd)
        def launch(gpu,job,stage):
            locked();destination=a.output/f'{stage}_{job["fold"]}_{job["method"]}_{a.output.name.rsplit("_",2)[-2]}_{a.output.name.rsplit("_",1)[-1]}'
            if stage in ('gate','train'):
                cmd=[sys.executable,'-u','-m','benchmarks.v5.heldout_comparison.training','--tasks',job['tasks'],
                    '--realized',job['gate_realized'] if stage=='gate' else job['realized'],'--method',job['method'],
                    '--seed','20261025','--output',str(destination)]
                if stage=='gate':cmd.append('--gate')
            else:
                train=done[job['key']+'|'+('gate' if stage=='gate_score' else 'train')]
                cmd=[sys.executable,'-u','-m','benchmarks.v5.heldout_comparison.evaluate','--training-tasks',job['tasks'],
                    '--training-run',train,'--tasks',job['heldout_local'],'--method',job['method'],'--output',str(destination)]
                if stage=='gate_score':cmd.append('--gate')
            log=Path(str(destination)+'.log');handle=log.open('x')
            env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(proc,handle,job,stage,destination,log)
            slots[gpu]=dict(gpu=gpu,status='running',job=job['key'],stage=stage,worker_pid=proc.pid,output=str(destination),log=str(log),command=cmd)
            print(json.dumps(slots[gpu]),flush=True)
        waiting={}
        while pending or active or waiting:
            locked()
            for gpu,(proc,handle,job,stage,destination,log) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del active[gpu]
                if code:
                    failed.append(dict(job=job['key'],stage=stage,exit_code=code,log=str(log)));slots[gpu]=dict(gpu=gpu,status='failed',job=job['key']);continue
                result=read(destination/'summary.json')
                if result['status']!='complete':raise ValueError('child incomplete summary')
                if stage=='gate':
                    if not all(result[k] for k in ('fresh_gate_resume','checkpoint_reload_exact','nonzero_joint_backbone_gradient')):
                        raise ValueError('native GPU gate incomplete')
                    if result['training_consumed_sha256']!=digest(Path(job['gate_realized'])/'training_plans.jsonl'):raise ValueError('native actual task mismatch')
                if stage=='train':
                    if result['step']!=10000 or not result['full_manifest_parity'] or result['training_consumed_sha256']!=job['training_plans_sha256']:
                        raise ValueError('full training budget/coordinates mismatch')
                if stage in ('gate_score','score'):
                    expected=job['expected_rows'] if stage=='score' else 4*len({p['dataset'] for p in read(Path(job['heldout_local'])/'validation_maximum.json')})
                    if result['rows']!=expected or sum(1 for _ in (destination/'rows.jsonl').open())!=expected:raise ValueError('terminal score coverage')
                done[job['key']+'|'+stage]=str(destination)
                index=STAGES.index(stage)
                if index+1<len(STAGES):waiting[gpu]=(job,STAGES[index+1])
                else:slots[gpu]=dict(gpu=gpu,status='complete',job=job['key'])
            occupied=occupied_gpus()
            for gpu in RAW:
                if gpu in active:continue
                if gpu not in waiting and pending:waiting[gpu]=(pending.pop(0),'gate')
                if gpu not in waiting:continue
                job,stage=waiting[gpu]
                if not ready(gpu):slots[gpu]=dict(gpu=gpu,status='waiting_live_raw',job=job['key'],next_stage=stage);continue
                if gpu in occupied:slots[gpu]=dict(gpu=gpu,status='waiting_GPU_idle',job=job['key'],next_stage=stage);continue
                launch(gpu,job,stage);del waiting[gpu]
            state('running_or_waiting',pending_jobs=len(pending),seconds=time.time()-started);time.sleep(5)
        if failed:raise RuntimeError('heldout conditions failed; no retries; independent outputs preserved')
        if len(done)!=len(jobs)*len(STAGES):raise ValueError('nine-condition stage coverage incomplete')
        write(a.output/'summary.json',dict(status='complete_local_execution',complete=done,
            pending='Independent full training/GT/selection/mask audit and statistics; separate heldout remote evaluation; not overall goal completion'))
        state('complete_local_execution')
    except Exception:
        for gpu,(proc,handle,job,stage,destination,log) in active.items():
            if proc.poll() is None:proc.terminate()
            code=proc.wait();handle.write(f'\nEXIT_CODE={code}\n');handle.close()
        state('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('protocols','points','cpu-gate','generic-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--point-worker',type=int,required=True);p.add_argument('--prepare-only',action='store_true');run(p.parse_args())
