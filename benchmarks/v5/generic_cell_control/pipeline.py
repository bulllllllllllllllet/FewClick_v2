"""Chain full raw/PCA, assembled controls, native gates and matched training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .assemble import ARMS
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.experiment_orchestration.mechanism_queue import occupied_gpus


def read(path): return json.loads(Path(path).read_text())


def observe(path):
    try:return read(path)
    except json.JSONDecodeError:return None


def run(root,cpu_gate,output):
    output.mkdir(parents=True,exist_ok=False);started=time.time();completed={};active={};failed=[]
    stamp='_'.join(output.name.rsplit('_',2)[-2:]);slots={}
    files=list(Path(__file__).parent.glob('*.py'))+list(Path('benchmarks/v5/fair_comparison').glob('*.py'))
    files += [Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/multidomain_training/losses.py')]
    snapshot={str(p):digest(p) for p in files}
    try:
        gate=read(cpu_gate/'summary.json')
        if gate['status']!='passed_cpu_model_gate' or gate['source_sha256']!=digest(Path(__file__).with_name('gate_model.py')):
            raise ValueError('same-source CPU actual model gate required')
        if {r['arm'] for r in gate['records']}!=set(ARMS):raise ValueError('both fixed controls required')
        for record in gate['records']:
            for path,sha in record['prior']['generic_sources'].items():
                if digest(path)!=sha:raise ValueError('CPU-gated builder changed')
        pca=root/'pca_full_train_1025_234700';pca_job=read(root/'pca_job_1025_234700.json')
        if pca_job['source_sha256']!=digest(Path(__file__).with_name('fit.py')):raise ValueError('PCA source changed')
        raw_shards=[Path(p) for p in pca_job['raw_inputs']]
        base=root.parent/'protocol_1025_124208';tasks=base/'tasks_1025_125400';realized=base/'realized_train_1025_125928'
        write(output/'contract.json',dict(coordinator_pid=os.getpid(),source_sha256=snapshot,CPU_gate=str(cpu_gate),
            CPU_gate_sha256=digest(cpu_gate/'summary.json'),PCA_dependency=str(pca),raw_shards=list(map(str,raw_shards)),
            methods=ARMS,GPU_schedule={'1':ARMS[0],'3':ARMS[1]},seed=20261025,steps=10000,actual_tasks=40000,
            initialization='fresh COCO/new heads, no J5, same64 cell appearance interface',
            order='complete train PCA -> full assembly -> native GPU gate -> NEW full train -> selected checkpoint6040 scores',
            failure_policy='no retry; affected arm stops, independent arm continues',
            reporting='both normalization variants; no best-variant selection, independent audit/report pending'))
        def locked():
            for path,sha in snapshot.items():
                if digest(path)!=sha:raise ValueError('locked genericHE pipeline source changed '+path)
        while True:
            locked();state=observe(pca/'status.json')
            if state is None:time.sleep(.5);continue
            if state['status']=='failed':raise ValueError('formal PCA/raw dependency failed')
            if state['status']=='complete_train_fit' and pca.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'):break
            if not Path('/proc/'+str(state['worker_pid'])).exists():raise ValueError('formal PCA worker missing before terminal evidence')
            write(output/'status.json',dict(status='waiting_live_PCA',coordinator_pid=os.getpid(),PCA_pid=state['worker_pid'],observed_at=time.time()))
            time.sleep(5)
        projection=read(pca/'summary.json')
        if projection['status']!='complete_train_fit' or projection['gate'] or projection['validation_raw_files_opened_for_fit']!=0:
            raise ValueError('formal PCA scope mismatch')
        assembly=output/f'assembly_{stamp}';log=assembly.with_suffix('.log')
        command=[sys.executable,'-u','-m','benchmarks.v5.generic_cell_control.assemble','--base-tasks',str(tasks),
            '--base-realized',str(realized),'--pca',str(pca),'--raw-shards',*map(str,raw_shards),'--output',str(assembly)]
        locked()
        with log.open('x') as handle:
            env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            worker=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            while worker.poll() is None:
                write(output/'status.json',dict(status='assembling_full_controls',coordinator_pid=os.getpid(),worker_pid=worker.pid,log=str(log),command=command))
                time.sleep(5)
            handle.write(f'\nEXIT_CODE={worker.returncode}\n')
        if worker.returncode:raise ValueError('formal assembly failed; output preserved')
        assembled=read(assembly/'summary.json');manifest=read(assembly/'manifest.json')
        if assembled['status']!='complete' or assembled['gate'] or assembled['fields']!=1737:raise ValueError('formal assembly incomplete')
        completed['assembly']=str(assembly)
        gates={}
        for arm,paths in manifest.items():
            target=output/f'gate_realized_{arm}_{stamp}';target.mkdir()
            original=(Path(paths['realized'])/'training_plans.jsonl').read_text().splitlines(keepends=True)
            by_domain={d:[] for d in ('crc','tiger')}
            for line in original:
                plan=json.loads(line);domain=plan['dataset']
                if len(by_domain[domain])<4:by_domain[domain].append(line)
                if all(len(v)==4 for v in by_domain.values()):break
            if any(len(v)!=4 for v in by_domain.values()):raise ValueError('formal native gate coverage')
            lines=[by_domain[d][i] for i in range(4) for d in ('crc','tiger')]
            (target/'training_plans.jsonl').write_text(''.join(lines))
            summary=read(Path(paths['realized'])/'summary.json')
            write(target/'summary.json',summary|dict(training_plans_sha256=digest(target/'training_plans.jsonl'),tasks=8,gate=True))
            gates[arm]=target
            if digest(Path(paths['realized'])/'training_plans.jsonl')!='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe':raise ValueError('formal40k changed')
        todo={1:ARMS[0],3:ARMS[1]}
        def launch(gpu,arm,stage):
            locked();paths=manifest[arm];destination=output/f'{stage}_{arm}_{stamp}'
            if stage=='evaluate':
                train=Path(completed['train|'+arm]);best=read(train/'best_checkpoint.json')
                if digest(best['path'])!=best['checkpoint_sha256']:raise ValueError('formal selected weight changed')
                cmd=[sys.executable,'-u','-m','benchmarks.v5.generic_cell_control.evaluate','--tasks',paths['tasks'],
                    '--method',arm,'--checkpoint',best['path'],'--output',str(destination)]
            else:
                cmd=[sys.executable,'-u','-m','benchmarks.v5.generic_cell_control.train','--tasks',paths['tasks'],
                    '--realized',str(gates[arm]) if stage=='gate' else paths['realized'],'--method',arm,'--seed','20261025','--output',str(destination)]
                if stage=='gate':cmd.append('--gate')
            log=destination.with_suffix('.log');handle=log.open('x')
            env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(proc,handle,arm,stage,destination,log)
            slots[gpu]=dict(status='running',gpu=gpu,arm=arm,stage=stage,worker_pid=proc.pid,output=str(destination),log=str(log),command=cmd)
            print(json.dumps(slots[gpu]),flush=True)
        while todo or active:
            locked();occupied=occupied_gpus()
            for gpu,arm in list(todo.items()):
                if gpu not in occupied:launch(gpu,arm,'gate');del todo[gpu]
                else:slots[gpu]=dict(status='waiting_GPU_idle',gpu=gpu,arm=arm)
            for gpu,(proc,handle,arm,stage,destination,log) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del active[gpu]
                if code:
                    failed.append(dict(gpu=gpu,arm=arm,stage=stage,exit_code=code,log=str(log)));slots[gpu]=dict(status='failed',gpu=gpu,arm=arm,stage=stage);continue
                result=read(destination/'summary.json')
                if result['status']!='complete':raise ValueError('formal child summary incomplete')
                if stage=='gate':
                    if not all(result[k] for k in ('fresh_gate_resume','checkpoint_reload_exact','nonzero_joint_backbone_gradient')):raise ValueError('formal native gate failed')
                    if result['training_consumed_sha256']!=digest(gates[arm]/'training_plans.jsonl'):raise ValueError('native gate coordinate mismatch')
                if stage=='train':
                    if result['step']!=10000 or not result['full_manifest_parity'] or result['training_consumed_sha256']!='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe':raise ValueError('formal full budget/task mismatch')
                if stage=='evaluate':
                    if sum(1 for _ in (destination/'rows.jsonl').open())!=6040:raise ValueError('formal score coverage')
                completed[stage+'|'+arm]=str(destination)
                if stage=='gate':launch(gpu,arm,'train')
                elif stage=='train':launch(gpu,arm,'evaluate')
                else:slots[gpu]=dict(status='complete',gpu=gpu,arm=arm,stage=stage)
            write(output/'status.json',dict(status='running',coordinator_pid=os.getpid(),workers=list(slots.values()),complete=completed,failed=failed,seconds=time.time()-started))
            time.sleep(5)
        if failed:raise RuntimeError('one or more genericHE formal queues failed; no retries')
        if len(completed)!=7:raise ValueError('formal two-arm stage coverage incomplete')
        write(output/'summary.json',dict(status='complete_execution',coordinator_pid=os.getpid(),complete=completed,seconds=time.time()-started,
            pending='Independent full task/GT/selection audit and paired statistics; not G6/goal completion'))
        write(output/'status.json',dict(status='complete_execution',coordinator_pid=os.getpid(),complete=completed))
    except Exception:
        # A shared contract/dependency failure invalidates any still-running child.
        # Explicitly stop only this coordinator's children; upstream caches remain intact.
        for gpu,(proc,handle,arm,stage,destination,log) in list(active.items()):
            if proc.poll() is None:proc.terminate()
            code=proc.wait()
            handle.write(f'\nEXIT_CODE={code}\n');handle.close()
            failed.append(dict(gpu=gpu,arm=arm,stage=stage,exit_code=code,reason='shared coordinator failure'))
        write(output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),complete=completed,failed=failed,
            workers=list(slots.values()),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cpu-gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('output already exists')
    run(a.root,a.cpu_gate,a.output)
