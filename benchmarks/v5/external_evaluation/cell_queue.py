"""Wait for the declared G6 GPU4 queue, then gate and extract external cells."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .prepare import digest,write
from benchmarks.v5.experiment_orchestration.mechanism_queue import occupied_gpus


def run(tasks,dependency,cpu_gate,output):
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    evidence=json.loads((cpu_gate/'summary.json').read_text())
    if evidence['status']!='passed' or not evidence['GT_unavailable_assembly']:
        raise ValueError('real CPU model/data gate missing')
    for path,expected in evidence['source_sha256'].items():
        if digest(path)!=expected:raise ValueError('CPU-gated source changed')
    names=['cell_queue.py','cells.py','cell_data.py','prepare.py','evaluate_public.py']
    files=[Path(__file__).with_name(n) for n in names]
    files += [Path('benchmarks/v5/multidomain_data/prepare_cells.py'),Path('benchmarks/v5/multidomain_data/fast_nuclei.py'),Path('benchmarks/v5/paper_multiscale/data.py')]
    snapshot={str(p):digest(p) for p in files}
    protocol_sha=digest(tasks/'contract.json')
    stamp='_'.join(output.name.rsplit('_',2)[-2:])
    native_gate=output/f'native_gate_{stamp}';full=output/f'native_full_{stamp}'
    write(output/'contract.json',dict(coordinator_pid=os.getpid(),tasks=str(tasks),dependency=str(dependency),
        cpu_gate=str(cpu_gate),cpu_gate_sha256=digest(cpu_gate/'summary.json'),protocol_sha256=protocol_sha,
        source_sha256=snapshot,gpu=4,required=['evaluate|m3_none','evaluate|m3_morphology'],
        stages=['native three-view one-field gate','all ten fields with gate cache reuse'],failure_policy='stop; no retry/substitution'))
    try:
        while True:
            try:state=json.loads((dependency/'status.json').read_text())
            except json.JSONDecodeError:
                # Observe an in-place upstream write again; no task is relaunched.
                time.sleep(1);continue
            if any(item.get('gpu')==4 for item in state.get('failed',[])):
                raise RuntimeError('GPU4 mechanism dependency failed')
            done=state.get('complete',{})
            keys=['evaluate|m3_none','evaluate|m3_morphology']
            ready=all(k in done for k in keys)
            if state['status']=='failed' and not ready:
                raise RuntimeError('terminal mechanism dependency lacks completed GPU4 queue')
            if ready:
                for k in keys:
                    if json.loads((Path(done[k])/'summary.json').read_text())['status']!='complete':
                        raise ValueError('dependency completion lacks summary')
                if 4 not in occupied_gpus():break
            write(output/'status.json',dict(status='waiting_dependency' if not ready else 'waiting_gpu_idle',
                coordinator_pid=os.getpid(),gpu_workers_started=False,seconds=time.time()-started))
            time.sleep(5)
        for stage,destination in [('gate',native_gate),('full',full)]:
            if digest(tasks/'contract.json')!=protocol_sha:raise ValueError('locked external protocol changed')
            for path,expected in snapshot.items():
                if digest(path)!=expected:raise ValueError('locked extraction source changed '+path)
            command=[sys.executable,'-u','-m','benchmarks.v5.external_evaluation.cells','--tasks',str(tasks),'--output',str(destination)]
            command += ['--gate'] if stage=='gate' else ['--reuse',str(native_gate)]
            log=destination.with_suffix('.log')
            with log.open('x') as handle:
                env=os.environ|dict(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
                write(output/'status.json',dict(status='running',coordinator_pid=os.getpid(),stage=stage,gpu=4,
                    worker_pid=process.pid,gpu_workers_started=True,output=str(destination),log=str(log)))
                code=process.wait();handle.write(f'\nEXIT_CODE={code}\n')
            if code:raise RuntimeError(f'{stage} native extraction failed; retained {log}')
            result=json.loads((destination/'summary.json').read_text())
            if result['status']!='complete' or result['fields']!=(1 if stage=='gate' else 10):
                raise ValueError('native extraction incomplete')
        write(output/'summary.json',dict(status='complete',coordinator_pid=os.getpid(),native_gate=str(native_gate),
            prepared=str(full),fields=10,seconds=time.time()-started))
        write(output/'status.json',dict(status='complete',coordinator_pid=os.getpid(),prepared=str(full)))
    except Exception:
        write(output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--dependency',type=Path,required=True)
    p.add_argument('--cpu-gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.tasks,a.dependency,a.cpu_gate,a.output)
