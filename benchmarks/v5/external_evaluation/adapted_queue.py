"""Score only development-selected checkpoints after native external cells exist."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .prepare import digest,write
from .evaluate_adapted import METHODS
from benchmarks.v5.experiment_orchestration.mechanism_queue import occupied_gpus


def run(tasks,development_tasks,adaptation,cells,cpu_gates,output):
    output.mkdir(parents=True,exist_ok=False);started=time.time();complete={}
    names=['adapted_queue.py','evaluate_adapted.py','cell_data.py','evaluate_public.py','prepare.py']
    files=[Path(__file__).with_name(n) for n in names]
    files += [Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/fair_comparison/metrics.py'),Path('benchmarks/v5/fair_comparison/protocol.py')]
    snapshot={str(p):digest(p) for p in files}
    protocol_sha=digest(tasks/'contract.json');development_sha=digest(development_tasks/'contract.json')
    stamp='_'.join(output.name.rsplit('_',2)[-2:])
    try:
        gates=json.loads(cpu_gates.read_text())
        if set(gates)!={'sam_ft','wsi_sam_ft','cell_assembly'}:raise ValueError('CPU gate map incomplete')
        for name,path in gates.items():
            result=json.loads((Path(path)/'summary.json').read_text())
            if name=='cell_assembly':
                if result['status']!='passed' or not result['GT_unavailable_assembly']:raise ValueError('real cell assembly CPU gate missing')
            else:
                if result['status']!='complete' or not result['gate'] or result['rows']!=8 or not result['GT_unavailable_forward_passed']:
                    raise ValueError('real adapted external CPU gate missing')
                metadata=json.loads((Path(path)/'run.json').read_text())
                for source,expected in metadata['source_sha256'].items():
                    if digest(source)!=expected:raise ValueError('CPU-gated code changed')
        write(output/'contract.json',dict(coordinator_pid=os.getpid(),gpu=4,methods=METHODS,tasks=str(tasks),
            development_tasks=str(development_tasks),adaptation=str(adaptation),cells=str(cells),cpu_gates=gates,
            protocol_sha256=protocol_sha,development_protocol_sha256=development_sha,source_sha256=snapshot,
            weight_selection='completed CRC/TIGER train best_checkpoint.json only; no external selection',
            stages='same-final-checkpoint native gate8 rows, full680 rows per method',failure_policy='stop and preserve; no retries/substitutions'))
        while True:
            try:
                cell_state=json.loads((cells/'status.json').read_text());development=json.loads((adaptation/'status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(1);continue
            if cell_state['status']=='failed':raise RuntimeError('native external cell preparation failed')
            if any(f.get('method') in METHODS for f in development.get('failed',[])):
                raise RuntimeError('required adaptation method failed')
            done=development.get('complete',{})
            ready=cell_state['status']=='complete' and all('train|'+m in done for m in METHODS)
            if development['status']=='failed' and not all('train|'+m in done for m in METHODS):
                raise RuntimeError('terminal adaptation lacks required completed train')
            if ready and 4 not in occupied_gpus():break
            write(output/'status.json',dict(status='waiting_dependencies' if not ready else 'waiting_gpu_idle',
                coordinator_pid=os.getpid(),gpu_workers_started=False,complete=complete,seconds=time.time()-started))
            time.sleep(5)
        prepared=Path(json.loads((cells/'summary.json').read_text())['prepared'])
        for method in METHODS:
            training=Path(done['train|'+method]);selected=json.loads((training/'best_checkpoint.json').read_text())
            if digest(selected['path'])!=selected['checkpoint_sha256']:raise ValueError('selected weight identity changed')
            gate=output/f'gate_{method}_{stamp}'
            for stage in ('gate','full'):
                for source,expected in snapshot.items():
                    if digest(source)!=expected:raise ValueError('locked external adapted code changed '+source)
                if digest(tasks/'contract.json')!=protocol_sha or digest(development_tasks/'contract.json')!=development_sha:
                    raise ValueError('locked protocol changed')
                destination=gate if stage=='gate' else output/f'full_{method}_{stamp}'
                command=[sys.executable,'-u','-m','benchmarks.v5.external_evaluation.evaluate_adapted',
                    '--tasks',str(tasks),'--development-tasks',str(development_tasks),'--checkpoint',selected['path'],
                    '--method',method,'--output',str(destination),'--device','cuda:0','--training',str(training)]
                if method.startswith('fewclick'):command+=['--prepared',str(prepared)]
                command += ['--gate'] if stage=='gate' else ['--verified-gate',str(gate)]
                log=destination.with_suffix('.log')
                with log.open('x') as handle:
                    env=os.environ|dict(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                    worker=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
                    write(output/'status.json',dict(status='running',coordinator_pid=os.getpid(),worker_pid=worker.pid,
                        method=method,stage=stage,gpu=4,gpu_workers_started=True,output=str(destination),log=str(log),complete=complete))
                    code=worker.wait();handle.write(f'\nEXIT_CODE={code}\n')
                if code:raise RuntimeError(f'external {method} {stage} failed: {log}')
                result=json.loads((destination/'summary.json').read_text())
                if result['status']!='complete' or result['rows']!=(8 if stage=='gate' else 680):raise ValueError('external adapted coverage incomplete')
                complete[stage+'|'+method]=str(destination)
        write(output/'summary.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete,seconds=time.time()-started,
            pending='paired complete-row audit, cluster report with public baselines, independent training seeds'))
        write(output/'status.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete))
    except Exception:
        write(output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),complete=complete,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--development-tasks',type=Path,required=True)
    p.add_argument('--adaptation',type=Path,required=True);p.add_argument('--cells',type=Path,required=True)
    p.add_argument('--cpu-gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.tasks,a.development_tasks,a.adaptation,a.cells,a.cpu_gates,a.output)
