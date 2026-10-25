"""Wait for the owned protocol builder, gate real clicks, then realize full folds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .audit import FOLDS
from benchmarks.v5.fair_comparison.audit import digest, write


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name('prepare.py'), Path(__file__).with_name('audit_prepared.py'),
        Path(__file__).with_name('audit.py'), Path('benchmarks/v5/fair_comparison/realize_training.py'),
        Path('benchmarks/v5/fair_comparison/protocol.py'), Path('benchmarks/v5/multidomain_training/click_policy.py')]
    hashes = {str(p):digest(p) for p in sources}
    write(a.output/'contract.json', dict(worker_pid=os.getpid(), protocol_builder_pid=a.protocol_worker,
        protocol_root=str(a.root), builder_log=str(a.builder_log), source_hashes=hashes,
        policy='All three 64-task pilots and full prepared audit must pass before three full40000 realizations; no retries',
        base_tasks=str(a.base_tasks), remote=str(a.remote), GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))

    def unchanged():
        for p,h in hashes.items():
            if digest(p) != h:raise ValueError('source changed: '+p)

    def state(status, **extra):
        write(a.output/'status.json', dict(status=status, worker_pid=os.getpid(), **extra))

    def finish(children):
        failed = []
        while children:
            unchanged()
            for name, (child, log, handle) in list(children.items()):
                code = child.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close()
                if code != 0:failed.append((name, code))
                del children[name]
            state('running_children', active_children={k:v[0].pid for k,v in children.items()}, failed_children=failed)
            if children:time.sleep(5)
        if failed:raise RuntimeError('child failure: '+str(failed))

    def launch(name, arguments):
        log = a.output/(name+'.log'); handle = log.open('x')
        child = subprocess.Popen([sys.executable, '-u', '-m', *arguments], stdout=handle, stderr=subprocess.STDOUT)
        write(a.output/(name+'_worker.json'), dict(worker_pid=child.pid, command=[sys.executable, '-u', '-m', *arguments], log=str(log)))
        return child, log, handle

    try:
        while not (a.root/'summary.json').exists():
            unchanged()
            proc = Path(f'/proc/{a.protocol_worker}/cmdline')
            if not proc.exists():raise RuntimeError('protocol builder ended without summary; inspect preserved log')
            cmd = proc.read_bytes().replace(b'\0', b' ').decode()
            if 'benchmarks.v5.heldout_comparison.prepare' not in cmd or str(a.root) not in cmd:
                raise RuntimeError('protocol builder PID identity mismatch')
            state('waiting_live_protocol_builder', protocol_builder_pid=a.protocol_worker)
            time.sleep(10)
        # A summary may precede process teardown. Wait for the shell exit evidence.
        while 'EXIT_CODE=' not in a.builder_log.read_text():
            unchanged();state('waiting_builder_exit_evidence');time.sleep(2)
        if not a.builder_log.read_text().rstrip().endswith('EXIT_CODE=0'):raise RuntimeError('builder failed')
        if json.loads((a.root/'summary.json').read_text())['status'] != 'complete_protocols':raise ValueError('protocols incomplete')
        for phase, limit, workers in [('pilot',64,2), ('full',None,4)]:
            folder = a.output/phase;folder.mkdir()
            children = {}
            for fold in FOLDS:
                args = ['benchmarks.v5.fair_comparison.realize_training', '--tasks',str(a.root/fold/'tasks'),
                    '--output',str(folder/fold),'--workers',str(workers)]
                if limit:args += ['--limit',str(limit)]
                children[phase+'_'+fold] = launch(phase+'_'+fold,args)
            finish(children)
            args = ['benchmarks.v5.heldout_comparison.audit_prepared','--root',str(a.root), '--realized',str(folder),
                '--base-tasks',str(a.base_tasks),'--remote',str(a.remote),'--output',str(a.output/(phase+'_audit'))]
            finish({phase+'_audit':launch(phase+'_audit',args)})
            evidence = json.loads((a.output/(phase+'_audit')/'summary.json').read_text())
            if evidence['status'] != 'passed_prepared_protocol_audit':raise ValueError('audit not passed')
            if phase == 'full' and not all(v['complete_actual_protocol'] and v['actual_tasks'] == 40000 for v in evidence['folds'].values()):
                raise ValueError('full coordinates incomplete')
        write(a.output/'summary.json',dict(status='complete_coordinates_and_protocol_audits',worker_pid=os.getpid(),
            full_audit_sha256=digest(a.output/'full_audit/summary.json'),scope='Three full40000 actual point sets; no model training or accuracy'))
        state('complete_coordinates_and_protocol_audits')
    except Exception:
        write(a.output/'failure.json',dict(worker_pid=os.getpid(),traceback=traceback.format_exc()))
        state('failed');raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','base-tasks','remote','builder-log','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol-worker',type=int,required=True)
    run(p.parse_args())
