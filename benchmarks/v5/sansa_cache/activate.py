"""Activate cached SANSA only after the independent exact-equivalence gate."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    while not (root/'gate_exit.json').exists():
        time.sleep(2)
    if json.loads((root/'gate_exit.json').read_text())['exit_code'] != 0:
        dump(root/'activation.json', dict(status='gate_failed_original_queue_retained'))
        return
    receipt = root/'acceptance.json'
    assert json.loads(receipt.read_text())['status'] == 'accepted_exact_cache_equivalence'
    expected = json.loads((root/'source_hashes.json').read_text())
    for path, digest in expected.items():
        assert sha(path) == digest, path
    registry = Path('/home/zhaoyh/FewClick_v2/docs/运行记录/模型实验/CVPR研究进度_1030_174832.json')
    reg = json.loads(registry.read_text())
    previous = Path(reg['whole_wsi_six_gpu_priority']['root'])
    pid = reg['whole_wsi_six_gpu_priority']['controller_pid']
    proc = Path(f'/proc/{pid}')
    # Gate launcher resumes dispatch in finally; wait for that before replacing it.
    while proc.exists() and proc.joinpath('stat').read_text().split()[2] == 'T':
        time.sleep(.2)
    assert b'whole_slide_sam.priority' in proc.joinpath('cmdline').read_bytes()
    os.kill(pid, signal.SIGSTOP)
    stopped = True
    try:
        jobs = json.loads((previous/'jobs.json').read_text())
        assert not any(j['status'] == 'failed' for j in jobs)
        switched = 0
        for job in jobs:
            if job['status'] == 'pending' and job['method'] == 'HG_mix+sansa':
                job['module'] = 'benchmarks.v5.sansa_cache.worker'
                switched += 1
        c = json.loads((previous/'contract.json').read_text())
        whole = Path(c['whole_root'])
        extension = dict(gate_acceptance=str(receipt), gate_acceptance_sha256=sha(receipt),
                         sources={str(Path('benchmarks/v5/sansa_cache')/name):
                                  sha(Path('benchmarks/v5/sansa_cache')/name)
                                  for name in ['model.py', 'worker.py']})
        assert not (whole/'cache_extension.json').exists()
        dump(whole/'cache_extension.json', extension)
        queue = root/'queue_1113_005000'
        queue.mkdir()
        (queue/'tmp').mkdir()
        dump(queue/'resume_jobs.json', jobs)
        c.update(resume_jobs=str(queue/'resume_jobs.json'), temporary_directory=str(queue/'tmp'),
                 previous_root=str(previous), sansa_cache_extension=extension)
        dump(queue/'contract.json', c)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        # Compute workers have independent sessions and keep their original code.
        command = [sys.executable, '-u', '-m', 'benchmarks.v5.whole_slide_sam.priority', '--root', str(queue)]
        log = (queue/'controller.log').open('x')
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 cwd='/home/zhaoyh/FewClick_v2', start_new_session=True)
        dump(queue/'controller_worker.json', dict(pid=child.pid, command=command))
        reg = json.loads(registry.read_text())
        reg['whole_wsi_six_gpu_priority'].update(root=str(queue), controller_pid=child.pid,
              status='running_cached_SANSA_for_pending_jobs_only', previous_root=str(previous))
        reg['sansa_cache_optimization'].update(status='gate_accepted_activated', queue=str(queue),
              pending_paired_jobs_switched=switched, gate_acceptance=str(receipt))
        dump(registry, reg)
        dump(root/'activation.json', dict(status='activated', queue=str(queue), controller_pid=child.pid,
             pending_paired_jobs_switched=switched, preserved_running_jobs=sum(j['status']=='running' for j in jobs)))
        note = f'\nSANSA缓存真实严格等价门控通过并切换后续{switched}个paired任务；当前worker全部保留。队列{queue}，controller {child.pid}；gate={receipt}。\n'
        for path in ['goal.md', 'IMPLEMENTATION_STATUS.md',
                     'docs/运行记录/模型实验/运行记录_SAM当前9500公平适配与WSI并行_1112_171010.md',
                     'docs/运行记录/模型实验/运行记录_SANSA缓存等价优化_1113_004500.md']:
            with Path(path).open('a') as handle:
                handle.write(note)
        rc = child.wait()
        log.write(f'\nEXIT_CODE={rc}\n')
        log.close()
        dump(queue/'controller_exit.json', dict(exit_code=rc))
    finally:
        if stopped and proc.exists():
            os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
