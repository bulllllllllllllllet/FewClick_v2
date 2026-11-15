"""Bounded four-GPU WSI queue alongside two training workers; fail closed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root; ext = json.loads((root/'wsi_contract.json').read_text()); jobs = []
    for method in ['HG_mix', 'sansa', 'M', 'no_cross_scale', 'proto_ms']:
        for case in range(3): jobs.append(dict(method=method, case=case, status='pending'))
    running = {}; blocked = set(); (root/'wsi_logs').mkdir(exist_ok=False)
    def save(): dump(root/'wsi_jobs.json', dict(controller_pid=os.getpid(), jobs=jobs, updated_at=time.time()))
    def ready(job):
        if job['method'] in blocked: return False
        if job['method'] == 'proto_ms':
            stage = root/'proto_ms/stage.json'
            if stage.exists() and json.loads(stage.read_text())['status'] == 'failed_stop_no_retry':
                blocked.add('proto_ms'); return False
            acc = root/'proto_ms/evaluation/acceptance.json'
            if not acc.exists(): return False
            assert json.loads(acc.read_text())['status'] == 'complete_accepted'
        if job['case'] > 0:
            # Full/SANSA entire first-slide reproduction before any later-slide fan-out.
            if not all(any(j['method'] == m and j['case'] == 0 and j['status'] == 'complete' for j in jobs) for m in ['HG_mix', 'sansa']): return False
            if not any(j['method'] == job['method'] and j['case'] == 0 and j['status'] == 'complete' for j in jobs): return False
        return True
    save()
    while any(j['status'] in ['pending', 'running'] for j in jobs):
        for gpu, (proc, log, job) in list(running.items()):
            code = proc.poll()
            if code is None: continue
            log.write(f'\nEXIT_CODE={code}\n'); log.close(); job['exit_code'] = code
            acc = root/'wsi'/job['method']/f"case_{job['case']:02}"/'acceptance.json'
            if code == 0 and acc.exists() and json.loads(acc.read_text())['status'] == 'complete_accepted': job['status'] = 'complete'
            else: job['status'] = 'failed'; blocked.add(job['method'])
            del running[gpu]; save()
        for job in jobs:
            if job['status'] == 'pending' and job['method'] in blocked: job['status'] = 'blocked_by_failed_arm'
        if any(j['case'] == 0 and j['method'] in ['HG_mix', 'sansa'] and j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['case'] > 0 and job['status'] == 'pending': job['status'] = 'blocked_by_reference_reproduction_failure'
        for gpu in [0, 1, 2, 4]:
            if gpu in running: continue
            candidates = [j for j in jobs if j['status'] == 'pending' and ready(j)]
            if not candidates: continue
            free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
            if free < 12288: continue
            for f, h in ext['sources'].items(): assert sha(f) == h, f
            job = candidates[0]; log = (root/'wsi_logs'/f"{job['method']}_{job['case']:02}.log").open('x')
            cmd = [sys.executable, '-u', '-m', 'benchmarks.v5.reviewer_sample.wsi', '--root', str(root), '--method', job['method'], '--case-index', str(job['case'])]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            job.update(status='running', gpu=gpu, worker_pid=proc.pid, command=cmd, started_at=time.time()); running[gpu] = (proc, log, job); save()
        save(); time.sleep(15)
    save()
    if all(j['status'] == 'complete' for j in jobs):
        subprocess.run([sys.executable, '-m', 'benchmarks.v5.reviewer_sample.wsi_summary', '--root', str(root)], check=True)
    else:
        dump(root/'wsi_terminal.json', dict(status='incomplete_failed_or_blocked', jobs=jobs))


if __name__ == '__main__': main()
