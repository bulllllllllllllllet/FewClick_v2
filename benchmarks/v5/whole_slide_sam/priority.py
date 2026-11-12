"""Six-GPU WSI-first dispatcher; keep existing workers and frozen protocols."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump


def occupied(pool):
    used = set()
    for path in pool.glob('slot_*.lock'):
        with path.open('r+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    used.add(int(json.loads(handle.read())['gpu']))
                except (ValueError, KeyError):
                    pass
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return used


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text())
    whole, baseline = Path(c['whole_root']), Path(c['baseline_root'])
    bc = json.loads((baseline/'contract.json').read_text())
    jobs = []
    for i in range(3):
        for arm in ['sam', 'sam2', 'wsi', 'med2d']:
            jobs.append(dict(method=arm, case=i, root=str(baseline), phase='pilot',
                             module='benchmarks.v5.whole_slide_sam.worker', status='pending'))
    # Each original worker runs HG-mix and SANSA in sequence on the same slide.
    for i in range(15):
        jobs.append(dict(method='HG_mix+sansa', case=i, root=str(whole), phase='formal',
                         module='benchmarks.v5.whole_slide_once.worker', status='pending'))
        for arm in ['sam', 'sam2', 'wsi', 'med2d']:
            jobs.append(dict(method=arm, case=i, root=str(baseline), phase='formal',
                             module='benchmarks.v5.whole_slide_sam.worker', status='pending'))
    active = []
    if 'resume_jobs' in c:
        jobs = json.loads(Path(c['resume_jobs']).read_text())
        class Existing:
            def __init__(self, job):
                self.job = job

            def poll(self):
                pid = self.job['pid']
                path = Path(f'/proc/{pid}/stat')
                if path.exists() and path.read_text().split()[2] != 'Z':
                    return None
                job = self.job
                out = Path(job['root'])
                if job['method'] != 'HG_mix+sansa':
                    out /= job['method']
                out = out/job['phase']/f"case_{job['case']:02}"/'acceptance.json'
                return 0 if out.exists() else 1
        for job in jobs:
            if job['status'] == 'running':
                active.append((Existing(job), (a.root/f"carried_{job['pid']}.log").open('x'), job))
    dump(a.root/'stage.json', dict(status='six_GPU_WSI_priority_running', controller_pid=os.getpid()))
    try:
        while any(j['status'] == 'pending' for j in jobs) or active:
            for proc, log, job in list(active):
                code = proc.poll()
                if code is None:
                    continue
                log.write(f'\nEXIT_CODE={code}\n')
                log.close()
                job.update(status='complete' if code == 0 else 'failed', exit_code=code)
                active.remove((proc, log, job))
                dump(a.root/'jobs.json', jobs)
                assert code == 0, job
            blocked = occupied(Path(bc['resource_pool'])) | {j['gpu'] for _, _, j in active}
            for gpu in range(6):
                if gpu in blocked or len(active) >= 6:
                    continue
                free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                           '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
                if free < 12288:
                    continue
                ready = []
                for job in jobs:
                    if job['status'] != 'pending':
                        continue
                    if job['method'] != 'HG_mix+sansa':
                        f = Path(bc['adaptations'][job['method']])/'formal/acceptance.json'
                        if not f.exists():
                            continue
                        receipt = json.loads(f.read_text())
                        assert receipt['status'] == 'complete_accepted_training'
                        assert receipt['updates'] == 2375 and receipt['consumed'] == 9500
                        if job['phase'] == 'formal' and not all(
                            (baseline/job['method']/'pilot'/f'case_{i:02}'/'acceptance.json').exists()
                            for i in range(3)):
                            continue
                    ready.append(job)
                if not ready:
                    continue
                job = ready[0]
                out = Path(job['root'])/job['phase']/f"case_{job['case']:02}"
                if job['method'] != 'HG_mix+sansa':
                    out = Path(job['root'])/job['method']/job['phase']/f"case_{job['case']:02}"
                assert not out.exists(), f'Existing artifact must be audited, never overwritten: {out}'
                cmd = [sys.executable, '-u', '-m', job['module'], '--root', job['root'],
                       '--case-index', str(job['case']), '--phase', job['phase']]
                if job['method'] != 'HG_mix+sansa':
                    cmd += ['--arm', job['method']]
                log = (a.root/f"{job['phase']}_{job['method']}_{job['case']:02}.log").open('x')
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                                TMPDIR=c['temporary_directory'], VIPS_DISC_THRESHOLD='100m'))
                job.update(status='running', gpu=gpu, pid=proc.pid, command=cmd)
                active.append((proc, log, job))
                blocked.add(gpu)
                dump(a.root/'jobs.json', jobs)
            dump(a.root/'progress.json', dict(active=len(active), complete=sum(j['status']=='complete' for j in jobs),
                  total=len(jobs), pending=sum(j['status']=='pending' for j in jobs),
                  GPU_leases_occupied=sorted(occupied(Path(bc['resource_pool'])))))
            time.sleep(10)
        from benchmarks.v5.whole_slide_once.run import aggregate as aggregate_original
        from .run import aggregate as aggregate_sam
        aggregate_original(whole, 'formal', list(range(15)))
        aggregate_sam(baseline, bc, 'formal', list(range(15)))
        original = json.loads((whole/'formal_report.json').read_text())
        comparison = json.loads((baseline/'formal_report.json').read_text())
        rows = original['rows']+comparison['rows']
        grouped = {}
        for row in rows:
            grouped.setdefault(row['method'], {})[(row['slide'], row['target'])] = row
        assert set(grouped) == {'HG_mix', 'sansa', 'sam', 'sam2', 'wsi', 'med2d'}
        reference = grouped['HG_mix']
        for method, entries in grouped.items():
            assert entries.keys() == reference.keys(), method
            for key, row in entries.items():
                for scope in ['whole_WSI', 'remote_only_WSI']:
                    for field in ['valid', 'positive']:
                        assert row[scope][field] == reference[key][scope][field]
        dump(a.root/'six_method_comparison.json', dict(status='complete_accepted_common_cohort',
             rows=rows, matched_targets=len(reference), same_valid_positive_pixels=True))
        # Optional extra15 waits for a measured six-method throughput decision.
        dump(whole/'expansion_decision.json', dict(expand_to30=False,
             reason='User prioritizes six-method formal15 on six GPUs; optional extra15 deferred until common results and measured cost.'))
        dump(a.root/'stage.json', dict(status='complete_accepted_formal15_six_methods', training_updates=0,
             six_method_join='Common case/target table must be checked before paper reporting', optional_extra15='deferred'))
    except Exception as error:
        dump(a.root/'stage.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
