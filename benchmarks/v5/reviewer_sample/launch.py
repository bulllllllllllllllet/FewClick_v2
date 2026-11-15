"""Launch only the two authorized matched arms with the accepted environment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    active = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    assert not any(' -m benchmarks.v5.reviewer_sample.run ' in line for line in active), 'active study already exists'
    subprocess.run([sys.executable, '-m', 'benchmarks.v5.reviewer_sample.prepare', '--root', str(a.root)], check=True)
    c = json.loads((a.root/'contract.json').read_text()); receipt = []
    for arm, gpu in c['gpus'].items():
        env = dict(CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(a.root/'tmp'), PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:'+str(Path.cwd()),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', VIPS_CONCURRENCY='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='2', VIPS_DISC_THRESHOLD='100m')
        cmd = [sys.executable, '-u', '-m', 'benchmarks.v5.reviewer_sample.run', '--root', str(a.root), '--arm', arm]
        log = a.root/arm/'launcher.log'; session = a.root.name+'_'+arm
        shell = shlex.join(['env', *[k+'='+v for k, v in env.items()], *cmd])+' > '+shlex.quote(str(log))+' 2>&1; rc=$?; echo EXIT_CODE=$rc >> '+shlex.quote(str(log))+'; exit $rc'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(Path.cwd()), shell], check=True)
        receipt.append(dict(arm=arm, gpu=gpu, session=session, command=cmd, log=str(log)))
    dump(a.root/'launch.json', dict(status='launched_not_complete', arms=receipt)); print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
