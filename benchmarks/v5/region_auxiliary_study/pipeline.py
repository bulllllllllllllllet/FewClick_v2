"""One arm per GPU: engineering -> learning -> fresh training -> same-WSI scoring."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha
from .model import ARMS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    root = a.root
    contract = json.load(open(root / 'contract.json'))
    arm = root / a.arm
    for path, digest in contract['weights'].items():
        assert sha(path) == digest, path
    assert sha(contract['training_source']) == contract['training_source_sha256']
    assert sha(Path(contract['evaluation_source']) / 'tasks.json') == contract['evaluation_tasks_sha256']

    def run(stage, module, args):
        for path, digest in contract['sources'].items():
            assert sha(path) == digest, path
        free = int(subprocess.check_output(['nvidia-smi', f'--id={a.gpu}', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 12288, (a.gpu, free)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
        cmd = [sys.executable, '-u', '-m', 'benchmarks.v5.region_auxiliary_study.' + module, *map(str, args)]
        with (arm / (stage + '_pipeline.log')).open('x') as handle:
            proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            dump(arm / 'pipeline_stage.json', dict(status='running', stage=stage, worker_pid=proc.pid,
                 controller_pid=os.getpid(), command=cmd, gpu=a.gpu))
            rc = proc.wait()
            handle.write(f'\nEXIT_CODE={rc}\n')
            assert rc == 0, (stage, rc)

    try:
        run('integration', 'integration', ['--root', root, '--arm', a.arm])
        run('workflow', 'workflow', ['--root', root, '--arm', a.arm])
        run('prepare_eval', 'prepare_eval', ['--root', root, '--arm', a.arm])
        er = arm / 'evaluation'
        run('eval_gate', 'evaluate', ['--root', er, '--method', a.arm, '--shard', 0, '--gate'])
        for shard in [0, 1]:
            run('eval_' + str(shard), 'evaluate', ['--root', er, '--method', a.arm, '--shard', shard])
        run('aggregate', 'aggregate', ['--root', er])
        dump(arm / 'pipeline_stage.json', dict(status='complete_accepted', quality_promotion='pending_paired_review'))
    except Exception as error:
        dump(arm / 'pipeline_stage.json', dict(status='failed', error=repr(error), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
