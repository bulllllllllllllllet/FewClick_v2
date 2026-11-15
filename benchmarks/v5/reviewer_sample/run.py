"""One assigned GPU per arm, sequential gated stages, no automatic retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True); a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text()); out = a.root/a.arm; er = out/'evaluation'
    stages = [('engineering', 'engineering', ['--root', a.root, '--arm', a.arm]),
              ('functional', 'functional', ['--cache', c['cache'], '--output', out/'functional', '--protocol', c['functional_source'], '--arm', a.arm]),
              ('smoke', 'train', ['--root', a.root, '--arm', a.arm, '--smoke']),
              ('formal', 'train', ['--root', a.root, '--arm', a.arm]),
              ('prepare_eval', 'prepare_eval', ['--root', a.root, '--arm', a.arm]),
              ('gate_eval', 'evaluate', ['--root', er, '--method', a.arm, '--shard', 0, '--gate']),
              ('eval_0', 'evaluate', ['--root', er, '--method', a.arm, '--shard', 0]),
              ('eval_1', 'evaluate', ['--root', er, '--method', a.arm, '--shard', 1]),
              ('aggregate', 'aggregate', ['--root', er])]
    try:
        for name, module, args in stages:
            for f, h in c['sources'].items(): assert sha(f) == h, f
            while int(subprocess.check_output(['nvidia-smi', '--id='+os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) < 12288:
                dump(out/'stage.json', dict(status='waiting_memory', next_stage=name)); time.sleep(30)
            cmd = [sys.executable, '-u', '-m', 'benchmarks.v5.reviewer_sample.'+module, *map(str, args)]
            with (out/(name+'.log')).open('x') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                dump(out/'stage.json', dict(status=name+'_running', worker_pid=proc.pid, controller_pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], command=cmd))
                rc = proc.wait(); log.write(f'\nEXIT_CODE={rc}\n')
            assert rc == 0, (name, rc)
        dump(out/'stage.json', dict(status='complete_accepted_patch', wsi='pending_separate_queue'))
    except Exception as error:
        dump(out/'stage.json', dict(status='failed_stop_no_retry', stage=name, error=repr(error))); raise


if __name__ == '__main__': main()
