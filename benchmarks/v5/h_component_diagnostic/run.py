"""Engineering -> exact historical six-task gate -> two independent shards."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    c = json.loads((root/'contract.json').read_text())

    def launch(stage, module, args, gpu):
        for path, digest in c['sources'].items():
            assert sha(path) == digest, path
        while int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'], text=True)) < 12288:
            dump(root/(stage+'_resource_wait.json'), dict(status='waiting_free_memory', gpu=gpu, at=time.time()))
            time.sleep(30)
        command = [sys.executable, '-u', '-m', 'benchmarks.v5.h_component_diagnostic.'+module,
                   '--root', str(root), *args]
        log = (root/(stage+'.log')).open('x')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        item = dict(stage=stage, pid=proc.pid, gpu=gpu, command=command)
        dump(root/(stage+'_worker.json'), item)
        return proc, log, item

    def finish(job):
        proc, log, _ = job
        code = proc.wait()
        log.write(f'\nEXIT_CODE={code}\n')
        log.close()
        return code

    def single(stage, module, args):
        job = launch(stage, module, args, c['gpus'][0])
        dump(root/'stage.json', dict(status=stage+'_running', **job[2]))
        assert finish(job) == 0, stage

    try:
        single('engineering', 'engineering', [])
        single('gate', 'evaluate', ['--shard', '0', '--gate'])
        jobs = []
        for shard, gpu in enumerate(c['gpus']):
            jobs.append(launch('shard_'+str(shard), 'evaluate', ['--shard', str(shard)], gpu))
            dump(root/'stage.json', dict(status='parallel_inference_running', workers=[j[2] for j in jobs]))
        codes = [finish(job) for job in jobs]
        assert codes == [0, 0], codes
        single('aggregate', 'aggregate', [])
        dump(root/'stage.json', dict(status='complete_accepted_inference_diagnostic', no_training=True))
    except Exception as error:
        dump(root/'stage.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
