"""Shared six-worker lease pool; file locks release if a controller exits."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def acquire(contract, gpu, stage):
    root = Path(contract['resource_pool'])
    root.mkdir(exist_ok=True)
    while True:
        free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
        if free >= 12288:
            for index in range(6):
                handle = (root/f'slot_{index}.lock').open('a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(dict(controller_pid=os.getpid(), gpu=gpu,
                                              stage=stage, acquired_at=time.time())))
                handle.flush()
                return handle
        time.sleep(10)
