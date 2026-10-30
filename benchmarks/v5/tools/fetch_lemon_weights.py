"""Fetch one pinned public LEMON checkpoint; verify its published LFS SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REVISION='d12246ba78018e1abe8f9477e951cc7c31c83209'
EXPECTED_SHA='1529389956ca0a128771c6481f0c9448682f231121f919c1c27eda54da0cb62e'
EXPECTED_BYTES=678346999


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    url=f'https://huggingface.co/aliceblondel/LEMON/resolve/{REVISION}/lemon_vits8.pth.tar'
    status=dict(status='running',worker_pid=os.getpid(),url=url,revision=REVISION,
                expected_sha256=EXPECTED_SHA,expected_bytes=EXPECTED_BYTES,bytes_received=0)
    def save():
        status['elapsed_seconds']=time.time()-start
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        digest=hashlib.sha256()
        with requests.get(url,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            with (a.output/'lemon_vits8.pth.tar.partial').open('xb') as target:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if not chunk:continue
                    target.write(chunk);digest.update(chunk);status['bytes_received']+=len(chunk)
                    if status['bytes_received']%(16*1024*1024)==0:save()
        actual=digest.hexdigest()
        if actual!=EXPECTED_SHA or status['bytes_received']!=EXPECTED_BYTES:
            raise ValueError(f'published LFS identity mismatch: {actual}, {status["bytes_received"]}')
        (a.output/'lemon_vits8.pth.tar.partial').rename(a.output/'lemon_vits8.pth.tar')
        status.update(status='complete_download_identity_only',sha256=actual,model_loading_accepted=False)
        save();print(json.dumps(status),flush=True)
    except Exception as exc:
        status.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
