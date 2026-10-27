"""Observe owned jobs using terminal logs, not mutable progress JSON files."""
import json
import os
from pathlib import Path
import re


def completed_receipt(log_path,worker_pid,summary_path,expected_status):
    """Return a stable successful receipt, or None while the owned job lives.

    A failed exit, missing receipt after exit0, malformed terminal receipt, or
    disappeared process without an exit record is an error. No retry, fallback,
    or recovery of failed computation is performed here.
    """
    log_path=Path(log_path);summary_path=Path(summary_path)
    log=log_path.read_text() if log_path.exists() else ''
    terminal=re.findall(r'^EXIT_CODE=(-?\d+)\s*$',log,re.MULTILINE)
    if terminal:
        if len(terminal)!=1 or int(terminal[0])!=0:
            raise RuntimeError('Dependency terminal failure: '+str(log_path))
        if not summary_path.is_file():
            raise RuntimeError('Successful exit lacks completion receipt: '+str(summary_path))
        receipt=json.loads(summary_path.read_text())
        if receipt['status']!=expected_status:
            raise RuntimeError('Unexpected completion receipt: '+str(receipt['status']))
        return receipt
    try:
        os.kill(worker_pid,0)
    except ProcessLookupError as e:
        raise RuntimeError('Owned dependency disappeared without terminal record: '+str(worker_pid)) from e
    return None


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.next')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
