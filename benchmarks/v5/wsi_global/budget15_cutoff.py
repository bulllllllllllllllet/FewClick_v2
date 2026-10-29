"""Preserve the original serial cell stream through an authorized WSI prefix.

The old producer is stopped only after every selected input is committed. Its
partial original run remains partial; a new reference manifest needs acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from .completion import atomic_json


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(pid):
    proc = Path('/proc')/str(pid)
    try:
        stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
        cmd = (proc/'cmdline').read_bytes()
        if stat[0] == 'Z' or not cmd:
            return None
        return dict(start_ticks=stat[19],cmdline_sha256=hashlib.sha256(cmd).hexdigest(),uid=proc.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_owned(pid, start_ticks, cmdline_sha256):
    actual = identity(pid)
    require(actual is not None and actual['uid'] == os.getuid()
        and actual['start_ticks'] == str(start_ticks) and actual['cmdline_sha256'] == cmdline_sha256,
        'refuse stop: owned PID/start/argv identity must match')
    os.kill(pid,signal.SIGTERM)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True)
    pid = os.getpid()
    def dump(status, **kwargs):
        atomic_json(a.output/'status.json',dict(status=status,worker_pid=pid,**kwargs))
    try:
        protocol = json.loads((a.protocol/'protocol.json').read_text())
        rows = list(map(json.loads,(a.protocol/'inputs.jsonl').open()))
        n = protocol['input_positions']
        require(n == len(rows) == 8895 and protocol['slides'] == 15 and len(set(protocol['WSI_ids'])) == 15, 'explicit authorized15 scope')
        require(sha(a.protocol/'inputs.jsonl') == protocol['selected_inputs_sha256'], 'immutable selected input bytes')
        original = Path(protocol['original_root'])/'cell_native_inputs_full_1028_012700/inputs.jsonl'
        require(sha(original) == protocol['source_inputs_sha256'], 'original full30 native inputs untouched')
        source_rows = list(map(json.loads,original.open()))
        require(rows == source_rows[:n] and [r['index'] for r in rows] == list(range(n)), 'exact original serial prefix, no selection by model results')
        cache = Path(protocol['source_cache'])
        source_contract = json.loads((cache/'contract.json').read_text())
        target = protocol['source_producer_pid']
        require(source_contract['worker_pid'] == target and source_contract['signature']['visible_devices'] == '1', 'original owned real-cell producer')
        source_hashes = source_contract['signature']['sources']
        for p,digest in source_hashes.items():
            require(sha(p) == digest, 'original model/preparation code unchanged')
        frozen = dict(protocol_sha256=sha(a.protocol/'protocol.json'),inputs_sha256=sha(a.protocol/'inputs.jsonl'),
            source_contract_sha256=sha(cache/'contract.json'),source_sha256=sha(__file__),
            source_producer_start_ticks=protocol['source_producer_start_ticks'],
            source_producer_cmdline_sha256=protocol['source_producer_cmdline_sha256'])
        atomic_json(a.output/'contract.json',dict(worker_pid=pid,signature=frozen,producer_pid=target,
            stop_reason='explicit user reduction to original first15 WSIs',expected_positions=n,
            policy='Never restart or resume producer. Preserve all source files. Prefix completion is not cache/quality acceptance.'))
        while True:
            require(sha(a.protocol/'protocol.json') == frozen['protocol_sha256']
                and sha(a.protocol/'inputs.jsonl') == frozen['inputs_sha256'], 'fixed authorized cohort')
            # Only newline-terminated records are committed; partial final writes
            # are observations of a live producer, never recovered computations.
            complete = (cache/'cache_manifest.jsonl').read_bytes().split(b'\n')[:-1]
            selected = complete[:n]
            entries = [json.loads(line) for line in selected]
            for index, entry in enumerate(entries):
                r = rows[index]
                require(entry['index'] == r['index'] and entry['identity']['key'] == r['key']
                    and entry['identity']['wsi_id'] == r['wsi_id'], 'every committed selected input in original order')
            actual = identity(target)
            if actual is not None:
                require(actual['uid'] == os.getuid() and actual['start_ticks'] == frozen['source_producer_start_ticks']
                    and actual['cmdline_sha256'] == frozen['source_producer_cmdline_sha256'], 'same original producer, not a reused PID')
            if a.check_only:
                result = dict(status='passed_budget15_cutoff_preflight_not_stopped',worker_pid=pid,positions_committed=len(entries),
                    selected_positions=n,remaining=n-len(entries),source_producer_alive=actual is not None,signature=frozen)
                atomic_json(a.output/'summary.json',result)
                dump(result['status'],positions_committed=len(entries),selected_positions=n)
                print(json.dumps(result),flush=True)
                return
            if len(entries) == n:
                last = entries[-1]
                require(Path(last['path']).stat().st_size == last['bytes']
                    and all(Path(c['path']).is_file() for c in last['identity']['cell_records']), 'last selected cache fully committed')
                if actual is not None:
                    stop_owned(target,frozen['source_producer_start_ticks'],frozen['source_producer_cmdline_sha256'])
                    dump('stopping_original30_after_selected15_committed',producer_pid=target,positions_committed=n)
                    while identity(target) is not None:
                        current = identity(target)
                        if current is None or current['start_ticks'] != frozen['source_producer_start_ticks']:
                            break
                        time.sleep(1)
                payload = b'\n'.join(selected)+b'\n'
                (a.output/'cache_manifest.jsonl').write_bytes(payload)
                result = dict(status='complete_budget15_real_cell_prefix_execution_unaccepted',worker_pid=pid,
                    inputs=n,views=3*n,slides=15,source_cache=str(cache),source_signature=source_contract['signature'],
                    source_producer_pid=target,source_stopped_after_selected_prefix=True,
                    source_original30_complete=False,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),
                    selected_inputs_sha256=frozen['inputs_sha256'],signature=frozen,
                    scope='Original first8895 committed real-cell fields referenced without rerun or RNG/order change; old30 run remains partial.',
                    remaining='Independent all selected cell/cache acceptance and same15 method prediction/GT/BF1/statistics. Not full30 or accepted15 quality.')
                atomic_json(a.output/'summary.json',result)
                dump(result['status'],positions_committed=n)
                print(json.dumps(result),flush=True)
                return
            require(actual is not None, 'original producer stopped before required15 prefix; preserve and investigate, no retry')
            dump('waiting_selected15_real_cell_prefix',producer_pid=target,positions_committed=len(entries),
                selected_positions=n,remaining=n-len(entries))
            time.sleep(30)
    except Exception:
        dump('failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    run(p.parse_args())
