"""Defer full-valid real-cell extension until primary fullcell WSI closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .completion import atomic_json, completed_receipt


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    pid = os.getpid()
    root = a.root
    def dump(status, **values):
        atomic_json(a.output/'status.json', dict(status=status, worker_pid=pid, **values))
    try:
        gate = root/'fullcell_valid_extension_gate_1028_234600'
        accepted = root/'fullcell_valid_extension_gate_acceptance_1028_234900'
        fixture = root/'fullcell_valid_index_fixture_1028_235400'
        fixture_audit = root/'fullcell_valid_index_fixture_acceptance_1028_235411'
        g, ga = read(gate/'summary.json'), read(accepted/'summary.json')
        f, fa = read(fixture/'summary.json'), read(fixture_audit/'summary.json')
        require(g['status'] == 'passed_fullcell_valid_extension_gate'
            and ga['status'] == 'accepted_fullcell_valid_extension_gate'
            and ga['source_summary_sha256'] == sha(gate/'summary.json'), 'actual accepted new8 cells gate')
        require(f['status'] == 'passed_fullcell_valid_index_fixture'
            and fa['status'] == 'accepted_fullcell_valid_index_fixture_only'
            and fa['source_summary_sha256'] == sha(fixture/'summary.json'), 'actual16 index and independent files fixture')
        paths = [Path(__file__), Path('benchmarks/v5/wsi_global/merge_fullcell_valid.py'),
            Path('benchmarks/v5/wsi_global/accept_fullcell_valid_extension.py'), Path('benchmarks/v5/wsi_global/completion.py')]
        sources = {str(p):sha(p) for p in paths}
        sources.update(g['signature']['sources'])
        sources.update(ga['dependencies'])
        require(ga['source_sha256'] == sources['benchmarks/v5/wsi_global/accept_fullcell_valid_extension.py']
            and f['recipe']['source_sha256'] == sources['benchmarks/v5/wsi_global/merge_fullcell_valid.py'], 'same actual validated source versions')
        def frozen():
            for p, digest in sources.items():
                require(sha(p) == digest, 'frozen extension source: '+p)
        frozen()
        dependency = root/'fullcell_prediction_chain_1028_034000'
        expected = 'complete_fullcell_main30_saved_prediction_scores_and_BF1_qualified'
        observe = lambda: completed_receipt(Path(str(dependency)+'.log'), 758578, dependency/'status.json', expected)
        current = observe()
        contract = dict(worker_pid=pid, sources=sources, dependency_worker_pid=758578,
            dependency_status=expected, GPU='physical1 when genuinely idle; never interrupt existing tasks',
            new_inputs=7996, merged_inputs=28452, gate_summary_sha256=sha(gate/'summary.json'),
            gate_acceptance_sha256=sha(accepted/'summary.json'), index_fixture_sha256=sha(fixture/'summary.json'),
            index_fixture_acceptance_sha256=sha(fixture_audit/'summary.json'),
            failure_policy='stop preserve; no retry/resume/fallback', primary_sources_modified=False)
        atomic_json(a.output/'contract.json', contract)
        if a.check_only:
            result = dict(status='passed_fullcell_valid_chain_prerequisite_fixture_not_live_queue', worker_pid=pid,
                dependency_observed='complete' if current else 'live_wait', sources_checked=len(sources),
                remaining='Actual queue, original primary completion, new7996 extension/acceptance/merge still required')
            atomic_json(a.output/'summary.json', result)
            dump(result['status'])
            print(json.dumps(result), flush=True)
            return
        while current is None:
            frozen()
            dump('waiting_owned_primary_fullcell_prediction_scores_and_BF1', dependency_worker_pid=758578)
            time.sleep(30)
            current = observe()
        def idle_GPU():
            uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
            while True:
                frozen()
                apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
                occupied = [line for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
                if not occupied:
                    return
                dump('waiting_GPU1_idle_for_fullcell_valid_extension', observed_processes=occupied)
                time.sleep(30)
        def child(stage, target, args, gpu=False):
            if gpu:
                idle_GPU()
            frozen()
            command = [sys.executable,'-u']+[str(x) for x in args]
            with Path(str(target)+'.log').open('x') as log:
                worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='1' if gpu else '', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                dump(stage, child_pid=worker.pid, command=command, output=str(target))
                rc = worker.wait()
                log.write('\nEXIT_CODE='+str(rc)+'\n')
            require(rc == 0, 'child failed, preserve and stop: '+str(target))
        full = root/'fullcell_valid_extension_full_1028_235600'
        full_accept = root/'fullcell_valid_extension_full_acceptance_1028_235600'
        index = root/'fullcell_valid_index_full_1028_235600'
        child('running_new7996_fullcell_prepare', full, ['-m','benchmarks.v5.wsi_global.fullcell_valid_extension',
            '--root',root,'--verified-gate',gate,'--gate-acceptance',accepted,'--output',full], True)
        child('independent_new7996_fullcell_cache_acceptance', full_accept, ['-m','benchmarks.v5.wsi_global.accept_fullcell_valid_extension',
            '--root',root,'--source',full,'--gate-acceptance',accepted,'--output',full_accept], True)
        require(read(full_accept/'summary.json')['status'] == 'accepted_fullcell_valid_extension_qualified', 'all actual7996 accepted')
        child('merging_accepted_full28452_real_cell_cache_index', index, ['-m','benchmarks.v5.wsi_global.merge_fullcell_valid',
            '--root',root,'--base',root/'fullcell_CRC_cache_full_1028_031000',
            '--base-acceptance',root/'fullcell_CRC_cache_accept_full_1028_031000',
            '--extension',full,'--extension-acceptance',full_accept,'--verified-fixture',fixture,'--output',index])
        require(read(index/'summary.json')['status'] == 'accepted_fullcell_valid_cache_index_qualified'
            and read(index/'summary.json')['items'] == 28452, 'complete actual full-cell index')
        frozen()
        dump('complete_fullcell_valid_cache_index_qualified_predictions_pending', output=str(index), inputs=28452,
             remaining='Actual full-valid XCell predictions/native/GT/BF1/statistics and original-remote invariance remain required')
    except Exception:
        dump('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    run(p.parse_args())
