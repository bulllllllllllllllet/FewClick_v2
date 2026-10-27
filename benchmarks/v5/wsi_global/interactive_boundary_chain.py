"""Run frozen physical BF1 production, independent recount and complete joins."""
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


METHODS = ('fewclick_m3d_image', 'sam_ft', 'wsi_sam_ft')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    files = [Path(__file__), Path('benchmarks/v5/fair_comparison/metrics.py')]
    files += [Path('benchmarks/v5/wsi_global') / n for n in (
        'completion.py', 'wsi_boundary.py', 'streaming_boundary.py',
        'accept_wsi_boundary.py', 'aggregate_wsi_boundary.py')]
    sources = {str(p): sha(p) for p in files}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def frozen():
        for path, digest in sources.items():
            if sha(path) != digest:
                raise ValueError('Queued source changed: ' + path)

    def status(value, **extra):
        atomic_json(a.output / 'status.json', dict(status=value, worker_pid=pid, **extra))

    def execute_many(stage, plans):
        workers = []
        try:
            for target, module, args in plans:
                frozen()
                log = Path(str(target) + '.log').open('x')
                command = [sys.executable, '-u', '-m', module] + list(map(str, args))
                try:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close()
                    raise
                workers.append((child, log, target, command))
                status(stage, children=[dict(pid=w.pid, output=str(t), command=c)
                                        for w, _, t, c in workers])
            codes = []
            for child, log, _, _ in workers:
                rc = child.wait()
                log.write('\nEXIT_CODE=' + str(rc) + '\n')
                log.flush()
                codes.append(rc)
            if any(codes):
                raise RuntimeError('BF1 stage failed; outputs preserved: ' + str(codes))
        finally:
            for child, log, _, _ in workers:
                if child.poll() is not None:
                    log.close()

    atomic_json(a.output / 'contract.json', dict(
        worker_pid=pid, sources=sources, fixture=a.fixture, methods=list(a.methods),
        dependency_worker_pid=None if a.fixture else 4172727,
        CPU_workers=1 if a.fixture else 4, threads_each=1,
        scope='Original accepted prediction rows; physical Euclidean 2um BF1; '
              'independent disk recount and one-to-one join. No paired statistics.',
        failure_policy='Stop and preserve; no retry, resume or fallback'))
    try:
        if not a.fixture:
            gate = read(a.verified_gate / 'summary.json') if a.verified_gate else {}
            if (gate.get('status') != 'passed_interactive_BF1_chain_fixture'
                    or gate.get('sources') != sources
                    or [r['method'] for r in gate.get('results', [])] != list(METHODS)):
                raise ValueError('Actual three-method same-source BF1 chain fixture required')
            dep = a.root / 'interactive_full_queue_1028_011000'
            while completed_receipt(Path(str(dep) + '.log'), 4172727, dep / 'status.json',
                    'complete_three_methods_full30_saved_prediction_GT_score_acceptance_qualified') is None:
                frozen()
                status('waiting_owned_three_methods_full30_prediction_acceptance', dependency_worker_pid=4172727)
                time.sleep(30)
        results = []
        for method in a.methods:
            suffix = 'CPU_gate_' + method + '_1028_004500' if a.fixture else 'full_' + method + '_1028_011000'
            predictions = a.root / ('interactive_prediction_' + suffix)
            receipt = a.root / (('interactive_score_CPU_accept_' + method + '_1028_010500') if a.fixture
                                else ('interactive_score_full_accept_' + method + '_1028_011000'))
            shards = 1 if a.fixture else 4
            base = ['--root', a.root]
            producer_gate = a.root / 'WSI_boundary_interactive_driver_gate_1028_033000'
            audit_gate = a.root / 'WSI_boundary_interactive_accept_gate_1028_035000'
            producers = [a.output / (method + '_producer_rank' + str(r)) for r in range(shards)]
            audits = [a.output / (method + '_audit_rank' + str(r)) for r in range(shards)]
            execute_many('running_' + method + '_BF1_producers', [
                (out, 'benchmarks.v5.wsi_global.wsi_boundary', base + [
                    '--source', predictions, '--acceptance', receipt, '--kind', 'interactive',
                    '--rank', r, '--shards', shards, '--output', out] +
                    (['--gate'] if a.fixture else ['--verified-gate', producer_gate]))
                for r, out in enumerate(producers)])
            execute_many('running_' + method + '_independent_BF1_audits', [
                (out, 'benchmarks.v5.wsi_global.accept_wsi_boundary', base + [
                    '--source', source, '--predictions', predictions,
                    '--prediction-acceptance', receipt, '--output', out] +
                    ([] if a.fixture else ['--verified-gate', audit_gate]))
                for source, out in zip(producers, audits)])
            target = a.output / (method + '_complete')
            execute_many('aggregating_' + method + '_BF1', [
                (target, 'benchmarks.v5.wsi_global.aggregate_wsi_boundary',
                 ['--sources'] + producers + ['--acceptances'] + audits +
                 ['--predictions', predictions, '--prediction-acceptance', receipt,
                  '--output', target] + (['--gate'] if a.fixture else []))])
            result = read(target / 'summary.json')
            expected = 'accepted_WSI_boundary_aggregation_fixture' if a.fixture else 'accepted_complete_WSI_physical_BF1_rows'
            if result['status'] != expected or (not a.fixture and result['rows'] != (11040 if method == METHODS[0] else 5520)):
                raise ValueError('Complete accepted BF1 row coverage required')
            results.append(dict(method=method, output=str(target), rows=result['rows'],
                                summary_sha256=sha(target / 'summary.json')))
        frozen()
        result = dict(status='passed_interactive_BF1_chain_fixture' if a.fixture else 'accepted_three_methods_full30_physical_BF1',
                      worker_pid=pid, sources=sources, results=results,
                      remaining='Full-cell main30 BF1, paired WSI statistics and full-valid-domain extension')
        atomic_json(a.output / 'summary.json', result)
        status(result['status'])
        print(json.dumps(result), flush=True)
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS))
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--verified-gate', type=Path)
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        parser.error('Duplicate methods forbidden')
    run(args)
