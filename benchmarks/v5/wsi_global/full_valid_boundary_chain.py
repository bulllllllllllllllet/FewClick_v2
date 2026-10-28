"""Full-valid closed/reference and three image-method BF1, kept separate.

Every branch waits for its own actual prediction/native/remote-invariance receipt.
"""
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


METHODS = ('closed12', 'fewclick_m3d_image', 'sam_ft', 'wsi_sam_ft')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    files = [Path(__file__), Path('benchmarks/v5/fair_comparison/metrics.py')]
    files += [Path('benchmarks/v5/wsi_global') / n for n in (
        'completion.py', 'full_valid_boundary.py', 'streaming_boundary.py',
        'accept_full_valid_boundary.py', 'aggregate_full_valid_boundary.py')]
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
        dependency_worker_pids=[] if a.fixture else [3749768, 4140128],
        CPU_workers=1 if a.fixture else 4, threads_each=1,
        scope='Separate full-valid support-inclusive rows; physical Euclidean 2um BF1; '
              'independent disk recount and one-to-one join. No paired statistics.',
        failure_policy='Stop and preserve; no retry, resume or fallback'))
    try:
        gates = {}
        if not a.fixture:
            if a.verified_gate is None:
                raise ValueError('Actual four-branch full-valid boundary chain gate required')
            gate = read(a.verified_gate / 'summary.json')
            if (gate.get('status') != 'passed_full_valid_BF1_chain_gate'
                    or gate.get('sources') != sources
                    or [r['method'] for r in gate.get('results', [])] != list(METHODS)
                    or list(a.methods) != list(METHODS)):
                raise ValueError('Same-source complete four-branch gate, no narrowed formal run')
            gates = {r['method']: Path(r['output']) for r in gate['results']}
        results = []
        for method in a.methods:
            kind = 'closed12' if method == 'closed12' else 'interactive'
            if not a.fixture:
                dep_name, dep_pid, wanted = (
                    ('closed12_full_valid_raster_chain_1028_144400', 3749768,
                     'complete_closed12_full_valid_native_GT_remote_invariance') if kind == 'closed12' else
                    ('full_valid_prediction_chain_1028_160200', 4140128,
                     'complete_three_image_methods_full_valid_saved_GT_and_remote_invariance_qualified'))
                dep = a.root / dep_name
                while completed_receipt(Path(str(dep) + '.log'), dep_pid, dep / 'status.json', wanted) is None:
                    frozen()
                    status('waiting_owned_accepted_full_valid_predictions', method=method, dependency_worker_pid=dep_pid)
                    time.sleep(30)
            if kind == 'closed12':
                predictions = a.root / ('closed12_full_valid_raster_gate_1028_144000' if a.fixture else 'closed12_full_valid_raster_full_1028_144400')
                receipt = a.root / ('closed12_full_valid_raster_accept_gate_1028_144000' if a.fixture else 'closed12_full_valid_raster_acceptance_1028_144400')
            else:
                predictions = a.root / (f'full_valid_prediction_CPU4_gate_{method}_1028_155800' if a.fixture else f'full_valid_prediction_full_{method}_1028_160200')
                receipt = a.root / (f'full_valid_prediction_CPU4_accept_{method}_1028_155900' if a.fixture else f'full_valid_prediction_accept_full_{method}_1028_160200')
            shards = 1 if a.fixture else 4
            base = ['--root', a.root]
            producer_gate = (gates[method].parent / (method + '_producer_rank0')) if not a.fixture else None
            audit_gate = (gates[method].parent / (method + '_audit_rank0')) if not a.fixture else None
            producers = [a.output / (method + '_producer_rank' + str(r)) for r in range(shards)]
            audits = [a.output / (method + '_audit_rank' + str(r)) for r in range(shards)]
            execute_many('running_' + method + '_BF1_producers', [
                (out, 'benchmarks.v5.wsi_global.full_valid_boundary', base + [
                    '--source', predictions, '--acceptance', receipt, '--kind', kind,
                    '--rank', r, '--shards', shards, '--output', out] +
                    (['--gate'] if a.fixture else ['--verified-gate', producer_gate]))
                for r, out in enumerate(producers)])
            execute_many('running_' + method + '_independent_BF1_audits', [
                (out, 'benchmarks.v5.wsi_global.accept_full_valid_boundary', base + [
                    '--source', source, '--predictions', predictions,
                    '--prediction-acceptance', receipt, '--output', out] +
                    ([] if a.fixture else ['--verified-gate', audit_gate]))
                for source, out in zip(producers, audits)])
            target = a.output / (method + '_complete')
            execute_many('aggregating_' + method + '_BF1', [
                (target, 'benchmarks.v5.wsi_global.aggregate_full_valid_boundary',
                 ['--sources'] + producers + ['--acceptances'] + audits +
                 ['--predictions', predictions, '--prediction-acceptance', receipt,
                  '--output', target] + (['--gate'] if a.fixture else []))])
            result = read(target / 'summary.json')
            expected = 'accepted_full_valid_boundary_aggregation_gate' if a.fixture else 'accepted_complete_full_valid_physical_BF1_rows'
            if result['status'] != expected or (not a.fixture and result['rows'] != (360 if kind == 'closed12' else 11040 if method == 'fewclick_m3d_image' else 5520)):
                raise ValueError('Complete accepted BF1 row coverage required')
            results.append(dict(method=method, output=str(target), rows=result['rows'],
                                summary_sha256=sha(target / 'summary.json')))
        frozen()
        result = dict(status='passed_full_valid_BF1_chain_gate' if a.fixture else 'accepted_closed_and_three_image_methods_full_valid_physical_BF1',
                      worker_pid=pid, sources=sources, results=results,
                      remaining='Separate full-valid WSI statistics and fullcell actual input/cache/predictions/BF1; original strict-remote comparisons remain distinct')
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
