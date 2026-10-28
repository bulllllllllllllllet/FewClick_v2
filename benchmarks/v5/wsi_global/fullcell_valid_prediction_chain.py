"""Wait for accepted real-cell full grid, then predict/audit/BF1 separately."""
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
        cache_gate = root/'fullcell_valid_prediction_cache_gate_1029_000500'
        cache_accept = root/'fullcell_valid_prediction_cache_acceptance_1029_000700'
        bf_gate = root/'fullcell_valid_BF1_gate_1029_001300'
        bf_accept = root/'fullcell_valid_BF1_accept_gate_1029_001400'
        bf_join = root/'fullcell_valid_BF1_join_gate_1029_001500'
        g, ga = read(cache_gate/'summary.json'), read(cache_accept/'summary.json')
        require(g['status'] == 'passed_full_valid_prediction_driver_gate' and g['rows'] == 48
            and ga['status'] == 'accepted_full_valid_prediction_gate' and ga['fresh_gate512_outputs'] == 48
            and ga['source_summary_sha256'] == sha(cache_gate/'summary.json'), 'actual real-cell GPU48 prediction/GT fixture')
        bg, ba, bj = read(bf_gate/'summary.json'), read(bf_accept/'summary.json'), read(bf_join/'summary.json')
        require(bg['status'] == 'passed_full_valid_boundary_driver_gate' and ba['status'] == 'accepted_full_valid_boundary_gate'
            and ba['source_summary_sha256'] == sha(bf_gate/'summary.json') and bj['status'] == 'accepted_full_valid_boundary_aggregation_gate'
            and bj['rows'] == 48, 'actual48 physical BF1 independent gate and complete join')
        paths = ['fullcell_valid_prediction_chain.py','fullcell_valid_prediction.py','accept_fullcell_valid_prediction.py',
            'fullcell_valid_boundary.py','accept_fullcell_valid_boundary.py','aggregate_fullcell_valid_boundary.py','completion.py']
        sources = {str(Path('benchmarks/v5/wsi_global')/p):sha(Path('benchmarks/v5/wsi_global')/p) for p in paths}
        sources.update(g['signature']['sources'])
        sources['benchmarks/v5/wsi_global/streaming_boundary.py'] = bg['signature']['kernel_sha256']
        sources['benchmarks/v5/fair_comparison/metrics.py'] = bg['signature']['frozen_metric_sha256']
        require(ga['recipe']['source_sha256'] == sources['benchmarks/v5/wsi_global/accept_fullcell_valid_prediction.py']
            and bg['signature']['source_sha256'] == sources['benchmarks/v5/wsi_global/fullcell_valid_boundary.py']
            and ba['recipe']['source_sha256'] == sources['benchmarks/v5/wsi_global/accept_fullcell_valid_boundary.py']
            and bj['source_sha256'] == sources['benchmarks/v5/wsi_global/aggregate_fullcell_valid_boundary.py'], 'same actually tested source versions')
        def frozen():
            for path, digest in sources.items():
                require(sha(path) == digest, 'frozen full-valid XCell prediction source: '+path)
        frozen()
        dependency = root/'fullcell_valid_chain_1028_235700'
        expected = 'complete_fullcell_valid_cache_index_qualified_predictions_pending'
        observe = lambda: completed_receipt(Path(str(dependency)+'.log'),2851935,dependency/'status.json',expected)
        current = observe()
        atomic_json(a.output/'contract.json', dict(worker_pid=pid,sources=sources,dependency_worker_pid=2851935,
            expected_dependency_status=expected,cache_fixture_acceptance_sha256=sha(cache_accept/'summary.json'),
            BF1_fixture_acceptance_sha256=sha(bf_accept/'summary.json'),BF1_join_sha256=sha(bf_join/'summary.json'),
            method='fewclick_m3d_xcell',rows=11040,full_grid_inputs=28452,
            scope='Source-inclusive full-valid diagnostic; formal full-index GPU gate still mandatory; compare every original remote512 digest and whole remote mask crop.',
            GPU='physical1 only when idle; no interruption',failure_policy='stop preserve; no retry/resume/fallback'))
        if a.check_only:
            result = dict(status='passed_fullcell_valid_prediction_chain_prerequisite_fixture_not_live_queue',
                worker_pid=pid,dependency_observed='complete' if current else 'live_wait',sources_checked=len(sources),
                remaining='Actual full28452 index, formal GPU48 gate,11040 predictions/scores/remote invariance/BF1 still pending')
            atomic_json(a.output/'summary.json',result)
            dump(result['status'])
            print(json.dumps(result),flush=True)
            return
        while current is None:
            frozen()
            dump('waiting_owned_full28452_real_cell_cache_index',dependency_worker_pid=2851935)
            time.sleep(30)
            current = observe()
        def idle():
            uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            while True:
                frozen()
                apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                occupied = [line for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
                if not occupied:
                    return
                dump('waiting_GPU1_idle_for_full_valid_XCell_prediction',observed_processes=occupied)
                time.sleep(30)
        def children(stage, jobs, gpu=False, threads='1'):
            if gpu:
                idle()
            frozen()
            workers = []
            for target, args in jobs:
                log = Path(str(target)+'.log').open('x')
                command = [sys.executable,'-u']+[str(x) for x in args]
                worker = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1' if gpu else '',OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads))
                workers.append((worker,log,target,command))
            dump(stage,children=[dict(pid=w.pid,output=str(t),command=c) for w,l,t,c in workers])
            codes = []
            for w,log,target,command in workers:
                rc = w.wait()
                codes.append(rc)
                log.write('\nEXIT_CODE='+str(rc)+'\n')
                log.close()
            require(not any(codes),'child failure, stop and preserve: '+str(codes))
        index = root/'fullcell_valid_index_full_1028_235600'
        primary = root/'fullcell_prediction_full_1028_034000'
        primary_accept = root/'fullcell_score_full_accept_1028_034000'
        gate = root/'fullcell_valid_prediction_GPU_gate_1029_001700'
        gate_accept = root/'fullcell_valid_prediction_GPU_gate_acceptance_1029_001700'
        full = root/'fullcell_valid_prediction_full_1029_001700'
        accepted = root/'fullcell_valid_prediction_acceptance_full_1029_001700'
        driver = ['-m','benchmarks.v5.wsi_global.fullcell_valid_prediction','--root',root,'--index',index,
            '--primary',primary,'--primary-acceptance',primary_accept]
        audit = ['-m','benchmarks.v5.wsi_global.accept_fullcell_valid_prediction','--root',root]
        children('formal_full_index_actual_GPU48_gate',[(gate,driver+['--gate','--output',gate])],True,'4')
        children('independent_formal_GPU48_gate_and_remote512',[(gate_accept,audit+['--source',gate,'--output',gate_accept])],True,'4')
        children('full28452_XCell_prediction_scores',[(full,driver+['--verified-gate',gate,'--gate-acceptance',gate_accept,'--output',full])],True,'4')
        children('independent_full_valid_masks_GT_and_all_remote_invariance',[(accepted,audit+['--source',full,'--verified-gate',gate_accept,'--output',accepted])],False,'4')
        receipt = read(accepted/'summary.json')
        require(receipt['status'] == 'accepted_full_valid_saved_predictions_GT_and_remote_invariance_qualified'
            and receipt['rows'] == 11040 and receipt['remote_mask_cells_checked'] > 0, 'all11040 scores and actual old remote crop accepted')
        producers = [root/f'fullcell_valid_BF1_rank{rank}_1029_001700' for rank in range(4)]
        audits = [root/f'fullcell_valid_BF1_accept_rank{rank}_1029_001700' for rank in range(4)]
        children('full_valid_XCell_physical_BF1_four_shards',[(target,['-m','benchmarks.v5.wsi_global.fullcell_valid_boundary',
            '--root',root,'--source',full,'--acceptance',accepted,'--kind','interactive','--verified-gate',bf_gate,
            '--rank',rank,'--shards',4,'--output',target]) for rank,target in enumerate(producers)])
        children('independent_full_valid_XCell_BF1_four_shards',[(target,['-m','benchmarks.v5.wsi_global.accept_fullcell_valid_boundary',
            '--root',root,'--source',source,'--predictions',full,'--prediction-acceptance',accepted,
            '--verified-gate',bf_accept,'--output',target]) for source,target in zip(producers,audits)])
        target = root/'fullcell_valid_BF1_complete_1029_001700'
        children('join_all11040_full_valid_XCell_BF1',[(target,['-m','benchmarks.v5.wsi_global.aggregate_fullcell_valid_boundary',
            '--sources']+producers+['--acceptances']+audits+['--predictions',full,'--prediction-acceptance',accepted,'--output',target])])
        require(read(target/'summary.json')['status'] == 'accepted_complete_full_valid_physical_BF1_rows'
            and read(target/'summary.json')['rows'] == 11040,'complete actual11040 BF1 rows')
        frozen()
        dump('complete_fullcell_full_valid_predictions_GT_remote_invariance_and_BF1_qualified',output=str(target),rows=11040,
            remaining='Separate full-valid paired statistics/report, provenance and reviewer closure; not overall goal completion')
    except Exception:
        dump('failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    run(p.parse_args())
