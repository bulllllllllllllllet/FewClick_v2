"""Prepare the authorized frozen 30-slide ablation extension and launch its queue."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from .full30 import ARMS, contract, read, validate_history
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.whole_slide_once.data import grid, gt_extent
from benchmarks.v5.wsi_quality_study.prepare import dump

BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
OLD = BASE / 'wsi_controls_full15_1116_214757'
ORIGINAL = BASE / 'whole_wsi_current_1112_171010'


def prepare(root):
    old = read(OLD / 'contract.json')
    original = read(ORIGINAL / 'contract.json')
    for path, digest in old['sources'].items():
        assert sha(path) == digest, path
    assert sha(ORIGINAL / 'cases.json') == old['original_cases_sha256']
    assert sha(ORIGINAL / 'unseen_audit.json') == original['audit_sha256']
    cases = read(ORIGINAL / 'cases.json')
    assert len(cases) == 30 and len({c['wsi_id'] for c in cases}) == 30
    assert cases[:15] == read(OLD / 'cases.json')
    assert all(c['eligible'] and not any(c['membership'].values()) for c in cases)
    audit = read(ORIGINAL / 'unseen_audit.json')
    assert audit['eligible'] == 30
    # Revalidate the current frozen consumption and development identities.
    bindings = {
        'actual_training': BASE / 'same_slide_training_1110_191022/training_consumption.jsonl',
        'development': BASE / 'wsi_same_slide_1108_013913/tasks.json',
        'support_plans': BASE / 'paper_closeout_1109_090821/confirmation_native_metadata/plans.json',
        'support_rows': BASE / 'paper_closeout_1109_090821/confirmation_native_metadata/query_rows.json',
    }
    for key, path in bindings.items():
        assert sha(path) == audit['sources'][key]
    windows, storage = {}, {}
    for index, case in enumerate(cases):
        assert Path(case['wsi_path']).is_file() and Path(case['gt_path']).is_file()
        for support in case['supports']:
            assert len(support['positive_xy']) == 3 and len(support['negative_xy']) == 2
            assert sha(support['support_labels']) == support['support_labels_sha256']
        groups, bw, bh = grid(case, case['supports'])
        windows[str(index)] = sum(len(xs) for _, xs in groups)
        packed = case['gt_height'] * math.ceil(case['gt_width'] / 8)
        window_bytes = 0
        for y, xs in groups:
            for x in xs:
                x0, y0, x1, y1 = gt_extent([x, y, bw, bh], case)
                window_bytes += math.ceil((x1-x0)*(y1-y0)/8)
        storage[str(index)] = (2*packed + window_bytes)*len(case['supports'])
    assert sum(len(c['supports']) for c in cases) == 307
    checkpoints = dict(old['checkpoints'], HG_mix=original['checkpoints']['HG_mix'])
    for checkpoint in checkpoints.values():
        assert sha(checkpoint['path']) == checkpoint['sha256']
    reuse = {m: {} for m in ARMS}
    for method in ARMS:
        for index in range(15):
            if method == 'HG_mix':
                spec = dict(old['reference_reports'][method][str(index)], kind='original')
            else:
                folder = (Path(old['reuse_root']) if index < 3 else OLD) / 'wsi' / method / f'case_{index:02}'
                path = folder / 'readouts/report.json'
                spec = dict(path=str(path), sha256=sha(path), kind='readouts')
            reuse[method][str(index)] = spec
    jobs = read(OLD / 'jobs.json')['jobs']
    rates = {}
    for method in ARMS[1:]:
        chosen = [j for j in jobs if j['method'] == method and not j['gate_only']]
        assert len(chosen) == 12 and all(j['status'] == 'complete' for j in chosen)
        rates[method] = sum(j['finished_at']-j['started_at'] for j in chosen) / sum(windows[str(j['case'])] for j in chosen)
    reports = [read(Path(old['reuse_root']) / 'wsi/HG_mix' / f'case_{i:02}/readouts/report.json') for i in range(3)]
    assert all(r['original_fused_mask_exact'] for r in reports)
    rates['HG_mix'] = sum(r['total_seconds'] for r in reports) / sum(r['windows'] for r in reports)
    new_windows = sum(windows[str(i)] for i in range(15, 30))
    gpu_hours = {m: rates[m]*new_windows/3600 for m in ARMS}
    estimated_bytes = sum(storage[str(i)] for i in range(15,30))*len(ARMS)
    free = shutil.disk_usage(root.parent).free
    assert free > estimated_bytes*1.25 + 120*1024**3, (free, estimated_bytes)
    root.mkdir(exist_ok=False)
    (root / 'tmp').mkdir()
    shutil.copy2(ORIGINAL / 'cases.json', root / 'cases.json')
    sources = dict(old['sources'])
    for path in [Path(__file__).resolve(), Path(__file__).with_name('full30.py').resolve()]:
        sources[str(path)] = sha(path)
    for path in [OLD/'summary.json', OLD/'contract.json', ORIGINAL/'contract.json', ORIGINAL/'unseen_audit.json', *bindings.values()]:
        sources[str(path)] = sha(path)
    c = dict(status='authorized_frozen_full30_preparing', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        methods=ARMS, slides=30, targets=307, new_cases=list(range(15,30)), new_method_slide_pairs=60,
        new_training=False, new_seed=False, sources=sources, cases_sha256=sha(root/'cases.json'),
        checkpoints=checkpoints, reuse_reports=reuse, original_summary=str(OLD/'summary.json'),
        original_cases=str(ORIGINAL/'cases.json'), support_budget=[3,2], query_human_clicks=0, threshold=0.5,
        gpus=list(range(6)), maximum_gpu_workers=6, minimum_free_MiB=12288,
        windows=windows, seconds_per_window=rates, estimated_GPU_hours=gpu_hours,
        estimated_ideal_six_GPU_hours=sum(gpu_hours.values())/6, expected_wall_hours=[24,30],
        estimated_new_mask_bytes=estimated_bytes, free_disk_bytes_at_prepare=free,
        source_case_order='Original pre-outcome SHA ordering, all30; previous first15 and extension15 reported separately',
        limitations='Original cohort has historical evaluation exposure; no new blind-test or independent-patient claim.')
    dump(root/'contract.json', c)
    snapshot = root/'source_snapshot'
    repo = Path.cwd()
    for filename in sources:
        path = Path(filename).resolve()
        if path.is_relative_to(repo):
            dest=snapshot/path.relative_to(repo); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    validate_history(root)
    dump(root/'preflight.json', dict(status='passed', source_hashes=len(sources), checkpoints_verified=ARMS,
        original_pool_audit_unchanged=True, actual_training_and_development_unchanged=True,
        selected_case_count=30, targets=307, new_targets=155, new_windows=new_windows,
        new_method_slide_pairs=60, historical_target_method_rows_replayed=608,
        estimated_GPU_hours=gpu_hours, ideal_six_GPU_hours=sum(gpu_hours.values())/6,
        estimated_storage_GiB=estimated_bytes/1024**3, expected_wall_hours=[24,30]))
    print(json.dumps(read(root/'preflight.json')), flush=True)


def launch(root):
    contract(root)
    assert read(root/'preflight.json')['status'] == 'passed'
    assert not (root/'launch.json').exists() and not (root/'jobs.json').exists()
    processes = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,args'],text=True)
    # No other FewClick compute worker may already consume this run's six-worker budget.
    live = [s for s in processes.splitlines() if ' -m benchmarks.v5.' in s and any(v in s for v in [' worker ', ' queue ', ' train '])]
    assert not live, live
    dump(root/'resource_snapshot.json', dict(processes=processes,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv'],text=True)))
    environment=dict(TMPDIR=str(root/'tmp'), PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:'+str(Path.cwd()),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VIPS_CONCURRENCY='2',
        VIPS_DISC_THRESHOLD='100m', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-u','-m','benchmarks.v5.reviewer_sample.full30','queue','--root',str(root)]
    log=root/'controller.log'; session=root.name
    shell=shlex.join(['env',*[k+'='+v for k,v in environment.items()],*command])+' > '+shlex.quote(str(log))+' 2>&1; rc=$?; echo EXIT_CODE=$rc >> '+shlex.quote(str(log))+'; exit $rc'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(Path.cwd()),shell],check=True)
    dump(root/'launch.json',dict(session=session,command=command,environment=environment,log=str(log)))
    print(json.dumps(read(root/'launch.json')),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--launch-existing',action='store_true')
    a=p.parse_args()
    if a.launch_existing:
        launch(a.root)
    else:
        prepare(a.root)


if __name__ == '__main__':
    main()
