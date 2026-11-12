"""Three paired pilots, measured ETA, then formal slides behind SAM priority."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha


def estimate(cases, reports, indices):
    methods = ['HG_mix', 'sansa']
    costs = {}
    for method in methods:
        rr = [r for r in reports if r['method'] == method]
        n = sum(r['windows'] for r in rr)
        enc = sum(r['timings']['query_encoding_seconds'] for r in rr)/n
        dec = sum(r['timings']['decode_seconds'] for r in rr)/n
        rest = max(0., sum(r['total_seconds'] for r in rr)/n-enc-dec)
        # Pilots have one target. Native SANSA decode includes its image encoder.
        costs[method] = dict(encoding_per_window=enc, per_target_per_window=dec+rest)
    seconds = 0.
    windows = 0
    for i in indices:
        case = cases[i]
        fine = case['supports'][0]['geometry']['native_boxes'][0]
        bw, bh = fine[2:]
        nx = 1+__import__('math').ceil((case['width']-bw)/max(1, bw//2))
        ny = 1+__import__('math').ceil((case['height']-bh)/max(1, bh//2))
        count = nx*ny+len(case['supports'])
        windows += count
        for cost in costs.values():
            seconds += count*(cost['encoding_per_window']+len(case['supports'])*cost['per_target_per_window'])
    return dict(estimated_query_windows=windows, serial_seconds=seconds,
                three_worker_hours=seconds/3/3600, with25pct_margin_hours=seconds/3/3600*1.25,
                measured_pilot_costs=costs, conditional_shared_GPU_IO_estimate=True)


def aggregate(root, phase, indices):
    from collections import defaultdict
    import numpy as np
    by = defaultdict(lambda: defaultdict(list))
    rows = []
    for index in indices:
        case = root/phase/f'case_{index:02}'
        a = json.loads((case/'acceptance.json').read_text())
        assert a['status'] == 'complete_accepted_paired_WSI_Dice'
        for method in ['HG_mix', 'sansa']:
            out = case/method
            receipt = json.loads((out/'acceptance.json').read_text())
            assert sha(out/'report.json') == receipt['report_sha256']
            report = json.loads((out/'report.json').read_text())
            for target, result in report['targets'].items():
                assert sha(out/f'target_{target}/mask.npy') == result['mask_sha256']
                by[method][target].append(result)
                rows.append(dict(method=method, slide=report['slide'], target=int(target),
                                 whole_WSI=result['whole_WSI'], remote_only_WSI=result['remote_only_WSI']))
    summaries = {}
    for method, targets in by.items():
        classes = {target: dict(slides=len(values),
                    whole_Dice=float(np.mean([r['whole_WSI']['dice'] for r in values])),
                    remote_Dice=float(np.mean([r['remote_only_WSI']['dice'] for r in values if r['remote_only_WSI']['dice'] is not None])))
                   for target, values in targets.items()}
        summaries[method] = dict(classes=classes,
             class_macro_whole_Dice=float(np.mean([r['whole_Dice'] for r in classes.values()])),
             class_macro_remote_Dice=float(np.mean([r['remote_Dice'] for r in classes.values()])))
    dump(root/(phase+'_report.json'), dict(status='complete_accepted_whole_slide_Dice', slides=len(indices),
         methods=summaries, rows=rows, unique_pixel_WSI_metrics=True,
         limitations='Not patch-average Dice; historical AAAI test exposure, no membership in audited train/val pools. Physical full-WSI BF1 pending.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    c = json.loads((root/'contract.json').read_text())
    cases = json.loads((root/'cases.json').read_text())
    records = []

    def launch(index, phase, gpu):
        for f, digest in c['sources'].items():
            assert sha(f) == digest, f
        command = [sys.executable, '-u', '-m', 'benchmarks.v5.whole_slide_once.worker',
                   '--root', str(root), '--case-index', str(index), '--phase', phase]
        log = (root/f'{phase}_{index:02}.log').open('x')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        item = dict(pid=proc.pid, gpu=gpu, case_index=index, phase=phase, command=command, status='running')
        records.append(item)
        dump(root/'workers.json', records)
        return proc, log, item

    def batch(indices, phase):
        pending = list(indices)
        active = []
        for gpu in c['gpus']:
            if pending:
                active.append(launch(pending.pop(0), phase, gpu))
        while active:
            for job in list(active):
                proc, log, item = job
                code = proc.poll()
                if code is None:
                    continue
                log.write(f'\nEXIT_CODE={code}\n')
                log.close()
                item.update(status='complete' if code == 0 else 'failed', exit_code=code)
                dump(root/'workers.json', records)
                active.remove(job)
                assert code == 0, item
                if pending:
                    active.append(launch(pending.pop(0), phase, item['gpu']))
            if active:
                time.sleep(5)

    try:
        dump(root/'stage.json', dict(status='three_pilot_engineering_and_inference_running'))
        batch([0, 1, 2], 'pilot')
        aggregate(root, 'pilot', [0, 1, 2])
        pilot = [json.loads((root/'pilot'/f'case_{index:02}'/method/'report.json').read_text())
                 for index in [0, 1, 2] for method in ['HG_mix', 'sansa']]
        estimates = dict(formal15=estimate(cases, pilot, list(range(15))),
                         additional15=estimate(cases, pilot, list(range(15, 30))))
        dump(root/'estimated_time.json', estimates)
        # Protect SAM priority and avoid filling the six-slot pool ahead of its eval shards.
        while True:
            stages = [json.loads((Path(c['sam_priority_root'])/arm/'controller_stage.json').read_text())['status']
                      for arm in ['sam', 'sam2', 'wsi']]
            if all(s in ['complete_accepted', 'failed_stop_no_retry'] for s in stages):
                break
            dump(root/'stage.json', dict(status='pilots_complete_waiting_for_SAM_priority', baseline_stages=stages,
                                         estimates=estimates))
            time.sleep(30)
        dump(root/'stage.json', dict(status='formal15_running', estimates=estimates))
        batch(list(range(15)), 'formal')
        aggregate(root, 'formal', list(range(15)))
        extra_hours = estimates['additional15']['with25pct_margin_hours']
        packed = sum(case['gt_width']*case['gt_height']/8*len(case['supports'])*2 for case in cases[15:30])
        expand = extra_hours <= 12 and packed*1.5 < shutil.disk_usage(root).free
        dump(root/'expansion_decision.json', dict(expand_to30=expand, estimated_extra_hours=extra_hours,
              estimated_extra_packed_bytes=packed, criterion='additional15 paired all-target ETA<=12h and1.5x packed-mask disk margin; all30 already audited train/val-unseen'))
        if expand:
            dump(root/'stage.json', dict(status='additional15_running'))
            batch(list(range(15, 30)), 'extra')
            aggregate(root, 'extra', list(range(15, 30)))
        dump(root/'stage.json', dict(status='complete_accepted_Dice_workflow', formal_slides=30 if expand else 15,
                                     training_updates=0, BF1='pending separate physical boundary study'))
    except Exception as error:
        dump(root/'stage.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
