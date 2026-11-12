"""Queue four read-only WSI baselines after final matched-adaptation acceptance."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha


def aggregate(root, c, phase, indices):
    rows = []
    methods = {}
    for method in c['adaptations']:
        classes = {}
        for i in indices:
            out = root/method/phase/f'case_{i:02}'/'prediction'
            receipt = json.loads((out/'acceptance.json').read_text())
            assert sha(out/'report.json') == receipt['report_sha256']
            report = json.loads((out/'report.json').read_text())
            assert report['independent_disk_score_exact'] and report['full_output_covered']
            for target, result in report['targets'].items():
                assert sha(out/f'target_{target}/mask.npy') == result['mask_sha256']
                classes.setdefault(target, []).append(result)
                rows.append(dict(method=method, slide=report['slide'], target=int(target),
                            whole_WSI=result['whole_WSI'], remote_only_WSI=result['remote_only_WSI']))
        summary = {}
        for target, values in classes.items():
            summary[target] = {}
            for scope in ['whole_WSI', 'remote_only_WSI']:
                valid = [v[scope]['dice'] for v in values if v[scope]['dice'] is not None]
                summary[target][scope] = dict(valid_slides=len(valid), Dice=sum(valid)/len(valid) if valid else None)
        means = {}
        for scope in ['whole_WSI', 'remote_only_WSI']:
            valid = [v[scope]['Dice'] for v in summary.values() if v[scope]['Dice'] is not None]
            means[scope] = sum(valid)/len(valid) if valid else None
        methods[method] = dict(classes=summary, class_macro_Dice=means)
    dump(root/(phase+'_report.json'), dict(status='complete_accepted_four_SAM_WSI',
         slides=len(indices), methods=methods, rows=rows, cases_sha256=c['cases_sha256'],
         whole_WSI_unique_pixels=True, query_human_clicks=0, support_budget='3+2'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    c = json.loads((root/'contract.json').read_text())
    records = []
    try:
        while True:
            states = {}
            for arm, path in c['adaptations'].items():
                f = Path(path)/'controller_stage.json'
                states[arm] = json.loads(f.read_text())['status'] if f.exists() else 'queued'
            assert not any(s == 'failed_stop_no_retry' for s in states.values()), states
            if all(s == 'complete_accepted' for s in states.values()):
                break
            dump(root/'stage.json', dict(status='waiting_final_fair_adaptations_and_patch_evaluations',
                                        adaptations=states, controller_pid=os.getpid()))
            time.sleep(30)

        def batch(indices, phase):
            pending = [(arm, i) for i in indices for arm in c['adaptations']]
            active = []

            def launch(gpu):
                arm, i = pending.pop(0)
                for f, digest in c['sources'].items():
                    assert sha(f) == digest, f
                cmd = [sys.executable, '-u', '-m', 'benchmarks.v5.whole_slide_sam.worker',
                       '--root', str(root), '--arm', arm, '--case-index', str(i), '--phase', phase]
                log = (root/f'{phase}_{arm}_{i:02}.log').open('x')
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
                record = dict(pid=proc.pid, arm=arm, case_index=i, gpu=gpu, phase=phase,
                              command=cmd, status='running')
                records.append(record)
                dump(root/'workers.json', records)
                active.append((proc, log, record))

            for gpu in c['gpus']:
                if pending:
                    launch(gpu)
            while active:
                for proc, log, record in list(active):
                    code = proc.poll()
                    if code is None:
                        continue
                    log.write(f'\nEXIT_CODE={code}\n')
                    log.close()
                    record.update(exit_code=code, status='complete' if code == 0 else 'failed')
                    dump(root/'workers.json', records)
                    active.remove((proc, log, record))
                    assert code == 0, record
                    if pending:
                        launch(record['gpu'])
                if active:
                    time.sleep(5)

        dump(root/'stage.json', dict(status='four_baselines_three_pilots_running'))
        batch(range(3), 'pilot')
        aggregate(root, c, 'pilot', range(3))
        dump(root/'stage.json', dict(status='four_baselines_formal15_running'))
        batch(range(15), 'formal')
        aggregate(root, c, 'formal', range(15))
        upstream = Path(c['whole_wsi_root'])
        while not (upstream/'expansion_decision.json').exists():
            state = json.loads((upstream/'stage.json').read_text())['status']
            assert state != 'failed_stop_no_retry', state
            dump(root/'stage.json', dict(status='formal15_complete_waiting_common_cohort_expansion'))
            time.sleep(30)
        if json.loads((upstream/'expansion_decision.json').read_text())['expand_to30']:
            dump(root/'stage.json', dict(status='four_baselines_additional15_running'))
            batch(range(15, 30), 'extra')
            aggregate(root, c, 'extra', range(15, 30))
        for phase in ['formal', 'extra']:
            if not (root/(phase+'_report.json')).exists():
                continue
            while not (upstream/(phase+'_report.json')).exists():
                state = json.loads((upstream/'stage.json').read_text())['status']
                assert state != 'failed_stop_no_retry', state
                dump(root/'stage.json', dict(status='waiting_HG_SANSA_common_report', phase=phase))
                time.sleep(30)
            baseline = json.loads((root/(phase+'_report.json')).read_text())
            original = json.loads((upstream/(phase+'_report.json')).read_text())
            assert original['status'] == 'complete_accepted_whole_slide_Dice'
            rows = baseline['rows']+original['rows']
            groups = {}
            for row in rows:
                key = (row['slide'], row['target'])
                groups.setdefault(row['method'], {})[key] = row
            assert set(groups) == {'sam', 'sam2', 'wsi', 'med2d', 'HG_mix', 'sansa'}
            reference = groups['HG_mix']
            for method, entries in groups.items():
                assert entries.keys() == reference.keys(), method
                for key, row in entries.items():
                    for scope in ['whole_WSI', 'remote_only_WSI']:
                        for field in ['valid', 'positive']:
                            assert row[scope][field] == reference[key][scope][field], (method, key, scope, field)
            dump(root/(phase+'_six_method_comparison.json'), dict(status='complete_accepted_six_method_common_cohort',
                 rows=rows, cases_sha256=c['cases_sha256'], matched_targets=len(reference),
                 source_reports={str(root/(phase+'_report.json')):sha(root/(phase+'_report.json')),
                                 str(upstream/(phase+'_report.json')):sha(upstream/(phase+'_report.json'))},
                 all_methods_same_valid_and_positive_pixels=True, query_human_clicks=0, support_budget='3+2'))
        dump(root/'stage.json', dict(status='complete_accepted_four_baseline_WSI', training_updates=0,
                                     six_method_join='Use common cases/targets with upstream HG_mix/SANSA reports'))
    except Exception as error:
        dump(root/'stage.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
