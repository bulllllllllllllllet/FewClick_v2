"""Recount heldout transfer rows and replay each selected-weight gate mask."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .audit_heldout_local import read, sha, require, close, key
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.transfer_comparison.data import TransferData


def run(root, protocols, local, output):
    output.mkdir(parents=True, exist_ok=False)
    def save(name, d):
        (output/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
    save('status.json', dict(status='running', worker_pid=os.getpid()))
    accepted = read(local/'summary.json')
    require(accepted['status'] == 'accepted_local_counts_and_binding_qualified', 'accepted local weight bindings required')
    complete = read(root/'summary.json')
    require(complete['status'] == 'complete_remote_execution' and root.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'remote execution terminal')
    checks = []; total = saved_masks = 0; started = time.time()
    try:
        for fold in ('necrosis', 'adipose', 'lymphoid_related'):
            tasks = protocols/fold/'heldout_remote'; data = TransferData(tasks)
            plans = read(tasks/'validation_maximum.json')
            expected = {key(p): p for m in plans for p in [nested(m, b) for b in data.contract['report_budgets']]}
            for method in ('fewclick_m3d_image', 'sam_ft', 'wsi_sam_ft'):
                name = fold+'|'+method
                binding = accepted['conditions'][name]
                gate_folder = Path(complete['complete'][name+'|gate']); gate_meta = read(gate_folder/'run.json')
                for stage in ('gate', 'score'):
                    folder = Path(complete['complete'][name+'|'+stage]); done = read(folder/'summary.json'); meta = read(folder/'run.json')
                    require(done['status'] == 'complete' and not done['checkpoint_gate'] and done['gate'] == (stage == 'gate'), 'full selected checkpoint gate/score scope')
                    require(meta['checkpoint_sha256'] == binding['checkpoint_sha256'] == sha(meta['checkpoint']), 'local/remote selected weight binding')
                    require(meta['actual_training_plans_sha256'] == binding['training_coordinate_sha256'], 'actual current-fold training')
                    require(meta['protocol_sha256'] == sha(str(tasks/'contract.json')), 'remote test contract SHA')
                    require(meta['source_sha256'] == gate_meta['source_sha256'] and meta['checkpoint_sha256'] == gate_meta['checkpoint_sha256'], 'each weight has its own same-source gate')
                    for p, value in meta['source_sha256'].items():
                        require(sha(p) == value, 'inference source changed')
                    require(meta['query_human_clicks'] == 0 and meta['query_class_ID_in_model'] is False and meta['GT_forbidden_input_API'] and meta['remote_supervised_updates'] == 0, 'remote information contract')
                    require(meta['threshold'] == .5 and meta['prototype_temperature'] == .1, 'frozen thresholds')
                    coverage = read(folder/'query_annotation_coverage.json')
                    for query, value in coverage.items():
                        gt = data.labels(query)
                        require(value == dict(valid_pixels=int((gt != 255).sum()), total_pixels=int(gt.size)), 'query annotation coverage')
                    if stage == 'gate':
                        require(not any(done['native_cached_errors']) and done['saved_count_replay'] and done['GT_forbidden_input_API'], 'native/cache/saved replay gate')
                    require(folder.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'stage exit')
                    for variant, out in done['outputs'].items():
                        out = Path(out); stats = read(out/'summary.json')
                        rows = [json.loads(line) for line in (out/'rows.jsonl').open()]
                        require(sha(str(out/'rows.jsonl')) == stats['rows_sha256'], 'row file SHA')
                        seen = set()
                        for index, row in enumerate(rows):
                            k = key(row)
                            require(k not in seen and k in expected, 'duplicate or unexpected task')
                            seen.add(k)
                            require(all(row.get(f) == value for f, value in expected[k].items()), 'task/point identity')
                            truth = data.supervision(row); t = truth['truth']; v = truth['valid']
                            positive = int((t & v).sum()); valid = int(v.sum())
                            require(row['positive'] == positive and row['valid'] == valid, 'actual original GT denominator')
                            tp, fp, fn = [row[f] for f in ('tp', 'fp', 'fn')]
                            require(all(type(n) is int and n >= 0 for n in (tp, fp, fn)) and tp+fn == positive and fp <= valid-positive, 'count algebra')
                            if positive:
                                close(row['dice'], 2*tp/(2*tp+fp+fn), 'present Dice')
                            else:
                                require(row['dice'] is None, 'absent Dice must be NA')
                            close(row['false_positive_fraction'], fp/valid, 'valid-area FP fraction')
                            if stage == 'gate':
                                p = np.load(out/f'prediction_{index:04d}.npy', allow_pickle=False) >= .5
                                require(p.shape == t.shape, 'saved probability shape')
                                require((int((p&t&v).sum()), int((p&~t&v).sum()), int((~p&t&v).sum())) == (tp, fp, fn), 'independent saved pixel recount')
                                saved_masks += 1
                        if stage == 'score':
                            require(seen == set(expected), 'full frozen task coverage')
                            total += len(rows)
                        metrics = {}
                        for budget in data.contract['report_budgets']:
                            selected = [r for r in rows if r['click_budget'] == budget]
                            pairs = defaultdict(list)
                            for r in selected:
                                if r['positive']:
                                    pairs[r['pair_id']].append(r['dice'])
                            dice = math.fsum(math.fsum(v)/len(v) for v in pairs.values())/len(pairs)
                            negative = [r['false_positive_fraction'] for r in selected if not r['positive']]
                            fpa = math.fsum(negative)/len(negative)
                            reported = stats['metrics'][budget]['overall']
                            close(dice, reported['present_dice'], 'pair macro Dice recount')
                            close(fpa, reported['absent_FPA'], 'absent FPA recount')
                            metrics[budget] = dict(dice=dice, absent_FPA=fpa)
                        checks.append(dict(fold=fold, method=method, variant=variant, stage=stage, rows=len(rows), metrics=metrics,
                            checkpoint_sha256=meta['checkpoint_sha256'], rows_sha256=stats['rows_sha256']))
                    save('status.json', dict(status='running', worker_pid=os.getpid(), completed_outputs=len(checks), full_rows=total, saved_masks=saved_masks))
                    print(json.dumps(dict(condition=name, stage=stage, outputs=len(done['outputs']))), flush=True)
        require(total == 19360, 'all four output tracks across three folds')
        save('summary.json', dict(status='accepted_remote_counts_binding_and_native_saved_gates_qualified', checks=checks,
            full_rows=total, saved_masks=saved_masks, worker_pid=os.getpid(), seconds=time.time()-started, source_sha256=sha(__file__),
            limits=['single seed; paired statistics pending', 'query unknown pixels excluded, absent not certified whole image negative',
                    'SAM and WSI-SAM feature prototypes, not native remote mask decoders', 'full masks not regenerated; each native gate saved mask independently recounted']))
        save('status.json', dict(status='accepted', worker_pid=os.getpid(), full_rows=total, saved_masks=saved_masks))
    except Exception as exc:
        save('status.json', dict(status='failed', worker_pid=os.getpid(), error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root', 'protocols', 'local', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.protocols, a.local, a.output)
