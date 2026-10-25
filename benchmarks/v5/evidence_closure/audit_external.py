"""Independent disk audit of complete external scores and saved native gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.metrics import VERSION, boundary, score
from benchmarks.v5.fair_comparison.protocol import nested


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(actual is None if expected is None else
            actual is not None and np.isclose(actual, expected, atol=1e-12, rtol=0), message)


def audit(tasks, report, prepared, output):
    contract = json.loads((tasks/'contract.json').read_text())
    require(digest(tasks/'fields.parquet') == contract['fields_sha256'], 'fields changed')
    require(digest(tasks/'local_maximum.json') == contract['clicks_sha256'], 'points changed')
    records = pd.read_parquet(tasks/'fields.parquet').to_dict('records')
    fields = {r['tile_id']: r for r in records}
    labels = {}
    for row in records:
        gt = np.asarray(Image.open(row['labels']))
        require(gt.shape == (512, 512) and hashlib.sha256(gt.tobytes()).hexdigest() ==
                row['labels_array_sha256'], 'actual GT identity')
        labels[row['tile_id']] = gt
        for entry in json.loads(row['images_json']):
            require(digest(entry['path']) == entry['sha256'], 'RGB changed')
            require(np.asarray(Image.open(entry['path'])).shape == (512, 512, 3), 'RGB geometry')
    plans = json.loads((tasks/'local_maximum.json').read_text())
    identity = lambda r: (r['pair_id'], r['prompt_group'], r['click_budget'])
    expected = {identity(p): p for maximum in plans for p in
                [nested(maximum, b) for b in contract['report_budgets']]}
    require(len(expected) == len(plans)*len(contract['report_budgets']), 'duplicate planned tasks')
    for plan in plans:
        require(plan['support_id'] == plan['query_id'], 'nonlocal external plan')
        gt = labels[plan['query_id']]
        for sign in ('positive', 'negative'):
            xy = np.asarray(plan[sign+'_xy'])
            require(np.isfinite(xy).all() and ((xy > 0) & (xy < 1)).all(), 'invalid point')
            pixel = np.floor(xy*512).astype(int)
            values = gt[pixel[:, 1], pixel[:, 0]]
            require((values == plan['target']).all() if sign == 'positive' else
                    ((values != plan['target']) & (values != 255)).all(), 'actual point sign')
    cells = json.loads((prepared/'summary.json').read_text())
    require(cells['status'] == 'complete' and not cells['gate'] and cells['fields'] == len(fields), 'cells incomplete')
    require(digest(prepared/'fields_with_cells.parquet') == cells['fields_sha256'], 'cell table changed')
    prepared_rows = pd.read_parquet(prepared/'fields_with_cells.parquet').to_dict('records')
    require({r['tile_id'] for r in prepared_rows} == set(fields) and len(prepared_rows) == len(fields), 'cell field coverage')
    cell_totals = [0, 0, 0]
    for row in prepared_rows:
        for key in ('images_json', 'labels_array_sha256', 'source_id', 'multiscale_json'):
            require(row[key] == fields[row['tile_id']][key], 'cell/source geometry identity')
        require(row['cell_cache_status'] == 'complete' and digest(row['cell_cache']) == row['cell_cache_sha256'], 'cell cache SHA')
        with np.load(row['cell_cache'], allow_pickle=False) as cache:
            for scale in range(3):
                features = cache[f'cells_{scale}']; total = int(cache[f'total_{scale}'])
                require(features.ndim == 2 and features.shape[1] == 74 and len(features) <= 255 and
                        total >= len(features) and np.isfinite(features).all(), 'cell feature/count contract')
                cell_totals[scale] += total
    summary = json.loads((report/'summary.json').read_text())
    require(summary['status'] == 'complete' and summary['protocol_sha256'] == digest(tasks/'contract.json'), 'report protocol')
    audited = {}; gt_stats = {}
    for plan in plans:
        k = plan['query_id'], plan['target']
        if k not in gt_stats:
            gt = labels[k[0]]; valid = gt != 255; truth = gt == k[1]
            gt_stats[k] = (int(valid.sum()), int((truth & valid).sum()), int(boundary(truth, valid).sum()))
    for method, record in summary['records'].items():
        run = Path(record['run']); meta = json.loads((run/'run.json').read_text())
        result = json.loads((run/'summary.json').read_text())
        require(result['status'] == 'complete' and not result['gate'] and meta['metric_version'] == VERSION, 'incomplete or gate run')
        require(meta['method'] == method and meta['protocol_sha256'] == digest(tasks/'contract.json'), 'method/protocol')
        row_sha = digest(run/'rows.jsonl')
        require(row_sha == result['rows_sha256'] == record['rows_sha256'], 'rows changed')
        rows = [json.loads(line) for line in (run/'rows.jsonl').read_text().splitlines()]
        table = {identity(r): r for r in rows}
        require(len(table) == len(rows) == result['rows'] and set(table) == set(expected), 'task coverage')
        for key, row in table.items():
            plan = expected[key]
            for field, value in plan.items():
                require(row[field] == value, 'saved plan differs: '+field)
            field = fields[row['query_id']]
            require(row['source_id'] == row['space_key'] == row['spatial_group_id'] == field['source_id'], 'source identity')
            valid, positive, true_boundary = gt_stats[row['query_id'], row['target']]
            require((row['valid'], row['positive'], row['true_boundary_pixels']) == (valid, positive, true_boundary), 'actual GT denominator')
            tp, fp, fn = (row[k] for k in ('tp', 'fp', 'fn'))
            require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)) and
                    tp+fn == positive and fp <= valid-positive, 'impossible counts')
            close(row['dice'], 2*tp/(2*tp+fp+fn) if positive else None, 'Dice recount')
            close(row['false_positive_fraction'], fp/valid, 'FPA recount')
            require(row['boundary_evaluable'] == (true_boundary > 0), 'boundary eligibility')
            if not true_boundary: require(row['bf1_um2'] is None, 'undefined BF1')
            else: require(row['bf1_um2'] is not None and 0 <= row['bf1_um2'] <= 1, 'invalid BF1')
            if true_boundary and row['predicted_boundary_pixels'] == 0: close(row['bf1_um2'], 0, 'missed boundary')
        evidence = dict(rows=len(rows), rows_sha256=row_sha, run_json_sha256=digest(run/'run.json'))
        if meta.get('development_training'):
            training = Path(meta['development_training']); trained = json.loads((training/'summary.json').read_text())
            best = json.loads((training/'best_checkpoint.json').read_text())
            require(trained['status'] == 'complete' and trained['step'] == 10000 and trained['full_manifest_parity'], 'training incomplete')
            consumed = digest(training/'training_consumed.jsonl')
            require(consumed == trained['training_consumed_sha256'] == 'fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', '40k task mismatch')
            require(Path(best['path']).resolve() == Path(meta['checkpoint']).resolve() and
                    digest(meta['checkpoint']) == best['checkpoint_sha256'] == meta['checkpoint_sha256'], 'development-selected weight mismatch')
            gate = run.with_name(run.name.replace('full_', 'gate_', 1))
            gate_meta = json.loads((gate/'run.json').read_text()); gate_result = json.loads((gate/'summary.json').read_text())
            require(gate_result['status'] == 'complete' and gate_result['gate'] and
                    gate_result['GT_unavailable_forward_passed'] and gate_result['saved_direct_count_replay'], 'native GT-forbidden gate')
            require(gate_meta['checkpoint_sha256'] == meta['checkpoint_sha256'] and
                    gate_meta['source_sha256'] == meta['source_sha256'] and gate_meta['device'] == meta['device'], 'gate identity')
            gate_rows = [json.loads(line) for line in (gate/'rows.jsonl').read_text().splitlines()]
            require(len(gate_rows) == gate_result['rows'] == 8 and digest(gate/'rows.jsonl') == gate_result['rows_sha256'], 'gate rows')
            for row in gate_rows:
                probability = np.load(gate/f'prediction_{row["pair_id"]}_{row["click_budget"]}.npy', allow_pickle=False)
                require(probability.shape == (512, 512) and np.isfinite(probability).all(), 'gate prediction')
                gt = labels[row['query_id']]; field = fields[row['query_id']]
                replay = score(probability >= .5, gt == row['target'], gt != 255, json.loads(field['input_mpp_xy_json']))
                for key, value in replay.items(): close(row[key], value, 'saved gate mask replay: '+key)
            evidence.update(checkpoint_sha256=meta['checkpoint_sha256'], consumed_sha256=consumed, saved_gate_masks=len(gate_rows))
        audited[method] = evidence
    write(output, dict(status='passed', methods=audited, fields=len(fields), planned_rows=len(expected),
        actual_GT_target_pairs=len(gt_stats), cells_total_by_scale=cell_totals,
        report_sha256=digest(report/'summary.json'), source_sha256=digest(__file__),
        scope='All complete row identities, actual GT denominators and count algebra; full-mask metric replay only for saved native gates, not all full predictions',
        limitations=['No regenerated full-run predictions', 'No patient independence or equal-total-pretraining claim']))
    print(json.dumps(dict(status='passed', methods=list(audited), rows_per_method=len(expected), output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('tasks', 'report', 'prepared', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.tasks, a.report, a.prepared, a.output)
