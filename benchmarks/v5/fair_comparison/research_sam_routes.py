"""Read-only evidence aggregation for candidate research, not a new benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


ROOT = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def read(path):
        bindings[str(path)] = sha(path)
        return json.loads(path.read_text())

    previous = ROOT / 'overall_gap_review_1029_172933'
    old = read(previous / 'statistics/summary.json')
    read(previous / 'independent_acceptance.json')
    formal = ROOT / 'dense_generalization_1029_190120/formal'
    accepted = read(formal / 'independent_acceptance.json')
    assert accepted['status'] == 'accepted_dense_generalization_training_qualified'
    assert sha(formal / 'summary.json') == accepted['summary_sha256']
    rows = {role: read(formal / role / 'rows.json')
            for role in ('development', 'matched_training')}
    selection = read(ROOT / 'matched_dense_diagnosis_1029_181751/selection.json')
    arms = ('original', 'continued_original_loss', 'color_augmentation')

    def key(row):
        return row['dataset'], row['query_id'], row['target']

    indexed = {}
    for role, records in rows.items():
        indexed[role] = {}
        expected = 302 if role == 'development' else 29
        for arm in arms:
            subset = [r for r in records if r['arm'] == arm]
            assert len(subset) == expected
            indexed[role][arm] = {key(r): r for r in subset}
            assert len(indexed[role][arm]) == expected
            for row in subset:
                denominator = 2 * row['tp'] + row['fp'] + row['fn']
                assert denominator > 0
                assert math.isclose(2 * row['tp'] / denominator, row['dice'], abs_tol=1e-12)
        assert all(set(indexed[role][arm]) == set(indexed[role]['original']) for arm in arms)
        for identity in indexed[role]['original']:
            truth = indexed[role]['original'][identity]
            for arm in arms:
                row = indexed[role][arm][identity]
                assert (row['positive'], row['valid'], row['GT_fraction']) == (
                    truth['positive'], truth['valid'], truth['GT_fraction'])

    def aggregate(records):
        return dict(n=len(records), dice=statistics.fmean(r['dice'] for r in records),
                    precision=statistics.fmean(r['tp'] / max(1, r['tp'] + r['fp']) for r in records),
                    recall=statistics.fmean(r['tp'] / (r['tp'] + r['fn']) for r in records))

    matched = {}
    assert len(selection['pairs']) == 29
    for group in ('all', 'small', 'non_small'):
        pairs = [p for p in selection['pairs'] if group == 'all'
                 or (p['val_fraction'] < .05) == (group == 'small')]
        matched[group] = {}
        for arm in arms:
            train = [indexed['matched_training'][arm][key(p['train'])] for p in pairs]
            val = [indexed['development'][arm][key(p['val'])] for p in pairs]
            matched[group][arm] = dict(train=aggregate(train), val=aggregate(val))
    for arm in arms:
        for group, predicate in [('all', lambda r: True), ('small', lambda r: r['GT_fraction'] < .05),
                                 ('large', lambda r: r['GT_fraction'] >= .5)]:
            result = aggregate([r for r in indexed['development'][arm].values() if predicate(r)])
            for field in ('n', 'dice', 'precision', 'recall'):
                assert math.isclose(result[field], accepted['groups'][arm][group][field], abs_tol=1e-12)

    fc = old['methods']['fewclick_m3d_xcell']
    sam = old['methods']['sam_ft']
    fsmall = fc['budgets']['3+2']['by_area']['<5%']
    ssmall = sam['budgets']['3+2']['by_area']['<5%']
    weight = fsmall['pairs'] / fc['primary']['pairs']
    arithmetic = dict(
        protocol='first seed, 302 pairs x five click groups, 3+2; not the color-training table',
        fewclick_dice=fc['primary']['dice'], sam_dice=sam['primary']['dice'], small_weight=weight,
        hypothetical_total_if_small_only_matches_sam=fc['primary']['dice'] + weight * (ssmall['dice'] - fsmall['dice']),
        required_small_dice_if_other_pairs_unchanged=fsmall['dice'] + (sam['primary']['dice'] - fc['primary']['dice']) / weight,
        caveat='Algebraic sensitivity only, not a measured model, ceiling, or predicted improvement.')
    result = dict(status='complete_readonly_research_aggregation', worker_pid=os.getpid(),
                  command=sys.argv, source_sha256=sha(Path(__file__)), inputs_sha256=bindings,
                  accepted_counts_recomputed=993, matched_groups=matched, arithmetic=arithmetic,
                  new_training=False, model_replaced=False, GPU_used=False,
                  limitations=['Existing development results reused; no new blind data or forward pass.',
                               'Matched anchors are class-area extremes, not representative or causal.',
                               'This recount checks saved count algebra; prior independent audits checked masks.'])
    (args.output / 'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], arithmetic=arithmetic, matched_small=matched['small']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
