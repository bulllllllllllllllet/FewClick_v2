"""Compare all P2 representations after complete fixed-budget mask acceptance."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from .compare_restored import BOUNDS
from benchmarks.v5.encoder_research.accept_continuous_pilot import summarize
from benchmarks.v5.encoder_research.compare_continuous_pilots import (
    paired_interval, resource_guards, summary_values,
)
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha

MODES = ['continuous', 'fixed', 'ema']


def summarize_arm(raw):
    flat = [dict(r['plan'], scope=r['scope'], budget=r['budget'], **r['metrics']) for r in raw]
    groups = defaultdict(list)
    for row in flat:
        groups[row['scope'] + '|' + row['budget']].append(row)
    assert {k: len(v) for k, v in groups.items()} == {
        f'{scope}|{budget}': n for scope, n in [('local', 302), ('remote', 935)]
        for budget in ['1+1', '3+2', '5+1']}
    local = groups['local|3+2']
    result = dict(summary={k: summarize(v) for k, v in groups.items()}, local_strata=dict(
        small=summarize([r for r in local if r['positive'] / r['valid'] < .05]),
        fat=summarize([r for r in local if r['dataset'] == 'crc' and r['target'] == 10])))
    return flat, result, summary_values(result)


def contrast(reference_rows, candidate_rows, reference_values, candidate_values, inventory):
    guards = resource_guards(reference_values, candidate_values, BOUNDS)
    efficiency_bounds = dict(BOUNDS, primary_Dice_gain=-.01)
    # This only gates eligibility for measured efficiency, not efficiency acceptance.
    noninferiority = resource_guards(reference_values, candidate_values, efficiency_bounds)
    intervals = {
        'remote_macro_Dice': paired_interval(reference_rows, candidate_rows, inventory, 'remote', 'dice', True, True),
        'remote_BF1': paired_interval(reference_rows, candidate_rows, inventory, 'remote', 'bf1_um2', True),
        'absent_FPA': paired_interval(reference_rows, candidate_rows, inventory, 'remote', 'false_positive_fraction', False),
        'local_Dice': paired_interval(reference_rows, candidate_rows, inventory, 'local', 'dice', True),
    }
    assert abs(guards['candidate_minus_reference']['remote_macro_Dice'] -
               intervals['remote_macro_Dice']['candidate_minus_reference']) < 1e-12
    return dict(quality_guards=guards, efficiency_quality_guards=noninferiority,
                efficiency_accepted=False, paired_uncertainty=intervals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert json.loads((a.run / 'stage.json').read_text())['status'] == 'complete', 'All three arms must finish first'
    contract = json.loads((a.run / 'contract.json').read_text())
    trainer = 'benchmarks/v5/architecture_bridge/train_vocabulary.py'
    rows, summaries, values, reports, evidence = {}, {}, {}, {}, {}
    for mode in MODES:
        folder = a.run / mode
        report = json.loads((folder / 'report.json').read_text())
        accepted = json.loads((folder / 'independent_acceptance.json').read_text())
        assert accepted['status'] == 'accepted_P2_vocabulary_evaluation'
        assert accepted['arm'] == report['arm'] == mode
        assert accepted['rows'] == 3711 and not accepted['smoke'] and not report['smoke']
        assert accepted['report_sha256'] == sha(folder / 'report.json')
        assert accepted['checkpoint_sha256'] == report['checkpoint_sha256'] == sha(folder / 'checkpoint.pt')
        assert report['updates'] == 2000 and report['consumed_tasks'] == 8000 and report['epochs'] == 2
        assert report['source_sha256'] == contract['sources'][trainer]
        assert (folder / 'training_consumed.jsonl').read_bytes() == (a.run / 'continuous/training_consumed.jsonl').read_bytes()
        rows[mode], summaries[mode], values[mode] = summarize_arm(json.loads((folder / 'rows.json').read_text()))
        reports[mode] = report
        evidence[mode] = {name: sha(folder / name) for name in [
            'report.json', 'independent_acceptance.json', 'rows.json', 'masks.npz',
            'training_consumed.jsonl', 'checkpoint.pt']}
    for mode in ['fixed', 'ema']:
        for key in ['training_sha256', 'inputs', 'source_sha256', 'gate_sha256']:
            assert reports[mode][key] == reports['continuous'][key]
        identity = {k: v for k, v in reports[mode]['identity'].items() if k != 'mode'}
        assert identity == {k: v for k, v in reports['continuous']['identity'].items() if k != 'mode'}
        for left, right in zip(rows['continuous'], rows[mode]):
            for key in ['scope', 'budget', 'query_id', 'support_id', 'dataset', 'target', 'kind',
                        'positive_xy', 'negative_xy', 'positive', 'valid', 'true_boundary_pixels']:
                assert left[key] == right[key]
    _, inventory = load_cell_catalog(TASKS)
    contrasts = {}
    for candidate, reference in [('fixed', 'continuous'), ('ema', 'continuous'), ('ema', 'fixed')]:
        contrasts[f'{candidate}-{reference}'] = contrast(
            rows[reference], rows[candidate], values[reference], values[candidate], inventory)
    quality = [mode for mode in ['fixed', 'ema'] if contrasts[mode + '-continuous']['quality_guards']['pass_all']]
    efficiency = [mode for mode in ['fixed', 'ema'] if contrasts[mode + '-continuous']['efficiency_quality_guards']['pass_all']]
    result = dict(status='accepted_P2_single_seed_quality_comparison', worker_pid=os.getpid(),
                  values=values, summaries=summaries, contrasts=contrasts, bounds=BOUNDS,
                  quality_seed2_eligible=quality, measured_efficiency_check_eligible=efficiency,
                  efficiency_accepted=False, P2_complete=False, WSI_promoted=False,
                  evidence=evidence, contract_sha256=sha(a.run / 'contract.json'), source_sha256=sha(__file__),
                  limitations=['one seed in exposed development regions; replication still required',
                               'conditional query-space intervals retain support and patient dependence',
                               'TIGER3 absent-only; NA retained',
                               'quality eligibility does not complete code stability or efficiency audits'])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(values=values, quality_seed2_eligible=quality,
                          measured_efficiency_check_eligible=efficiency)), flush=True)


if __name__ == '__main__':
    main()
