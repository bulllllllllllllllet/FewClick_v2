"""Explicit first3-slide diagnostics; no inference from three samples to all slides."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    methods = ['HG_mix', 'sansa', 'M', 'no_cross_scale', 'proto_ms']; results = {}; per_slide = {}
    for method in methods:
        by = defaultdict(list); per_slide[method] = []
        for i in range(3):
            folder = a.root/'wsi'/method/f'case_{i:02}'/'readouts'; acc = json.loads((folder/'acceptance.json').read_text())
            assert sha(folder/'report.json') == acc['report_sha256']; report = json.loads((folder/'report.json').read_text())
            per_slide[method].append(report)
            for t, values in report['targets'].items():
                for kind in ['fused', 'owner']:
                    for domain in ['all', 'remote']:
                        if values[kind][domain]['dice'] is not None: by[kind, domain, t].append(values[kind][domain]['dice'])
                for domain in ['all', 'remote']:
                    value = values['window'][domain]['present_Dice']
                    if value is not None: by['window_present', domain, t].append(value)
        results[method] = {kind+'_'+domain:dict(macro_Dice=float(np.mean([np.mean(v) for (k, d, t), v in by.items() if (k, d) == (kind, domain)])),
                        per_tissue={t:float(np.mean(v)) for (k, d, t), v in by.items() if (k, d) == (kind, domain)})
                        for kind in ['fused', 'owner', 'window_present'] for domain in ['all', 'remote']}
    paired = {}
    for other in methods[1:]:
        paired[other] = {key:results['HG_mix'][key]['macro_Dice']-results[other][key]['macro_Dice'] for key in results['HG_mix']}
        paired[other]['fusion_difference_in_differences_all'] = paired[other]['fused_all']-paired[other]['owner_all']
        paired[other]['fusion_difference_in_differences_remote'] = paired[other]['fused_remote']-paired[other]['owner_remote']
    dump(a.root/'wsi_summary.json', dict(status='complete_accepted_sample_diagnostic', slides=3, results=results, paired_differences=paired,
        per_slide=per_slide, limitations='Fixed first3, descriptive mechanistic analysis only. Window-present Dice and unique-pixel whole-slide Dice have different estimands. No new significance or blind-test claim.'))
    dump(a.root/'wsi_terminal.json', dict(status='complete_accepted', report_sha256=sha(a.root/'wsi_summary.json')))


if __name__ == '__main__': main()
