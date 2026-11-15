"""Read-only subset replay and paired original15-WSI statistics on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from .aggregate import summarize
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.fair_comparison.metrics import score


def paired_wsi(root, out):
    slides = []
    for index in range(15):
        reports = {m: json.loads((root/'formal'/f'case_{index:02}'/m/'report.json').read_text()) for m in ['HG_mix', 'sansa']}
        assert reports['HG_mix']['slide'] == reports['sansa']['slide']
        assert reports['HG_mix']['targets'].keys() == reports['sansa']['targets'].keys()
        delta = {}
        for target in reports['HG_mix']['targets']:
            delta[target] = {}
            for domain in ['whole_WSI', 'remote_only_WSI']:
                x, y = (reports[m]['targets'][target][domain] for m in ['HG_mix', 'sansa'])
                assert x['valid'] == y['valid'] and x['positive'] == y['positive']
                delta[target][domain] = x['dice']-y['dice'] if x['dice'] is not None else None
        slides.append(dict(slide=reports['HG_mix']['slide'], patient_id=reports['HG_mix']['patient_id'], differences=delta))
    targets = sorted({k for s in slides for k in s['differences']}); assert len(targets) == 11
    patient_ids = sorted({s['patient_id'] for s in slides}); by_patient = [[i for i, s in enumerate(slides) if s['patient_id'] == p] for p in patient_ids]
    def macro(indices, domain):
        values = []
        for target in targets:
            got = [slides[i]['differences'].get(target, {}).get(domain) for i in indices]
            got = [v for v in got if v is not None]
            if not got: return None
            values.append(np.mean(got))
        return float(np.mean(values))
    rng = np.random.default_rng(20261116); replicates = {k: [] for k in ['whole_WSI', 'remote_only_WSI']}
    for _ in range(10000):
        sampled = rng.integers(len(by_patient), size=len(by_patient)); indices = [i for p in sampled for i in by_patient[p]]
        for domain in replicates:
            value = macro(indices, domain)
            if value is not None: replicates[domain].append(value)
    dump(out/'paired_original15.json', dict(status='complete_existing_predictions_only', slides=slides,
         cluster='recorded patient_id; if metadata represents slide IDs this is slide-clustered only, not independently verified patient linkage',
         clusters=len(by_patient), requested_replicates=10000, statistical_seed=20261116,
         summaries={k:dict(difference=macro(list(range(15)), k), CI95=np.percentile(v, [2.5, 97.5]).tolist(), retained=len(v)) for k, v in replicates.items()},
         limitations='Conditional on all11 tissues represented and fixed trained weights; previously exposed cohort; no training-seed inference.'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text()); sample = Path(c['sample']); tasks = json.loads((sample/'tasks.json').read_text())
    sampling = json.loads((sample/'sampling.json').read_text()); selected = {t['index']:t for t in tasks}
    out = a.root/'references'; out.mkdir(exist_ok=False)
    data = DecoderData(Path(c['cache']), 'match'); data.rows.update(json.loads((sample/'query_rows.json').read_text()))
    for method, source in c['reference_evaluations'].items():
        source = Path(source); rows = {}; identities = []
        for shard in [0, 1]:
            folder = source/f'{method}_{shard}'; acc = json.loads((folder/'acceptance.json').read_text())
            assert acc['status'] == 'complete_accepted'
            assert sha(folder/'rows.json') == acc['rows_sha256'] and sha(folder/'masks.npy') == acc['masks_sha256']
            stored = np.load(folder/'masks.npy', mmap_mode='r'); original_rows = json.loads((folder/'rows.json').read_text())
            assert len(stored) == len(original_rows)
            for row, packed in zip(original_rows, stored):
                if row['index'] not in selected: continue
                t = selected[row['index']]; gt, mpp = data.labels(t['query_id'])
                mask = np.unpackbits(packed, count=512*512).reshape(512, 512).astype(bool)
                actual = score(mask, gt == t['target'], gt != 255, mpp); assert actual == row['metrics']
                assert row['index'] not in rows; rows[row['index']] = actual
            identities.append(dict(acceptance=str(folder/'acceptance.json'), sha256=sha(folder/'acceptance.json')))
        dump(out/(method+'.json'), dict(status='complete_accepted', method=method, sources=identities,
            sampled_mask_replay_exact=len(rows), tasks_sha256=c['sample_tasks_sha256'], rows=rows, **summarize(rows, tasks, sampling)))
        print(method, 'accepted', len(rows), flush=True)
    paired_wsi(Path(c['original_wsi_root']), out)
    dump(out/'acceptance.json', dict(status='complete_accepted', tasks=len(tasks), files={str(p.name):sha(p) for p in out.glob('*.json')}))


if __name__ == '__main__': main()
