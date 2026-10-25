"""Bind the transfer report to native gates, selected weights and provenance."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.evidence_closure.audit_external import close, require
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.metrics import score


def read(path): return json.loads(Path(path).read_text())


def run(root, report, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = read(report/'summary.json')
    require(summary['status'] == 'complete' and len(summary['records']) == 10, 'complete ten-output report required')
    tasks = root/'tasks_1025_132300'; contract = read(tasks/'contract.json')
    fields = {r['tile_id']: r for r in pd.read_parquet(Path(contract['catalog'])/'tiles.parquet').to_dict('records')}
    development = root.parent/'protocol_1025_124208/adaptation_1025_131451'
    weights = {}
    for method in ('sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell'):
        trained = development/f'train_{method}_1025_131451'
        result = read(trained/'summary.json'); best = read(trained/'best_checkpoint.json')
        require(result['status'] == 'complete' and result['step'] == 10000 and result['full_manifest_parity'], 'local training incomplete')
        actual = digest(trained/'training_consumed.jsonl')
        require(actual == result['training_consumed_sha256'] == 'fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', 'actual local training consumption')
        require(digest(best['path']) == best['checkpoint_sha256'], 'local selected weight')
        clean = root/f'clean_{method}_1025_162450'; meta = read(clean/'run.json')
        require(meta['checkpoint_sha256'] == best['checkpoint_sha256'] and
                Path(meta['checkpoint']).resolve() == Path(best['path']).resolve() and
                meta['local_training_actual_sha256'] == actual and meta['remote_training_steps'] == 0, 'clean transfer training binding')
        complete = read(clean/'summary.json')
        require(complete['GT_forbidden_input_API'] and (complete['query_maps_zero'] if method != 'sam_ft' else True), 'clean input isolation')
        for path, sha in meta['source_sha256'].items(): require(digest(path) == sha, 'frozen clean source changed')
        weights[method] = dict(checkpoint_sha256=best['checkpoint_sha256'], actual_40k_sha256=actual)
    context = root/'context_full_1025_152401'; meta = read(context/'run.json')
    gate = root/'gate_context_locked_1025_152208'; gm = read(gate/'run.json'); gr = read(gate/'summary.json')
    require(gr['status'] == 'complete' and gr['gate'] and gr['GT_forbidden_input_object'] and
            gr['native_context_forward_identical'] and gr['saved_count_replay'] and
            gr['native_SAM_errors'] == [0., 0.], 'native context gate')
    for key in ('source_sha256', 'protocol_sha256', 'sources_sha256', 'weight_manifest_sha256', 'generator', 'prior'):
        require(gm[key] == meta[key], 'context native/full identity: '+key)
    for path, sha in meta['source_sha256'].items(): require(digest(path) == sha, 'context source changed')
    weight = meta['prior']['weight']; require(digest(weight['output']) == weight['sha256'], 'MultiverSeg weight')
    require(digest(meta['generator']['checkpoint']) == meta['generator']['checkpoint_sha256'], 'SAM generator weight')
    replayed = 0
    for track in ('predicted_sparse', 'full_GT_reference'):
        folder = gate/track; result = read(folder/'summary.json')
        require(result['rows_sha256'] == digest(folder/'rows.jsonl'), 'native gate rows changed')
        for number, line in enumerate((folder/'rows.jsonl').open()):
            row = json.loads(line); field = fields[row['query_id']]
            gt = np.asarray(Image.open(field['labels']))
            probability = np.load(folder/f'prediction_{number:04d}.npy', allow_pickle=False)
            actual = score(probability >= .5, gt == row['target'], gt != 255, json.loads(field['input_mpp_xy_json']))
            for key, value in actual.items(): close(row[key], value, 'native context saved mask replay')
            replayed += 1
    require(replayed == 20, 'native context gate coverage')
    # Preserve raw runtime snapshots; correct their historical provenance only in a derived copy.
    clarification = root.parent/'mechanisms_1025_134753/XCell用户来源确认_1025_140237.json'
    require(clarification.exists(), 'user provenance source missing')
    corrected = copy.deepcopy(summary)
    for name, record in corrected['records'].items():
        if name.startswith('retained_m3'):
            record['training'] = 'Historical J5 plus prior mixed local/transfer adaptation; not a zero-R-training control'
        if 'xcell' in name or name.startswith('retained_m3'):
            record['accepted_XCell_provenance'] = 'User confirmed internal dataset training; current CRC/TIGER not used; upstream CTransPath separate'
            if isinstance(record.get('prior'), dict) and 'XCell_overlap' in record['prior']:
                record['prior']['XCell_overlap'] = record['accepted_XCell_provenance']
    corrected['raw_report_sha256'] = digest(report/'summary.json')
    corrected['XCell_provenance_source'] = dict(path=str(clarification), sha256=digest(clarification))
    write(output/'summary_user_provenance.json', corrected)
    text = (report/'report.md').read_text()
    text = text.replace('|historical J5|', '|historical J5 + mixed local/transfer adaptation|')
    text += '\nXCell来源采用用户确认：内部数据训练，当前CRC/TIGER未用于XCell训练；上游CTransPath另记。历史runtime元数据原样保留，来源更正后的汇总为summary_user_provenance.json。\n'
    text += '\n历史retained M3除J5先验外还有此前混合local/transfer增量训练；清洁G3模型只做本图适配。因此历史M3较高分不能归因于架构、XCell或与清洁行等远端训练预算。\n'
    (output/'report_user_provenance.md').write_text(text)
    write(output/'acceptance.json', dict(status='passed', raw_report_sha256=digest(report/'summary.json'),
        corrected_summary_sha256=digest(output/'summary_user_provenance.json'), selected_local_weights=weights,
        MultiverSeg_weight_sha256=weight['sha256'], context_gate_masks_replayed=replayed,
        clean_gate_masks_replayed=summary['saved_clean_gate_masks_replayed'],
        source_sha256=digest(__file__), scope='Predeclared G4 within-cohort transfer comparison accepted; no R-optimized training, new-class or WSI claim'))
    print(json.dumps(dict(status='passed', context_masks=replayed, clean_masks=summary['saved_clean_gate_masks_replayed'], output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'report', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.root, a.report, a.output)
