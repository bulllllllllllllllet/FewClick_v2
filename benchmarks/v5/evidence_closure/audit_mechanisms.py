"""Accept completed mechanism runs and interpret predefined paired contrasts."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from .audit_external import require, close
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.metrics import VERSION, boundary
from benchmarks.v5.fair_comparison.protocol import nested


def read(path): return json.loads(Path(path).read_text())


CONTRASTS = [
    ('region_XCell_minus_none', 'm3_none', 'm3_xcell', 'Cell branch adds capacity and internal feature prior'),
    ('region_XCell_minus_morphology', 'm3_morphology', 'm3_xcell', 'Same capacity; appearance64 replaces zeros; not isolated mIF objective'),
    ('dense_XCell_minus_none', 'fewclick_m3d_image', 'fewclick_m3d_xcell', 'Cell branch adds capacity and internal feature prior'),
    ('dense_XCell_minus_morphology', 'm3d_morphology', 'fewclick_m3d_xcell', 'Same capacity; geometry/count and detection prior retained'),
    ('region_morphology_minus_none', 'm3_none', 'm3_morphology', 'Geometry/count plus cell branch parameters'),
    ('dense_morphology_minus_none', 'fewclick_m3d_image', 'm3d_morphology', 'Geometry/count plus cell branch parameters'),
    ('M3_minus_M1', 'm1_xcell', 'm3_xcell', 'Views and encoder capacity change together'),
    ('independent_minus_shared', 'm3_shared_encoder', 'm3_xcell', 'Same three views; encoder capacity differs'),
    ('hierarchy_minus_no_hierarchy', 'm3_no_hierarchy', 'm3_xcell', 'Cross-scale decoder graph edges retained in both'),
    ('shared_M3_minus_M1', 'm1_xcell', 'm3_shared_encoder', 'Three views shared weights; parameter counts still slightly differ'),
    ('dense_minus_region_XCell', 'm3_xcell', 'fewclick_m3d_xcell', 'Adds Dense path and jointly trained fine semantic features'),
    ('dense_minus_region_none', 'm3_none', 'fewclick_m3d_image', 'Adds Dense path and jointly trained fine semantic features'),
    ('dense_minus_region_morphology', 'm3_morphology', 'm3d_morphology', 'Adds Dense path and jointly trained fine semantic features'),
]


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    comparison = root/'mechanism_formal_1025_141300/comparison_1025_141300/summary.json'
    original = read(comparison); require(original['status'] == 'complete', 'comparison incomplete')
    local = root.parent/'protocol_1025_124208'; tasks = local/'tasks_1025_125400'
    contract = read(tasks/'contract.json'); protocol_sha = digest(tasks/'contract.json')
    require(original['protocol_sha256'] == protocol_sha and
            digest(tasks/'validation_maximum.json') == contract['validation_sha256'], 'protocol identity')
    require(digest(Path(contract['catalog'])/'tiles.parquet') == contract['catalog_sha256'], 'catalog identity')
    fields = {r['tile_id']: r for r in pd.read_parquet(Path(contract['catalog'])/'tiles.parquet').to_dict('records')}
    plans = read(tasks/'validation_maximum.json')
    key = lambda r: (r['pair_id'], r['prompt_group'], r['click_budget'])
    expected = {key(p): p for maximum in plans for p in [nested(maximum, b) for b in contract['report_budgets']]}
    require(len(expected) == 6040, 'fixed local coverage')
    labels = {}; denominators = {}
    for plan in plans:
        tile = plan['query_id']; row = fields[tile]
        if tile not in labels:
            gt = np.asarray(Image.open(row['labels'])); labels[tile] = gt
            require(hashlib.sha256(gt.tobytes()).hexdigest() == row['labels_array_sha256'], 'GT array SHA')
        gt = labels[tile]
        for sign in ('positive', 'negative'):
            xy = np.asarray(plan[sign+'_xy']); require(np.isfinite(xy).all() and ((xy > 0) & (xy < 1)).all(), 'point geometry')
            pixel = np.floor(xy*np.array(gt.shape[::-1])).astype(int); v = gt[pixel[:, 1], pixel[:, 0]]
            require((v == plan['target']).all() if sign == 'positive' else ((v != plan['target']) & (v != 255)).all(), 'point sign')
        if (tile, plan['target']) not in denominators:
            valid = gt != 255; truth = gt == plan['target']
            denominators[tile, plan['target']] = dict(valid=int(valid.sum()), positive=int((truth & valid).sum()),
                true_boundary_pixels=int(boundary(truth, valid).sum()))
    def validate(row):
        identity = key(row); require(identity in expected, 'unplanned local row')
        for field, value in expected[identity].items(): require(row[field] == value, 'plan mismatch: '+field)
        for field, value in denominators[row['query_id'], row['target']].items(): require(row[field] == value, 'actual GT denominator')
        tp, fp, fn = [row[k] for k in ('tp', 'fp', 'fn')]
        require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)) and tp+fn == row['positive'] and fp <= row['valid']-row['positive'], 'impossible counts')
        close(row['dice'], 2*tp/(2*tp+fp+fn), 'Dice/count replay')
        close(row['false_positive_fraction'], fp/row['valid'], 'FPA/count replay')
        require(row['boundary_evaluable'] == (row['true_boundary_pixels'] > 0), 'boundary eligibility')
        if row['true_boundary_pixels'] == 0: require(row['bf1_um2'] is None, 'undefined BF1')
        else: require(row['bf1_um2'] is not None and 0 <= row['bf1_um2'] <= 1, 'BF1 range')
        if row['true_boundary_pixels'] > 0 and row['predicted_boundary_pixels'] == 0: close(row['bf1_um2'], 0, 'missing predicted boundary')
        return identity
    tables = {}; evidence = {}
    for name, record in original['records'].items():
        evaluated = Path(record['run']); meta = read(evaluated/'run.json'); result = read(evaluated/'summary.json')
        require(result['status'] == 'complete' and not result['gate'] and meta['method'] == name and
                meta['protocol_sha256'] == protocol_sha and result['metric_version'] == VERSION, 'evaluation contract')
        require(digest(evaluated/'rows.jsonl') == result['rows_sha256'] == record['rows_sha256'], 'score rows SHA')
        rows = [json.loads(line) for line in (evaluated/'rows.jsonl').open()]
        seen = {validate(row) for row in rows}; require(len(seen) == len(rows) == 6040 and seen == set(expected), 'full score coverage')
        for row in rows:
            for field in ('space_key', 'source_id', 'spatial_group_id'):
                require(row[field] == fields[row['query_id']][field], 'score source provenance')
        trained = Path(meta['checkpoint']).parent; tm = read(trained/'run.json')['contract']; tr = read(trained/'summary.json'); best = read(trained/'best_checkpoint.json')
        require(tr['status'] == 'complete' and not tr['gate'] and tr['step'] == tm['steps'] == 10000 and
                tm['method'] == name and tm['protocol_sha256'] == protocol_sha and tm['seed'] == 20261025 and
                tm['accumulate'] == 4 and tm['warmup'] == 1000 and tr['full_manifest_parity'] and
                tr['checkpoint_reload_exact'] and tr['nonzero_joint_backbone_gradient'], 'full training budget/contract')
        require(not tm['initialization']['J5_loaded'] and not tm['initialization']['target_cohort_initialization'], 'clean initialization')
        consumed = digest(trained/'training_consumed.jsonl')
        require(consumed == tr['training_consumed_sha256'] == tm['training_plans_sha256'] ==
                'fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', '40k consumed tasks')
        require(sum(1 for _ in (trained/'training_consumed.jsonl').open()) == 40000, '40k count')
        require(Path(best['path']).resolve() == Path(meta['checkpoint']).resolve() and
                digest(best['path']) == best['checkpoint_sha256'] == meta['checkpoint_sha256'], 'selected checkpoint')
        history = tr['history']; require([h['step'] for h in history] == list(range(1000, 10001, 1000)), 'validation schedule')
        expected_selection = {k for k in expected if k[1] == 0 and k[2] == '3+2'}
        for h in history:
            validation = read(trained/f'validation_{h["step"]:05d}_1025/rows.json')
            require(len(validation) == len(expected_selection) and {validate(r) for r in validation} == expected_selection, 'actual selection tasks')
            close(h['score'], float(np.mean([r['dice'] for r in validation])), 'selection score recount')
        chosen = max(history, key=lambda h: h['score'])
        require(best['step'] == chosen['step'], 'best checkpoint selection rule'); close(best['score'], chosen['score'], 'best selection score')
        selected_rows = [r for r in rows if r['prompt_group'] == 0 and r['click_budget'] == '3+2']
        close(float(np.mean([r['dice'] for r in selected_rows])), best['score'], 'full scorer selected group replay')
        gate_evidence = None
        if name.startswith('m'):
            gate = root/'mechanism_formal_1025_141300'/f'gate_{name}_1025_141300'; gr = read(gate/'summary.json')
            require(gr['status'] == 'complete' and gr['gate'] and gr['fresh_gate_resume'] and
                    gr['checkpoint_reload_exact'] and gr['nonzero_joint_backbone_gradient'], 'native mechanism gate')
            gate_evidence = digest(gate/'summary.json')
        for stage in (trained, evaluated):
            require(stage.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'job terminal evidence')
        tables[name] = rows
        evidence[name] = dict(checkpoint_sha256=meta['checkpoint_sha256'], rows_sha256=result['rows_sha256'],
            consumed_sha256=consumed, selected_step=best['step'], selection_rows_recounted=3020,
            native_gate_summary_sha256=gate_evidence)
        print(json.dumps(dict(stage='audited', method=name, rows=len(rows))), flush=True)
        write(output/'status.json', dict(status='running', worker_pid=os.getpid(), methods_audited=len(evidence)))
    clusters = sorted({r['space_key'] for r in next(iter(tables.values()))}); ix = {s: i for i, s in enumerate(clusters)}
    rng = np.random.default_rng(20261025); sample = rng.integers(len(clusters), size=(2000, len(clusters)))
    values = {}; bootstrap = {}
    for method, rows in tables.items():
        values[method] = {}; bootstrap[method] = {}
        for budget in contract['report_budgets']:
            pairs = defaultdict(list)
            for row in rows:
                if row['click_budget'] == budget: pairs[row['pair_id']].append(row)
            sums = np.zeros(len(clusters)); counts = np.zeros(len(clusters)); pair_means = []
            for group in pairs.values():
                require(len(group) == 5, 'five groups per pair')
                mean = float(np.mean([r['dice'] for r in group])); pair_means.append(mean)
                j = ix[group[0]['space_key']]; sums[j] += mean; counts[j] += 1
            v = float(np.mean(pair_means)); close(v, original['metrics'][method][budget]['macro_pair_dice'], 'aggregate report replay')
            values[method][budget] = v; bootstrap[method][budget] = sums[sample].sum(1)/counts[sample].sum(1)
    contrasts = []
    for label, a, b, limitation in CONTRASTS:
        for budget in contract['report_budgets']:
            contrast = dict(label=label, a=a, b=b, budget=budget, b_minus_a=values[b][budget]-values[a][budget],
                paired_source_cluster_ci95=np.quantile(bootstrap[b][budget]-bootstrap[a][budget], [.025, .975]).tolist(), interpretation_limit=limitation)
            contrasts.append(contrast)
    corrected = copy.deepcopy(original)
    for name, record in corrected['records'].items():
        prior = record['prior']
        if prior.get('cell_mode') == 'xcell':
            prior['XCell_overlap'] = 'User-confirmed internal training; current CRC/TIGER not used; upstream CTransPath separate'
    corrected.update(raw_summary_sha256=digest(comparison), accepted_contrasts=contrasts)
    write(output/'summary_user_provenance.json', corrected)
    capacities = read(root/'parameter_capacity_1025_140730.json')
    write(output/'acceptance.json', dict(status='passed', methods=evidence, actual_GT_fields=len(labels),
        actual_GT_target_pairs=len(denominators), validation_rows_recounted_per_method=3020,
        protocol_sha256=protocol_sha, raw_summary_sha256=digest(comparison), source_sha256=digest(__file__),
        capacity_audit_sha256=digest(root/'parameter_capacity_1025_140730.json'),
        paired_contrasts=contrasts, bootstrap_repetitions=2000, source_clusters=len(clusters),
        scope='Seven planned mechanism runs plus two clean Dense controls accepted; generic H&E feature/pretraining-objective control remains pending',
        limitations=['Full predicted masks were not saved: actual GT/count algebra and selection are replayed, not all mask TP/FP assignments',
            'Single training seed; source-space bootstrap is development uncertainty, not independent-patient or training-seed inference',
            'No multiple-comparison correction; contrasts are descriptive evidence, not significance certification']))
    text = ['# 七机制完整验收与受控配对解释', '', '九组各10k更新/40k相同实际任务，302对×5点组×4预算6040行。选模十次302对group0重算通过。', '',
        '| 方法 | 主3+2 Dice |', '|---|---:|']
    for method, metric in values.items(): text.append(f'|{method}|{metric["3+2"]:.4f}|')
    text += ['', '| 对比（后者减前者） | Dice差 | 来源聚类95%区间 | 解释限制 |', '|---|---:|---|---|']
    for c in contrasts:
        if c['budget'] == '3+2':
            lo, hi = c['paired_source_cluster_ci95']; text.append(f'|{c["label"]}|{c["b_minus_a"]:.6f}|[{lo:.6f}, {hi:.6f}]|{c["interpretation_limit"]}|')
    text += ['', '形态控制保留10维位置/面积/unknown类型和检测计数，XCell64置零；与完整XCell等容量。无细胞分支容量略不同。',
        'M3去层级仍有跨尺度图边；共享编码器仍有三幅真实物理视野。M3与M1同时改变视野和编码器容量。Dense还引入fine语义latent联合训练。',
        '当前Dense收益较明确；细胞、独立多编码器、层级的收益需按配对区间和单seed限制解释，不以模块必须有效为验收条件。',
        '普通H&E细胞特征与同网络预训练目标控制未完成；不能把XCell差异归因于mIF监督本身。',
        '75来源空间、单seed开发选模；不作病例总体显著性或多重比较校正后的结论。全量mask未保存，验收不声称重新推理全部预测。',
        'XCell按用户确认的内部数据训练记账，CRC/TIGER未参与该训练；上游公共CTransPath另记。原始快照保留。']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    write(output/'status.json', dict(status='complete', worker_pid=os.getpid(), methods=len(evidence)))
    print(json.dumps(dict(status='passed', methods=len(evidence), output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a.root, a.output)
    except Exception:
        if a.output.is_dir(): write(a.output/'failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
