"""Accept every repeated seed and report all runs, never the best seed."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.evidence_closure.audit_external import require, close
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.metrics import VERSION, boundary
from benchmarks.v5.fair_comparison.protocol import nested

METHODS = ('sam_ft', 'wsi_sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell')
SEEDS = (20261025, 20261026, 20261027)


def read(path): return json.loads(Path(path).read_text())


def run(root, output, gate=False, wait=False):
    output.mkdir(parents=True, exist_ok=False)
    formal = root/'formal_1025_154653'
    if wait and not gate:
        while True:
            try: state = read(formal/'status.json')
            except json.JSONDecodeError:
                time.sleep(.5); continue
            # Coordinator summary is written just before the shell appends EXIT_CODE.
            # Wait for both terminal signals rather than failing in that small window.
            if state['status'] == 'complete' and formal.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'): break
            require(state['status'] != 'failed', 'upstream repeated-training queue failed')
            pid = state['coordinator_pid']
            require(Path(f'/proc/{pid}').exists(), 'upstream coordinator no longer alive; inspect terminal evidence')
            write(output/'status.json', dict(status='waiting_live_coordinator', worker_pid=os.getpid(),
                coordinator_pid=pid, complete_stages=len(state.get('complete', {})), observed_at=time.time()))
            time.sleep(5)
    if not gate:
        completed = read(formal/'summary.json')
        require(completed['status'] == 'complete' and len(completed['complete']) == 24 and
                formal.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'all extra runs must finish first')
    base = root.parent/'protocol_1025_124208'; base_tasks = base/'tasks_1025_125400'; manifest = read(root/'manifest.json')
    base_contract = read(base_tasks/'contract.json'); base_actual = base/'realized_train_1025_125928/training_plans.jsonl'
    original_lines = base_actual.read_text().splitlines(keepends=True)
    require(len(original_lines) == 40000, 'original40k task count')
    pool_sha = hashlib.sha256(''.join(sorted(original_lines)).encode()).hexdigest()
    sources = {SEEDS[0]: dict(tasks=str(base_tasks), realized=str(base_actual.parent),
        protocol_sha256=digest(base_tasks/'contract.json'), coordinate_sha256=digest(base_actual))}
    for seed in SEEDS[1:]:
        info = manifest[str(seed)]; tasks = Path(info['tasks']); actual = Path(info['realized'])/'training_plans.jsonl'
        require(digest(tasks/'contract.json') == info['protocol_sha256'] and digest(actual) == info['coordinate_sha256'], 'seed protocol/order SHA')
        tc = read(tasks/'contract.json'); require(tc['replication_seed'] == seed and tc['base_protocol_sha256'] == sources[SEEDS[0]]['protocol_sha256'], 'seed provenance')
        lines = actual.read_text().splitlines(keepends=True); permutation = read(tasks.parent/'permutation.json')
        require(sorted(permutation) == list(range(40000)) and lines == [original_lines[i] for i in permutation], 'actual-task bijection')
        require(hashlib.sha256(''.join(sorted(lines)).encode()).hexdigest() == pool_sha == tc['training_coordinate_multiset_sha256'], 'same task/click multiset')
        require(digest(tasks/'validation_maximum.json') == base_contract['validation_sha256'], 'same validation bytes')
        sources[seed] = info
    require(len({v['coordinate_sha256'] for v in sources.values()}) == 3, 'seed training orders identical')
    fields = {r['tile_id']: r for r in pd.read_parquet(Path(base_contract['catalog'])/'tiles.parquet').to_dict('records')}
    require(digest(Path(base_contract['catalog'])/'tiles.parquet') == base_contract['catalog_sha256'], 'catalog SHA')
    plans = read(base_tasks/'validation_maximum.json'); key = lambda r: (r['pair_id'], r['prompt_group'], r['click_budget'])
    expected = {key(p): p for maximum in plans for p in [nested(maximum, b) for b in base_contract['report_budgets']]}
    labels = {}; denominators = {}
    for plan in plans:
        tile = plan['query_id']; field = fields[tile]
        if tile not in labels:
            gt = np.asarray(Image.open(field['labels'])); labels[tile] = gt
            require(hashlib.sha256(gt.tobytes()).hexdigest() == field['labels_array_sha256'], 'actual GT SHA')
        gt = labels[tile]; valid = gt != 255; truth = gt == plan['target']
        if (tile, plan['target']) not in denominators:
            denominators[tile, plan['target']] = dict(valid=int(valid.sum()), positive=int((truth & valid).sum()),
                true_boundary_pixels=int(boundary(truth, valid).sum()))
    def validate(row):
        k = key(row); require(k in expected, 'unplanned task')
        for field, value in expected[k].items(): require(row[field] == value, 'task/click mismatch: '+field)
        for field, value in denominators[row['query_id'], row['target']].items(): require(row[field] == value, 'actual GT denominator')
        tp, fp, fn = [row[f] for f in ('tp', 'fp', 'fn')]
        require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)) and tp+fn == row['positive'] and fp <= row['valid']-row['positive'], 'impossible counts')
        close(row['dice'], 2*tp/(2*tp+fp+fn), 'Dice/count replay')
        require(row['boundary_evaluable'] == (row['true_boundary_pixels'] > 0), 'boundary eligibility')
        if row['true_boundary_pixels'] == 0: require(row['bf1_um2'] is None, 'undefined BF1')
        else: require(row['bf1_um2'] is not None and 0 <= row['bf1_um2'] <= 1, 'BF1 range')
        if row['true_boundary_pixels'] > 0 and row['predicted_boundary_pixels'] == 0: close(row['bf1_um2'], 0., 'empty boundary')
        return k
    metrics = {}; evidence = {}; seeds = SEEDS[:1] if gate else SEEDS
    for seed in seeds:
        info = sources[seed]; metrics[str(seed)] = {}; evidence[str(seed)] = {}
        for method in METHODS:
            if seed == SEEDS[0]:
                trained = base/f'adaptation_1025_131451/train_{method}_1025_131451'
                evaluated = base/f'adaptation_1025_131451/evaluate_{method}_1025_131451'
            else:
                trained = formal/f'train_{method}_seed{seed}_1025_154653'
                evaluated = formal/f'evaluate_{method}_seed{seed}_1025_154653'
                native = formal/f'gate_{method}_seed{seed}_1025_154653'
                gr = read(native/'summary.json')
                require(gr['status'] == 'complete' and gr['gate'] and gr['fresh_gate_resume'] and gr['checkpoint_reload_exact'] and gr['nonzero_joint_backbone_gradient'], 'native extra-seed gate')
                require(digest(native/'training_consumed.jsonl') == gr['training_consumed_sha256'] == info['gate_coordinate_sha256'], 'native actual gate consumption')
            tr = read(trained/'summary.json'); tm = read(trained/'run.json')['contract']; best = read(trained/'best_checkpoint.json')
            em = read(evaluated/'run.json'); er = read(evaluated/'summary.json')
            require(tr['status'] == er['status'] == 'complete' and not tr['gate'] and not er['gate'], 'incomplete full run')
            require(tm['seed'] == seed and tm['method'] == method and tm['steps'] == tr['step'] == 10000 and
                    tm['accumulate'] == 4 and tm['warmup'] == 1000 and tm['protocol_sha256'] == em['protocol_sha256'] == info['protocol_sha256'], 'seed training contract')
            require(tr['full_manifest_parity'] and tr['checkpoint_reload_exact'] and tr['nonzero_joint_backbone_gradient'], 'full training checks')
            require(digest(trained/'training_consumed.jsonl') == tr['training_consumed_sha256'] == tm['training_plans_sha256'] == info['coordinate_sha256'], 'actual matched seed task order')
            require(Path(best['path']).resolve() == Path(em['checkpoint']).resolve() and
                    digest(best['path']) == best['checkpoint_sha256'] == em['checkpoint_sha256'], 'selected checkpoint identity')
            require(digest(evaluated/'rows.jsonl') == er['rows_sha256'] and er['metric_version'] == VERSION, 'evaluation metric/rows SHA')
            for path, sha in tm['source_sha256'].items(): require(digest(path) == sha, 'frozen source changed')
            require(not tm['initialization']['J5_loaded'] and not tm['initialization']['target_cohort_initialization'], 'target-cohort initialization') if method.startswith('fewclick') else None
            history = tr['history']; require([h['step'] for h in history] == list(range(1000, 10001, 1000)), 'selection schedule')
            selection_keys = {k for k in expected if k[1] == 0 and k[2] == '3+2'}
            for h in history:
                validation = read(trained/f'validation_{h["step"]:05d}_1025/rows.json')
                require(len(validation) == 302 and {validate(r) for r in validation} == selection_keys, 'selection task coverage')
                close(h['score'], float(np.mean([r['dice'] for r in validation])), 'selection score replay')
            chosen = max(history, key=lambda h: h['score']); require(best['step'] == chosen['step'], 'best development checkpoint')
            close(best['score'], chosen['score'], 'best development score')
            rows = [json.loads(line) for line in (evaluated/'rows.jsonl').open()]
            require(len(rows) == 6040 and {validate(r) for r in rows} == set(expected), 'complete6040 tasks')
            for row in rows:
                for field in ('space_key', 'source_id', 'spatial_group_id'): require(row[field] == fields[row['query_id']][field], 'query provenance')
            selected = [r['dice'] for r in rows if r['prompt_group'] == 0 and r['click_budget'] == '3+2']
            close(float(np.mean(selected)), best['score'], 'selected-checkpoint full score replay')
            for stage in (trained, evaluated): require(stage.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'terminal job evidence')
            by_budget = {}
            for budget in base_contract['report_budgets']:
                pair = defaultdict(list); bf = []
                for row in rows:
                    if row['click_budget'] == budget:
                        pair[row['pair_id']].append(row['dice'])
                        if row['bf1_um2'] is not None: bf.append(row['bf1_um2'])
                require(len(pair) == 302 and all(len(v) == 5 for v in pair.values()), 'pair groups')
                dice = float(np.mean([np.mean(v) for v in pair.values()])); close(dice, er['metrics'][budget]['macro_pair_dice'], 'saved aggregate replay')
                by_budget[budget] = dict(dice=dice, bf1_um2=float(np.mean(bf)), bf1_denominator=len(bf))
            metrics[str(seed)][method] = by_budget
            evidence[str(seed)][method] = dict(checkpoint_sha256=em['checkpoint_sha256'], selected_step=best['step'],
                consumed_sha256=info['coordinate_sha256'], rows_sha256=er['rows_sha256'])
            write(output/'status.json', dict(status='running', worker_pid=os.getpid(), seed=seed, method=method))
            print(json.dumps(dict(stage='audited', seed=seed, method=method)), flush=True)
    summary = {}; differences = []
    if not gate:
        for method in METHODS:
            summary[method] = {}
            for budget in base_contract['report_budgets']:
                summary[method][budget] = {}
                for metric in ('dice', 'bf1_um2'):
                    values = [metrics[str(s)][method][budget][metric] for s in SEEDS]
                    summary[method][budget][metric] = dict(all_seeds=values, mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), n=3)
        for a, b in [('fewclick_m3d_image', 'fewclick_m3d_xcell'), ('sam_ft', 'fewclick_m3d_xcell'), ('wsi_sam_ft', 'fewclick_m3d_xcell')]:
            for budget in base_contract['report_budgets']:
                values = [metrics[str(s)][b][budget]['dice']-metrics[str(s)][a][budget]['dice'] for s in SEEDS]
                differences.append(dict(a=a, b=b, budget=budget, all_seed_differences=values, mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1))))
    result = dict(status='complete_gate' if gate else 'complete', gate=gate, metrics=metrics, all_seed_summary=summary,
        paired_seed_differences=differences, evidence=evidence, source_sha256=digest(__file__),
        actual_task_multiset_sha256=pool_sha, seed_protocols=sources, seeds=list(seeds),
        limits=['Three seeds only; sample standard deviation ddof1, not population significance',
            'All development-selected on identical spatial data; no blind-test or independent-patient claim',
            'Task/click multiset fixed; order/new heads/train RNG vary; pretrained weights shared and Dense internal initialization fixed',
            'Public/internal priors and input views/capacity differ; same incremental supervision is not equal total training',
            'All full-row GT denominators/count algebra replayed; full prediction masks not regenerated'])
    write(output/'summary.json', result)
    text = ['# 三seed本图完整对比', '', '固定302对×5点组；主3正2负；各seed按固定开发组0独立选模。所有seed均保留，不挑最好seed。', '',
        '| 方法 | seed25 | seed26 | seed27 | Dice均值 ± 样本标准差 |', '|---|---:|---:|---:|---:|']
    if not gate:
        for method in METHODS:
            m = summary[method]['3+2']['dice']; v = m['all_seeds']
            text.append(f'|{method}|{v[0]:.4f}|{v[1]:.4f}|{v[2]:.4f}|{m["mean"]:.4f} ± {m["sample_std"]:.4f}|')
    text += ['', '全部预算、BF1和同seed配对差见summary.json。样本标准差使用ddof1，不是病例置信区间。',
        'Dense内部初始化仍固定；seed重复涵盖顺序、新头及训练随机性，不声称所有参数独立初始化。',
        '当前CRC/TIGER未参与XCell内部预训练，上游CTransPath另记；SAM/WSI-SAM/FewClick预训练和视野仍不同。',
        '门控只验收原始四种完整路径及seed排列合同，不能宣称额外seed完整训练评分已验收。' if gate else '四主方法三seed完整验收不代表七个机制控制均有三seed。']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    write(output/'status.json', dict(status=result['status'], worker_pid=os.getpid()))
    print(json.dumps(dict(status=result['status'], seeds=list(seeds), output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', action='store_true'); p.add_argument('--wait-for-coordinator', action='store_true'); a = p.parse_args()
    try: run(a.root, a.output, a.gate, a.wait_for_coordinator)
    except Exception:
        if a.output.is_dir(): write(a.output/'failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
