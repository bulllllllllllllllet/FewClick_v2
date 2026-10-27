"""CRC-only class report from accepted all-label runs, without model reselection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import yaml


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison')
    accepted_path = root/'replication_1025_153837/complete_report_1025_230100/summary.json'
    accepted = read(accepted_path)
    require(accepted['status'] == 'complete', 'accepted repeated runs')
    palette = Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml')
    classes = yaml.safe_load(palette.read_text())['data']['class_map']
    require({c['id'] for c in classes} == set(range(12)), '12 original labels')
    budgets = ('3+2', '1+1', '3+1', '5+1')
    methods = ('sam_ft', 'wsi_sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell')
    metrics = defaultdict(lambda: defaultdict(list)); coverage = []; inputs = []
    baseline_keys = None
    for seed in accepted['seeds']:
        for method in methods:
            if seed == 20261025:
                parent = root/'protocol_1025_124208/adaptation_1025_131451'
                suffix = f'{method}_1025_131451'
            else:
                parent = root/'replication_1025_153837/formal_1025_154653'
                suffix = f'{method}_seed{seed}_1025_154653'
            rows_path = parent/f'evaluate_{suffix}/rows.jsonl'
            train_path = parent/f'train_{suffix}/training_consumed.jsonl'
            evidence = accepted['evidence'][str(seed)][method]
            require(sha(rows_path) == evidence['rows_sha256'], 'accepted score SHA')
            require(sha(train_path) == evidence['consumed_sha256'], 'accepted actual training SHA')
            train = Counter()
            for line in train_path.open():
                row = json.loads(line)
                if row['dataset'] == 'crc':
                    train[row['target']] += 1
            require(set(train) == set(range(12)) - {2}, 'all11 tissue targets trained')
            pairs = defaultdict(list); keys = set(); provenance = defaultdict(set)
            total_rows = 0
            for line in rows_path.open():
                r = json.loads(line); total_rows += 1
                denominator = 2*r['tp']+r['fp']+r['fn']
                dice = 2*r['tp']/denominator if denominator else 1.
                require(abs(dice-r['dice']) < 1e-12 and r['positive'] == r['tp']+r['fn'], 'Dice and GT denominator')
                if r['dataset'] != 'crc':
                    continue
                key = (r['pair_id'], r['prompt_group'], r['click_budget'])
                require(key not in keys and r['target'] in train, 'unique trained CRC target')
                keys.add(key)
                pairs[r['target'], r['click_budget'], r['pair_id']].append(r['dice'])
                provenance[r['target']].add(r['space_key'])
            require(total_rows == 6040 and len(keys) == 4400, 'complete rows and CRC subset')
            if baseline_keys is None:
                baseline_keys = keys
            require(keys == baseline_keys, 'same task identities across methods/seeds')
            for target in sorted(train):
                coverage.append(dict(seed=seed, method=method, target=target, training_tasks=train[target], evaluation_pairs=20, source_spaces=len(provenance[target])))
                for budget in budgets:
                    values = [math.fsum(v)/len(v) for (t,b,_),v in pairs.items() if t == target and b == budget]
                    require(len(values) == 20 and all(len(v) == 5 for (t,b,_),v in pairs.items() if t == target and b == budget), '20pairs x5groups per tissue')
                    metrics[method][f'{target}|{budget}'].append(math.fsum(values)/20)
            inputs.append(dict(seed=seed, method=method, rows=str(rows_path), rows_sha256=sha(rows_path), training=str(train_path), training_sha256=sha(train_path), checkpoint_sha256=evidence['checkpoint_sha256']))
    summary = {m:{k:dict(seeds=v, mean=statistics.mean(v), sample_std=statistics.stdev(v)) for k,v in values.items()} for m,values in metrics.items()}
    result = dict(status='complete_CRC11_tissue_plus_background_coverage_report', worker_pid=os.getpid(), classes=classes, background_id=2,
        background_positive_target_evaluated=False, coverage=coverage, metrics=summary, inputs=inputs, acceptance_summary_sha256=sha(accepted_path), palette_sha256=sha(palette), source_sha256=sha(__file__),
        scope='CRC all-label adapted local development, 220pairs x5groups x4budgets per seed/method; not main30/blind or 12-positive-target evaluation',
        remaining=['Main30 cohort/GT/physical-domain audit and evaluation', 'Closed12-class reference and background metrics remain separate', 'Replacement heldout ontology not yet selected; preserve existing failures'])
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    text = ['# CRC全标签逐类结果（三seed）', '', '11类组织+背景；背景不是交互正目标。主3+2，Dice百分数，三seed均值±样本标准差。开发集220对，非main30终评。', '', '| ID | 标签 | SAM-FT | WSI-SAM-FT | FewClick无细胞 | FewClick含细胞 |', '|---|---|---:|---:|---:|---:|']
    for c in classes:
        values = ['未设正目标']*4 if c['id'] == 2 else [f"{summary[m][str(c['id'])+'|3+2']['mean']*100:.2f} ± {summary[m][str(c['id'])+'|3+2']['sample_std']*100:.2f}" for m in methods]
        text.append('|'+str(c['id'])+'|'+c['name']+'|'+'|'.join(values)+'|')
    text += ['', '全部四预算和实际训练覆盖、来源空间数、输入/权重SHA见summary.json。脂肪正常参与训练；不以脂肪留出权重替代全标签权重。闭集CE分类参考单列，输入视野/先验/容量不同；不宣称等总监督或未见类别泛化。']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=result['status'], worker_pid=os.getpid(), output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
