"""Recount frozen BC/TC probe predictions and paired source-cluster intervals."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
PROBE = BASE / 'cell_linear_probe_1031_012400'
MANIFEST = BASE / 'p0_cell_probe_manifest_1030_182424/cells.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_f1(counts):
    # Last two axes are true/predicted class. Matches the fixed two-class probe.
    true_positive = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(-1) + counts.sum(-2)
    values = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator > 0)
    return values.mean(-1)


def confusion(true, predicted):
    result = np.zeros((2, 2), np.int64)
    np.add.at(result, (true, predicted), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    accepted = json.loads((PROBE / 'independent_acceptance.json').read_text())
    assert accepted['status'] == 'accepted_fixed_single_seed_cell_probe'
    manifest = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    validation = [r for r in manifest if r['split'] == 'val']
    assert len(validation) == 186
    ids = np.array([r['cell_id'] for r in validation])
    true = np.array([r['label'] - 1 for r in validation])
    predictions, point, evidence = {}, {}, {}
    for reference in accepted['rows']:
        arm, condition = reference['arm'], reference['condition']
        path = PROBE / 'data' / arm / (condition + '_predictions.npz')
        with np.load(path, allow_pickle=False) as file:
            assert np.array_equal(file['cell_ids'], ids) and np.array_equal(file['true'], true)
            predicted = file['pred'].copy()
            assert np.array_equal(predicted, file['logits'].argmax(1))
        score = float(macro_f1(confusion(true, predicted)))
        assert abs(score - reference['macro_F1']) < 1e-12
        key = arm + '|' + condition
        point[key] = score
        evidence[key] = sha(path)
        if condition == 'paired':
            predictions[arm] = predicted
    grouping = {}
    for key in ['field_id', 'slide', 'patient']:
        sources = sorted({r[key] for r in validation})
        assert all(sources)
        weights = np.random.default_rng(20261031).multinomial(len(sources), np.ones(len(sources)) / len(sources), size=2000)
        arms, replicated = {}, {}
        cell_counts = np.array([sum(r[key] == source for r in validation) for source in sources])
        for arm, predicted in predictions.items():
            matrices = np.stack([confusion(true[[r[key] == source for r in validation]],
                                           predicted[[r[key] == source for r in validation]]) for source in sources])
            totals = (weights @ matrices.reshape(len(sources), 4)).reshape(2000, 2, 2)
            values = macro_f1(totals)
            replicated[arm] = values
            arms[arm] = dict(macro_F1=point[arm + '|paired'], interval95=np.quantile(values, [.025, .975]).tolist(),
                             draws_missing_a_true_class=int((totals.sum(-1) == 0).any(1).sum()))
        differences = {}
        for candidate, reference in [('lemon', 'ctranspath'), ('lemon', 'phikon'), ('phikon', 'ctranspath')]:
            values = replicated[candidate] - replicated[reference]
            differences[candidate + '_minus_' + reference] = dict(
                difference=point[candidate + '|paired'] - point[reference + '|paired'],
                interval95=np.quantile(values, [.025, .975]).tolist())
        grouping[key] = dict(clusters=len(sources), bootstrap_draws=2000, seed=20261031,
                             min_cells_per_cluster=int(cell_counts.min()), max_cells_per_cluster=int(cell_counts.max()),
                             arms=arms, paired_differences=differences)
    result = dict(status='accepted_fixed_probe_source_uncertainty', worker_pid=os.getpid(),
                  source_sha256=sha(__file__), probe_acceptance_sha256=sha(PROBE / 'independent_acceptance.json'),
                  manifest_sha256=sha(MANIFEST), prediction_sha256=evidence, validation_cells=186,
                  validation_class_counts=np.bincount(true, minlength=2).tolist(), fixed_point_estimates=point,
                  by_grouping=grouping, training_updates=0, split_changed=False, candidate_changed=False,
                  limitations=['BC/TC historically exposed developer probe only, not tissue segmentation',
                               'one frozen probe head per arm; intervals conditional on fixed training and checkpoints',
                               'patient grouping is statistical resampling only; spatial-block split remains unchanged',
                               'same-case train/val and upstream pretraining exposure are not erased',
                               'CTransPath is common point-crop control, not native mask/context XCell'])
    (args.output / 'uncertainty.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 固定细胞probe的来源相关性', '', '186个验证细胞，99个视野、57张切片、41位患者；固定两类、固定预测，不重新训练或改split。', '',
             '| 重采样单位 | cluster数 | LEMON−CTransPath点中心F1 | 95%区间 |', '|---|---:|---:|---|']
    for key, value in grouping.items():
        change = value['paired_differences']['lemon_minus_ctranspath']
        lower, upper = change['interval95']
        lines.append(f"| {key} | {value['clusters']} | {change['difference']:+.4f} | [{lower:+.4f}, {upper:+.4f}] |")
    lines += ['', '这是固定模型下的来源重采样区间，不替代训练seed、独立确认或组织mask下游验证。',
              '患者分组只用于统计，不修改用户的空间块划分。CTransPath点中心对照不能冒充原XCell。']
    (args.output / 'uncertainty.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], by_grouping=grouping)), flush=True)


if __name__ == '__main__':
    main()
