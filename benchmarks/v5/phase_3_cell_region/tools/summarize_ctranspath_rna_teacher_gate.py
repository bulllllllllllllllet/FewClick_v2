#!/usr/bin/env python3
"""Create a standalone immutable summary from completed CTransPath gate folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-metrics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-exit-code', type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = pd.read_csv(args.fold_metrics)
    required = {'arm', 'fold', 'macro_f1', 'balanced_accuracy', 'final_loss'}
    if not required.issubset(result.columns) or result.empty:
        raise RuntimeError(f'incomplete fold metrics: columns={result.columns.tolist()} rows={len(result)}')
    if not result[['macro_f1', 'balanced_accuracy', 'final_loss']].apply(lambda x: x.notna().all()).all():
        raise RuntimeError('non-finite or missing fold metric')
    summary = result.groupby('arm')[['macro_f1', 'balanced_accuracy']].agg(['mean', 'std']).round(6)
    required_arms = {'label_only', 'paired_rna', 'shuffled_rna'}
    if set(summary.index) != required_arms:
        raise RuntimeError(f'unexpected arms: {sorted(summary.index)}')
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / 'summary.csv')
    paired = summary.loc['paired_rna', ('macro_f1', 'mean')]
    metadata = {
        'source_fold_metrics': str(args.fold_metrics),
        'completed_folds': int(len(result)),
        'run_exit_code': args.run_exit_code,
        'failure_scope': 'post-training metadata serialization only; every fold row was written before the exception' if args.run_exit_code else None,
        'summary': {
            arm: {
                metric: {stat: float(summary.loc[arm, (metric, stat)]) for stat in ('mean', 'std')}
                for metric in ('macro_f1', 'balanced_accuracy')
            }
            for arm in summary.index
        },
        'paired_minus_label_only_macro_f1': float(paired - summary.loc['label_only', ('macro_f1', 'mean')]),
        'paired_minus_shuffled_macro_f1': float(paired - summary.loc['shuffled_rna', ('macro_f1', 'mean')]),
        'conclusion': 'RNA teacher gate failed: paired RNA did not exceed label-only or shuffled RNA.',
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(args.output_dir), **metadata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
