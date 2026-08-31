#!/usr/bin/env python3
"""Evaluate fixed XCell features and paired RNA against Xenium cell annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))
from run_xenium_rna_pilot import load_transcripts, normalized_counts, spatial_cv, split_gene_views, svd_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcell-artifact", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    parser.add_argument("--min-class-total", type=int, default=30)
    parser.add_argument("--shuffle-seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.output_dir / "metrics.json").exists(): raise FileExistsError(args.output_dir / "metrics.json")
    summary = pd.read_csv(args.xcell_artifact / "cell_summary.csv", dtype={"cell_id": str})
    xcell = np.load(args.xcell_artifact / "xcell_reg64.npy")
    if len(summary) != len(xcell): raise RuntimeError("XCell feature rows do not match summary rows")
    annotations = pd.read_csv(args.annotations, dtype={"cell_id": str})
    cells = summary.merge(annotations, on="cell_id", how="left", validate="one_to_one")
    totals = cells.annotation.value_counts()
    eligible = totals[totals >= args.min_class_total].index
    cells = cells[cells.annotation.isin(eligible)].copy()
    block_counts = pd.crosstab(cells.annotation, cells.region)
    good = block_counts.index[(block_counts.reindex(columns=range(4), fill_value=0) > 0).all(axis=1)]
    cells = cells[cells.annotation.isin(good)].copy()
    original_position = pd.Index(summary.cell_id).get_indexer(cells.cell_id)
    if (original_position < 0).any() or len(good) < 2: raise RuntimeError("insufficient aligned annotated cells")
    transcript = load_transcripts(args.data_root / "transcripts/transcripts.parquet", set(summary.cell_id), 1000, 2000, 1500, 20.0)
    counts = normalized_counts(transcript, summary.cell_id.tolist(), 30)
    if not counts.index.equals(pd.Index(summary.cell_id)): raise RuntimeError("RNA ordering differs from fixed XCell artifact")
    feature_genes, _ = split_gene_views(counts.columns.tolist())
    rna = svd_features(counts[feature_genes].to_numpy())[original_position]
    labels = cells.annotation.to_numpy()
    locations = (cells.center_x.to_numpy(), cells.center_y.to_numpy())
    visual = spatial_cv(xcell[original_position], labels, *locations, 1000, 2000, 1500)
    fused = spatial_cv(np.concatenate((xcell[original_position], rna), axis=1), labels, *locations, 1000, 2000, 1500)
    shuffled = spatial_cv(np.concatenate((xcell[original_position], rna[np.random.default_rng(args.shuffle_seed).permutation(len(rna))]), axis=1), labels, *locations, 1000, 2000, 1500)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output_dir / "cell_summary_with_annotation.csv", index=False)
    metrics = {"purpose": "dataset-provider cell annotation classification; not final FewClick segmentation", "retained_cells": len(cells), "classes": sorted(good.tolist()), "class_by_spatial_block": pd.crosstab(cells.annotation, cells.region).to_dict(), "xcell_visual": visual, "xcell_plus_rna": fused, "xcell_plus_shuffled_rna": shuffled, "rna_gain_macro_f1": fused["macro_f1_mean"] - visual["macro_f1_mean"], "rna_gain_balanced_accuracy": fused["balanced_accuracy_mean"] - visual["balanced_accuracy_mean"], "real_minus_shuffled_macro_f1": fused["macro_f1_mean"] - shuffled["macro_f1_mean"], "real_minus_shuffled_balanced_accuracy": fused["balanced_accuracy_mean"] - shuffled["balanced_accuracy_mean"], "acceptance": "correct RNA exceeds shuffled RNA by >= 0.10 on both metrics"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output_dir), **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
