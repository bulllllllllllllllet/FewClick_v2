#!/usr/bin/env python3
"""Diagnose cached frozen-gate checkpoints without updating model parameters."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_multisection_frozen_cell_gate import CellAdapter, standardize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_cache(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if not index.get("all_complete"):
        raise RuntimeError("cache is incomplete")
    frames, matrices = [], []
    for cohort in sorted(index["cohorts"]):
        shard = cache_dir / cohort
        cells = pd.read_csv(shard / "cells.csv", dtype={"cell_id": str})
        features = np.load(shard / "phikon_dualview_features.npz")
        if not np.array_equal(cells.cell_id.to_numpy(dtype=str), features["cell_id"].astype(str)):
            raise RuntimeError(f"{cohort}: cache identity mismatch")
        frames.append(cells)
        matrices.append(np.concatenate([features["phikon_cls_256px"], features["phikon_cls_384px"]], axis=1).astype(np.float32))
    return pd.concat(frames, ignore_index=True), np.concatenate(matrices, axis=0)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    cells, values = load_cache(args.cache_dir)
    classes = sorted(cells.broad_annotation.unique())
    lookup = {name: idx for idx, name in enumerate(classes)}
    rows, predictions = [], defaultdict(lambda: {"truth": [], "pred": []})
    for checkpoint_path in sorted(args.gate_dir.glob("checkpoint_*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        held_out, arm, seed = checkpoint["held_out_cohort"], checkpoint["arm"], checkpoint["seed"]
        test_mask = cells.cohort_id.eq(held_out).to_numpy(); train_mask = ~test_mask
        x_train, x_test = standardize(values[train_mask], values[test_mask])
        model = CellAdapter(len(classes)).eval(); model.load_state_dict(checkpoint["state_dict"])
        with torch.inference_mode():
            pred_idx = model(torch.tensor(x_test, dtype=torch.float32)).argmax(1).numpy()
        truth = cells.loc[test_mask, "broad_annotation"].to_numpy(); pred = np.asarray([classes[i] for i in pred_idx])
        predictions[(held_out, arm)]["truth"].extend(truth); predictions[(held_out, arm)]["pred"].extend(pred)
        rows.append({"checkpoint": checkpoint_path.name, "held_out_cohort": held_out, "arm": arm, "seed": seed, "macro_f1": f1_score(truth, pred, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(truth, pred)})
    pd.DataFrame(rows).to_csv(args.output_dir / "checkpoint_metrics.csv", index=False)
    cohort_rows, class_rows = [], []
    for (cohort, arm), result in sorted(predictions.items()):
        truth, pred = np.asarray(result["truth"]), np.asarray(result["pred"])
        cohort_rows.append({"held_out_cohort": cohort, "arm": arm, "macro_f1": f1_score(truth, pred, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(truth, pred), "n_predictions": len(truth)})
        precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=classes, zero_division=0)
        for name, p, r, score, n in zip(classes, precision, recall, f1, support):
            class_rows.append({"held_out_cohort": cohort, "arm": arm, "class": name, "precision": p, "recall": r, "f1": score, "support": n})
        matrix = confusion_matrix(truth, pred, labels=classes)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(args.output_dir / f"confusion_{cohort}_{arm}.csv")
    pd.DataFrame(cohort_rows).to_csv(args.output_dir / "cohort_aggregate_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(args.output_dir / "class_metrics.csv", index=False)
    paired = pd.DataFrame(cohort_rows).pivot(index="held_out_cohort", columns="arm", values=["macro_f1", "balanced_accuracy"])
    paired.columns = ["_".join(column) for column in paired.columns]
    paired["macro_f1_delta_paired_minus_shuffled"] = paired.macro_f1_paired_he - paired.macro_f1_shuffled_he
    paired["balanced_accuracy_delta_paired_minus_shuffled"] = paired.balanced_accuracy_paired_he - paired.balanced_accuracy_shuffled_he
    paired.reset_index().to_csv(args.output_dir / "cohort_deltas.csv", index=False)
    report = {"purpose": "checkpoint-only failure diagnosis; no parameters updated and RNA is not input", "cells": len(cells), "feature_shape": list(values.shape), "classes": classes, "checkpoint_count": len(rows), "output_files": ["cohort_deltas.csv", "class_metrics.csv", "checkpoint_metrics.csv", "confusion_<cohort>_<arm>.csv"]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output_dir), **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
