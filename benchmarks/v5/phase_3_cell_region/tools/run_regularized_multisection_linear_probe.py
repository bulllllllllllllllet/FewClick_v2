#!/usr/bin/env python3
"""Run regularized, shared-label H&E-only probes on cached multisection features.

This deliberately uses a low-capacity L2 logistic head.  It evaluates only labels
present in both sides of an outer split and compares real H&E pairing to a training
feature-row shuffle.  RNA is never loaded as a model input.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe"))
    parser.add_argument("--timestamp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260907, 20260908, 20260909])
    parser.add_argument("--c-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--protocols", nargs="+", choices=["loco_all", "same_platform_cross_cancer", "same_cancer_cross_platform"], default=["loco_all", "same_platform_cross_cancer", "same_cancer_cross_platform"])
    return parser.parse_args()


def load_cache(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if not index.get("all_complete"):
        raise RuntimeError("cache index is incomplete")
    frames, matrices = [], []
    for cohort in sorted(index["cohorts"]):
        shard = cache_dir / cohort
        cells = pd.read_csv(shard / "cells.csv", dtype={"cell_id": str})
        features = np.load(shard / "phikon_dualview_features.npz")
        if not np.array_equal(cells.cell_id.to_numpy(dtype=str), features["cell_id"].astype(str)):
            raise RuntimeError(f"{cohort}: feature-to-cell identity mismatch")
        if not cells.cohort_id.eq(cohort).all():
            raise RuntimeError(f"{cohort}: shard cohort identity mismatch")
        values = np.concatenate([features["phikon_cls_256px"], features["phikon_cls_384px"]], axis=1).astype(np.float32)
        if values.shape != (len(cells), 2048) or not np.isfinite(values).all():
            raise RuntimeError(f"{cohort}: invalid feature matrix {values.shape}")
        frames.append(cells)
        matrices.append(values)
    return pd.concat(frames, ignore_index=True), np.concatenate(matrices, axis=0)


def build_splits(cells: pd.DataFrame, protocol: str) -> list[tuple[str, list[str], str]]:
    cohorts = sorted(cells.cohort_id.unique())
    if protocol == "loco_all":
        return [(held, [item for item in cohorts if item != held], held) for held in cohorts]
    if protocol == "same_platform_cross_cancer":
        xenium = sorted(cells.loc[cells.platform.eq("Xenium"), "cohort_id"].unique())
        return [(held, [item for item in xenium if item != held], held) for held in xenium]
    if protocol == "same_cancer_cross_platform":
        rows = []
        for cancer in ("COAD", "HCC"):
            pair = sorted(cells.loc[cells.cancer.eq(cancer), "cohort_id"].unique())
            if len(pair) != 2:
                raise RuntimeError(f"{cancer}: expected exactly two platform cohorts, found {pair}")
            rows.extend((held, [item for item in pair if item != held], held) for held in pair)
        return rows
    raise ValueError(protocol)


def shared_labels(y_train: np.ndarray, y_test: np.ndarray) -> list[str]:
    return sorted(set(y_train).intersection(y_test))


def make_model(c_value: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=2000),
    )


def inner_splits(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        return list(GroupKFold(n_splits=len(unique_groups)).split(np.zeros(len(y)), y, groups))
    counts = pd.Series(y).value_counts()
    folds = min(3, int(counts.min()))
    if folds < 2:
        raise RuntimeError("cannot tune C: fewer than two examples in a class")
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260907).split(np.zeros(len(y)), y))


def select_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, candidates: list[float]) -> tuple[float, list[dict]]:
    results = []
    for value in candidates:
        fold_scores = []
        for inner_train, inner_val in inner_splits(y, groups):
            labels = shared_labels(y[inner_train], y[inner_val])
            train_keep = np.isin(y[inner_train], labels)
            val_keep = np.isin(y[inner_val], labels)
            if len(labels) < 2 or not train_keep.any() or not val_keep.any():
                continue
            model = make_model(value)
            model.fit(x[inner_train][train_keep], y[inner_train][train_keep])
            prediction = model.predict(x[inner_val][val_keep])
            truth = y[inner_val][val_keep]
            fold_scores.append((f1_score(truth, prediction, average="macro", zero_division=0), balanced_accuracy_score(truth, prediction)))
        if not fold_scores:
            raise RuntimeError("no valid train-only inner split for C selection")
        results.append({"c_value": value, "inner_macro_f1": float(np.mean([item[0] for item in fold_scores])), "inner_balanced_accuracy": float(np.mean([item[1] for item in fold_scores])), "inner_folds": len(fold_scores)})
    best = max(results, key=lambda row: (row["inner_macro_f1"], row["inner_balanced_accuracy"], -row["c_value"]))
    return float(best["c_value"]), results


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in args.c_values):
        raise ValueError("all C values must be positive")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"regularized_linear_cell_probe_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cells, values = load_cache(args.cache_dir)
    metric_rows, tuning_rows = [], []
    for protocol in args.protocols:
        for held_out, train_cohorts, split_name in build_splits(cells, protocol):
            test_mask = cells.cohort_id.eq(held_out).to_numpy()
            train_mask = cells.cohort_id.isin(train_cohorts).to_numpy()
            if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
                raise RuntimeError(f"{protocol}/{held_out}: invalid cohort split")
            train_y_all = cells.loc[train_mask, "broad_annotation"].to_numpy()
            test_y_all = cells.loc[test_mask, "broad_annotation"].to_numpy()
            labels = shared_labels(train_y_all, test_y_all)
            if len(labels) < 2:
                raise RuntimeError(f"{protocol}/{held_out}: fewer than two shared labels")
            train_keep = np.isin(train_y_all, labels)
            test_keep = np.isin(test_y_all, labels)
            x_train = values[train_mask][train_keep]
            y_train = train_y_all[train_keep]
            x_test = values[test_mask][test_keep]
            y_test = test_y_all[test_keep]
            groups = cells.loc[train_mask, "cohort_id"].to_numpy()[train_keep]
            excluded_train = sorted(set(train_y_all).difference(labels))
            excluded_test = sorted(set(test_y_all).difference(labels))
            # Select regularization once from the real training pairing only.  The
            # same train-only choice is then locked for shuffled controls, so the
            # null arm cannot gain an advantage from repeated hyperparameter search.
            c_value, tuning = select_c(x_train, y_train, groups, args.c_values)
            for row in tuning:
                tuning_rows.append({"protocol": protocol, "held_out_cohort": held_out, "arm": "paired_train_selection", "seed": "", **row})
            for arm in ("paired_he", "shuffled_he"):
                for seed in args.seeds:
                    observed_train = x_train.copy()
                    if arm == "shuffled_he":
                        observed_train = observed_train[np.random.default_rng(seed).permutation(len(observed_train))]
                    model = make_model(c_value)
                    model.fit(observed_train, y_train)
                    prediction = model.predict(x_test)
                    metric_rows.append({
                        "protocol": protocol, "split": split_name, "held_out_cohort": held_out, "train_cohorts": ";".join(train_cohorts), "arm": arm, "seed": seed,
                        "selected_c": c_value, "shared_labels": ";".join(labels), "shared_class_count": len(labels),
                        "excluded_train_labels": ";".join(excluded_train), "excluded_test_labels": ";".join(excluded_test),
                        "train_cells": len(y_train), "test_cells": len(y_test),
                        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                    })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "split_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / "inner_tuning_metrics.csv", index=False)
    aggregate = metrics.groupby(["protocol", "arm"])[["macro_f1", "balanced_accuracy"]].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "protocol_summary.csv", index=False)
    paired = metrics.pivot(index=["protocol", "held_out_cohort", "seed"], columns="arm", values=["macro_f1", "balanced_accuracy"])
    paired.columns = ["_".join(column) for column in paired.columns]
    paired["macro_f1_delta_paired_minus_shuffled"] = paired["macro_f1_paired_he"] - paired["macro_f1_shuffled_he"]
    paired["balanced_accuracy_delta_paired_minus_shuffled"] = paired["balanced_accuracy_paired_he"] - paired["balanced_accuracy_shuffled_he"]
    paired.reset_index().to_csv(output / "paired_deltas.csv", index=False)
    report = {
        "purpose": "regularized shared-label H&E-only linear probe; RNA is not model input",
        "cache_dir": str(args.cache_dir), "cells": len(cells), "feature_shape": list(values.shape), "protocols": args.protocols,
        "seeds": args.seeds, "c_values": args.c_values,
        "acceptance": "within each protocol, paired mean macro-F1 and balanced accuracy must both exceed shuffled; this remains section-level, not donor-level evidence",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "protocol_summary": aggregate.reset_index().to_dict(orient="records")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
