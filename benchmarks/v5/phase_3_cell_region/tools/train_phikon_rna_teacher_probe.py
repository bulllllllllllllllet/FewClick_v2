#!/usr/bin/env python3
"""Train H&E-only Phikon adapters with optional paired RNA teacher targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SCRIPT_DIR), str(SRC_ROOT)]
from phikon_v2 import CellFeatureAdapter  # noqa: E402
from run_xenium_rna_pilot import load_transcripts, normalized_counts, split_gene_views, svd_features  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/backbone_probe/phikon_backbone_probe_0907_024957"))
    p.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260907, 20260908, 20260909])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rna-weight", type=float, default=0.25)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def train_fold(x_train, y_train, r_train, x_test, *, classes, seed, epochs, lr, rna_weight, shuffled, device):
    torch.manual_seed(seed)
    adapter = CellFeatureAdapter().to(device)
    classifier = torch.nn.Linear(256, len(classes)).to(device)
    regressor = torch.nn.Linear(256, r_train.shape[1]).to(device)
    inverse = {label: i for i, label in enumerate(classes)}
    labels = torch.tensor([inverse[x] for x in y_train], device=device)
    counts = torch.bincount(labels, minlength=len(classes)).float().clamp_min(1)
    class_weight = (len(labels) / (len(classes) * counts)).to(device)
    x_mean, x_std = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True).clip(1e-6)
    r_mean, r_std = r_train.mean(0, keepdims=True), r_train.std(0, keepdims=True).clip(1e-6)
    x_train, x_test = (x_train - x_mean) / x_std, (x_test - x_mean) / x_std
    r_train = (r_train - r_mean) / r_std
    target = torch.tensor(r_train[np.random.default_rng(seed).permutation(len(r_train))] if shuffled else r_train, dtype=torch.float32, device=device)
    feature = torch.tensor(x_train, dtype=torch.float32, device=device)
    optimiser = torch.optim.AdamW(list(adapter.parameters()) + list(classifier.parameters()) + list(regressor.parameters()), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        embedding = adapter(feature)
        loss = torch.nn.functional.cross_entropy(classifier(embedding), labels, weight=class_weight)
        if rna_weight:
            loss = loss + rna_weight * torch.nn.functional.mse_loss(regressor(embedding), target)
        loss.backward(); optimiser.step()
    with torch.inference_mode():
        prediction = classifier(adapter(torch.tensor(x_test, dtype=torch.float32, device=device))).argmax(1).cpu().numpy()
    return np.asarray([classes[x] for x in prediction])


def main() -> None:
    a = args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    device = torch.device(a.device)
    cells = pd.read_csv(a.probe_dir / "cells.csv", dtype={"cell_id": str})
    feature = np.load(a.probe_dir / "phikon_v2_1024.npy")
    transcript = load_transcripts(a.data_root / "transcripts/transcripts.parquet", set(cells.cell_id), 1000, 2000, 1500, 20.0)
    counts = normalized_counts(transcript, cells.cell_id.tolist(), 30)
    order = pd.Index(cells.cell_id).get_indexer(counts.index)
    cells, feature = cells.iloc[order].reset_index(drop=True), feature[order]
    genes, _ = split_gene_views(counts.columns.tolist())
    rna = svd_features(counts[genes].to_numpy())
    feature, rna = feature.astype(np.float32), rna.astype(np.float32)
    groups = cells.region.to_numpy(); labels = cells.annotation.to_numpy(); classes = sorted(set(labels))
    rows = []
    for arm, weight, shuffled in (("h_and_e_label_only", 0.0, False), ("h_and_e_plus_paired_rna_teacher", a.rna_weight, False), ("h_and_e_plus_shuffled_rna_teacher", a.rna_weight, True)):
        for seed in a.seeds:
            predictions, truths = [], []
            for train, test in GroupKFold(n_splits=4).split(feature, labels, groups):
                predicted = train_fold(feature[train], labels[train], rna[train], feature[test], classes=classes, seed=seed, epochs=a.epochs, lr=a.lr, rna_weight=weight, shuffled=shuffled, device=device)
                predictions.extend(predicted); truths.extend(labels[test])
            rows.append({"arm": arm, "seed": seed, "macro_f1": f1_score(truths, predictions, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(truths, predictions)})
    result = pd.DataFrame(rows); result.to_csv(a.output_dir / "seed_metrics.csv", index=False)
    grouped = result.groupby("arm")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
    summary = {
        arm: {metric: {stat: float(grouped.loc[arm, (metric, stat)]) for stat in ("mean", "std")} for metric in ("macro_f1", "balanced_accuracy")}
        for arm in grouped.index
    }
    metrics = {"purpose": "exploratory RNA-teacher adapter training; H&E-only at test time", "cells": len(cells), "classes": classes, "seeds": a.seeds, "epochs": a.epochs, "rna_dim": int(rna.shape[1]), "summary": summary, "acceptance": "paired RNA must exceed label-only and shuffled RNA across seeds; otherwise do not claim RNA-teacher gain"}
    (a.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(a.output_dir), **metrics}, ensure_ascii=False))


if __name__ == "__main__": main()
