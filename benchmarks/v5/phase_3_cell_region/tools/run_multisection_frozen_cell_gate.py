#!/usr/bin/env python3
"""Leave-one-section-out frozen H&E cell-adapter gate from verified dual-view features."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, help="legacy combined smoke artifact")
    parser.add_argument("--cache-dir", type=Path, help="cohort-sharded feature cache artifact")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate"))
    parser.add_argument("--timestamp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260907, 20260908, 20260909])
    return parser.parse_args()


class CellAdapter(torch.nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.adapter = torch.nn.Sequential(torch.nn.LayerNorm(2048), torch.nn.Linear(2048, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 256))
        self.classifier = torch.nn.Linear(256, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(values))


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True).clip(1e-6)
    return (train - mean) / std, (test - mean) / std


def train_fold(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, classes: list[str], seed: int, epochs: int, lr: float, device: torch.device) -> tuple[np.ndarray, float, dict]:
    torch.manual_seed(seed)
    x_train, x_test = standardize(x_train, x_test)
    class_index = {name: index for index, name in enumerate(classes)}
    train_y = torch.tensor([class_index[value] for value in y_train], dtype=torch.long, device=device)
    counts = torch.bincount(train_y, minlength=len(classes)).float().clamp_min(1)
    weights = (len(train_y) / (len(classes) * counts)).to(device)
    model = CellAdapter(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train(); logits = model(x); loss = F.cross_entropy(logits, train_y, weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.tensor(x_test, dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return np.asarray([classes[index] for index in prediction]), float(loss.detach().cpu()), state


def load_features(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load either legacy combined smoke output or an identity-checked shard cache."""
    if bool(args.smoke_dir) == bool(args.cache_dir):
        raise ValueError("provide exactly one of --smoke-dir or --cache-dir")
    if args.smoke_dir:
        cells = pd.read_csv(args.smoke_dir / "smoked_cells.csv", dtype={"cell_id": str})
        # The legacy extractor emits one cohort at a time in sorted cohort order.
        cells = cells.sort_values("cohort_id", kind="stable").reset_index(drop=True)
        features = np.load(args.smoke_dir / "phikon_dualview_features.npz")
        observed_ids = features["cell_id"].astype(str)
        expected_ids = cells.cell_id.to_numpy(dtype=str)
        if len(cells) != len(observed_ids) or not np.array_equal(expected_ids, observed_ids):
            raise RuntimeError("feature-to-cell identity mismatch")
        values = np.concatenate([features["phikon_cls_256px"], features["phikon_cls_384px"]], axis=1).astype(np.float32)
        return cells, values, str(args.smoke_dir)
    index = json.loads((args.cache_dir / "index.json").read_text(encoding="utf-8"))
    if not index.get("all_complete"):
        raise RuntimeError("cache index is not complete")
    frames, matrices = [], []
    for cohort in sorted(index["cohorts"]):
        shard = args.cache_dir / cohort
        cells = pd.read_csv(shard / "cells.csv", dtype={"cell_id": str})
        features = np.load(shard / "phikon_dualview_features.npz")
        ids = features["cell_id"].astype(str)
        if len(cells) != len(ids) or not np.array_equal(cells.cell_id.to_numpy(dtype=str), ids):
            raise RuntimeError(f"{cohort}: feature-to-cell identity mismatch")
        if not cells.cohort_id.eq(cohort).all():
            raise RuntimeError(f"{cohort}: shard cohort identity mismatch")
        values = np.concatenate([features["phikon_cls_256px"], features["phikon_cls_384px"]], axis=1).astype(np.float32)
        frames.append(cells); matrices.append(values)
    return pd.concat(frames, ignore_index=True), np.concatenate(matrices, axis=0), str(args.cache_dir)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and lr must be positive")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"multisection_frozen_gate_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    cells, values, feature_source = load_features(args)
    if values.shape != (len(cells), 2048) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid dual-view feature matrix: {values.shape}")
    classes = sorted(cells.broad_annotation.unique())
    all_truth: dict[str, list[str]] = {"paired_he": [], "shuffled_he": []}
    all_prediction: dict[str, list[str]] = {"paired_he": [], "shuffled_he": []}
    fold_rows: list[dict] = []
    for held_out in sorted(cells.cohort_id.unique()):
        test_mask = cells.cohort_id.eq(held_out).to_numpy()
        train_mask = ~test_mask
        train_cohorts = sorted(cells.loc[train_mask, "cohort_id"].unique())
        if held_out in train_cohorts:
            raise RuntimeError("split leakage")
        for arm in ("paired_he", "shuffled_he"):
            for seed in args.seeds:
                x_train = values[train_mask].copy()
                if arm == "shuffled_he":
                    x_train = x_train[np.random.default_rng(seed).permutation(len(x_train))]
                prediction, final_loss, state = train_fold(x_train, cells.loc[train_mask, "broad_annotation"].to_numpy(), values[test_mask], classes, seed, args.epochs, args.lr, device)
                truth = cells.loc[test_mask, "broad_annotation"].to_numpy()
                all_truth[arm].extend(truth); all_prediction[arm].extend(prediction)
                metric = {"held_out_cohort": held_out, "train_cohorts": ";".join(train_cohorts), "arm": arm, "seed": seed, "train_cells": int(train_mask.sum()), "test_cells": int(test_mask.sum()), "final_loss": final_loss, "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(truth, prediction)}
                fold_rows.append(metric)
                checkpoint = {"classes": classes, "held_out_cohort": held_out, "train_cohorts": train_cohorts, "arm": arm, "seed": seed, "state_dict": state}
                torch.save(checkpoint, output / f"checkpoint_{arm}_{held_out}_seed{seed}.pt")
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    summaries = {}
    for arm in all_truth:
        summaries[arm] = {"macro_f1": float(f1_score(all_truth[arm], all_prediction[arm], average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(all_truth[arm], all_prediction[arm])), "fold_macro_f1_mean": float(fold_frame[fold_frame.arm.eq(arm)].macro_f1.mean()), "fold_balanced_accuracy_mean": float(fold_frame[fold_frame.arm.eq(arm)].balanced_accuracy.mean())}
    report = {"purpose": "frozen H&E dual-view cross-cohort cell-adapter gate; RNA is not model input", "manifest_dir": str(args.manifest_dir), "feature_source": feature_source, "manifest_sha256": hashlib.sha256((args.manifest_dir / "sampled_cells.csv").read_bytes()).hexdigest(), "smoked_cells": len(cells), "feature_shape": list(values.shape), "classes": classes, "held_out_cohorts": sorted(cells.cohort_id.unique()), "seeds": args.seeds, "epochs": args.epochs, "summaries": summaries, "acceptance": "paired_he macro-F1 and balanced accuracy must exceed shuffled_he; this is not donor-level generalization"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "summaries": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
