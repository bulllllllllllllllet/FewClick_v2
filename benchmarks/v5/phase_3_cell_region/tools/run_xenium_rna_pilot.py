#!/usr/bin/env python3
"""Run a bounded COAD Xenium RNA-feature evidence pilot.

The pilot uses a spatially held-out RNA program to compare handcrafted H&E /
cell-geometry features with those features augmented by an independent RNA
feature view.  It is a representation gate, not a FewClick segmentation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyvips
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot"))
    parser.add_argument("--timestamp")
    parser.add_argument("--x", type=int, default=1000)
    parser.add_argument("--y", type=int, default=2000)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--qv-threshold", type=float, default=20.0)
    parser.add_argument("--min-transcripts", type=int, default=30)
    parser.add_argument("--max-cells", type=int, default=1200)
    parser.add_argument("--clusters", type=int, default=3)
    return parser.parse_args()


def cell_boxes(path: Path, x0: int, y0: int, size: int) -> pd.DataFrame:
    """Read only polygons touching the requested window, then retain contained cells."""
    x1, y1 = x0 + size, y0 + size
    values: dict[str, list[float]] = defaultdict(lambda: [np.inf, np.inf, -np.inf, -np.inf, 0.0])
    for chunk in pd.read_csv(path, usecols=["cell_id", "vertex_x", "vertex_y"], chunksize=500_000):
        chunk = chunk[(chunk.vertex_x >= x0 - 64) & (chunk.vertex_x < x1 + 64) & (chunk.vertex_y >= y0 - 64) & (chunk.vertex_y < y1 + 64)]
        for row in chunk.itertuples(index=False):
            box = values[str(row.cell_id)]
            box[0] = min(box[0], float(row.vertex_x)); box[1] = min(box[1], float(row.vertex_y))
            box[2] = max(box[2], float(row.vertex_x)); box[3] = max(box[3], float(row.vertex_y)); box[4] += 1
    records = [
        {"cell_id": key, "min_x": box[0], "min_y": box[1], "max_x": box[2], "max_y": box[3], "vertices": int(box[4])}
        for key, box in values.items()
        if box[0] >= x0 and box[1] >= y0 and box[2] < x1 and box[3] < y1 and box[4] >= 3
    ]
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError("no fully contained cell polygons in selected window")
    result["center_x"] = (result.min_x + result.max_x) / 2
    result["center_y"] = (result.min_y + result.max_y) / 2
    result["bbox_width"] = result.max_x - result.min_x
    result["bbox_height"] = result.max_y - result.min_y
    return result


def load_transcripts(path: Path, cells: set[str], x0: int, y0: int, size: int, qv: float) -> pd.DataFrame:
    x1, y1 = x0 + size, y0 + size
    source = ds.dataset(path, format="parquet")
    expression = (
        (ds.field("x_location") >= x0) & (ds.field("x_location") < x1)
        & (ds.field("y_location") >= y0) & (ds.field("y_location") < y1)
        & (ds.field("qv") >= qv) & ds.field("is_gene")
    )
    table = source.to_table(columns=["cell_id", "feature_name", "x_location", "y_location", "qv"], filter=expression)
    data = table.to_pandas()
    data = data[(data.cell_id != "UNASSIGNED") & data.cell_id.isin(cells)]
    data = data[~data.feature_name.str.contains("control", case=False, na=False)]
    if data.empty:
        raise RuntimeError("no assigned, quality-filtered transcripts matched selected cell polygons")
    return data


def normalized_counts(transcripts: pd.DataFrame, cell_ids: list[str], min_transcripts: int) -> pd.DataFrame:
    counts = pd.crosstab(transcripts.cell_id, transcripts.feature_name).reindex(cell_ids, fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_transcripts]
    counts = counts.loc[:, counts.sum(axis=0) >= 5]
    if counts.shape[0] < 80 or counts.shape[1] < 12:
        raise RuntimeError(f"insufficient RNA matrix after filtering: {counts.shape}")
    scale = counts.sum(axis=1).to_numpy(dtype=np.float64, copy=False)
    return pd.DataFrame(np.log1p(counts.to_numpy(dtype=np.float64) / scale[:, None] * 1e4), index=counts.index, columns=counts.columns)


def split_gene_views(columns: list[str]) -> tuple[list[str], list[str]]:
    feature, label = [], []
    for gene in columns:
        bucket = int(hashlib.sha256(gene.encode()).hexdigest()[:8], 16) % 3
        (label if bucket == 0 else feature).append(gene)
    if len(feature) < 8 or len(label) < 8:
        raise RuntimeError(f"gene views too small: feature={len(feature)} label={len(label)}")
    return feature, label


def visual_features(he_path: Path, boxes: pd.DataFrame, x0: int, y0: int, size: int) -> np.ndarray:
    image = pyvips.Image.new_from_file(str(he_path), access="sequential")
    crop = image.crop(x0, y0, size, size)
    array = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(size, size, crop.bands))[:, :, :3]
    values = []
    for row in boxes.itertuples(index=False):
        left = max(int(np.floor(row.min_x)) - x0, 0); right = min(int(np.ceil(row.max_x)) - x0 + 1, size)
        top = max(int(np.floor(row.min_y)) - y0, 0); bottom = min(int(np.ceil(row.max_y)) - y0 + 1, size)
        patch = array[top:bottom, left:right]
        if patch.size == 0:
            raise RuntimeError(f"empty H&E crop for {row.cell_id}")
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        values.append(np.concatenate((
            patch.mean((0, 1)), patch.std((0, 1)),
            np.array([gray.mean(), gray.std(), grad.mean(), grad.std(), row.bbox_width, row.bbox_height, row.vertices], dtype=np.float64),
        )))
    return np.asarray(values, dtype=np.float64), array


def svd_features(matrix: np.ndarray, limit: int = 16) -> np.ndarray:
    components = min(limit, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 2:
        raise RuntimeError(f"insufficient matrix shape for SVD: {matrix.shape}")
    return TruncatedSVD(n_components=components, random_state=20260831).fit_transform(matrix)


def spatial_cv(features: np.ndarray, labels: np.ndarray, x: np.ndarray, y: np.ndarray, x0: int, y0: int, size: int) -> dict:
    groups = (x >= x0 + size / 2).astype(np.int64) + 2 * (y >= y0 + size / 2).astype(np.int64)
    if len(np.unique(groups)) < 4:
        raise RuntimeError("selected cells do not cover all four spatial blocks")
    folds = []
    for train, test in GroupKFold(n_splits=4).split(features, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260831))
        model.fit(features[train], labels[train])
        prediction = model.predict(features[test])
        folds.append({
            "test_block": int(groups[test][0]),
            "cells": int(len(test)),
            "macro_f1": float(f1_score(labels[test], prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
        })
    return {
        "folds": folds,
        "macro_f1_mean": float(np.mean([item["macro_f1"] for item in folds])),
        "balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in folds])),
    }


def save_overlay(path: Path, image: np.ndarray, cells: pd.DataFrame, labels: np.ndarray, x0: int, y0: int) -> None:
    max_side = 1200
    scale = min(max_side / image.shape[1], max_side / image.shape[0], 1.0)
    canvas = Image.fromarray(image).resize((round(image.shape[1] * scale), round(image.shape[0] * scale)))
    draw = ImageDraw.Draw(canvas)
    colors = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15)]
    for row, label in zip(cells.itertuples(index=False), labels, strict=True):
        cx = (row.center_x - x0) * scale; cy = (row.center_y - y0) * scale
        radius = max(2, round(3 * scale)); color = colors[int(label) % len(colors)]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(255, 255, 255))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"xenium_rna_feature_pilot_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    he = args.data_root / "HE.tif"
    boundaries = args.data_root / "segmentation_mask/cell_boundaries.csv"
    transcripts = args.data_root / "transcripts/transcripts.parquet"
    for path in (he, boundaries, transcripts):
        if not path.is_file():
            raise FileNotFoundError(path)

    boxes = cell_boxes(boundaries, args.x, args.y, args.size)
    rna = load_transcripts(transcripts, set(boxes.cell_id), args.x, args.y, args.size, args.qv_threshold)
    count = normalized_counts(rna, boxes.cell_id.tolist(), args.min_transcripts)
    boxes = boxes.set_index("cell_id").loc[count.index].reset_index()
    if len(boxes) > args.max_cells:
        boxes = boxes.sample(args.max_cells, random_state=20260831).sort_values("cell_id").reset_index(drop=True)
        count = count.loc[boxes.cell_id]
    rna_feature_genes, label_genes = split_gene_views(count.columns.tolist())
    rna_feature = svd_features(count[rna_feature_genes].to_numpy())
    molecular = svd_features(count[label_genes].to_numpy())
    labels = KMeans(n_clusters=args.clusters, n_init=20, random_state=20260831).fit_predict(molecular)
    visual, crop = visual_features(he, boxes, args.x, args.y, args.size)
    visual_result = spatial_cv(visual, labels, boxes.center_x.to_numpy(), boxes.center_y.to_numpy(), args.x, args.y, args.size)
    fused_result = spatial_cv(np.concatenate((visual, rna_feature), axis=1), labels, boxes.center_x.to_numpy(), boxes.center_y.to_numpy(), args.x, args.y, args.size)
    delta_f1 = fused_result["macro_f1_mean"] - visual_result["macro_f1_mean"]
    delta_balanced = fused_result["balanced_accuracy_mean"] - visual_result["balanced_accuracy_mean"]
    passed = delta_f1 >= 0.02 and delta_balanced >= 0.02
    boxes["molecular_cluster"] = labels
    boxes["rna_transcript_count"] = count.sum(axis=1).to_numpy(dtype=int)
    boxes.to_csv(output / "cell_summary.csv", index=False)
    save_overlay(output / "molecular_cluster_overlay.png", crop, boxes, labels, args.x, args.y)
    metrics = {
        "purpose": "RNA feature evidence gate; not a final FewClick segmentation evaluation",
        "data_root": str(args.data_root), "window": {"x": args.x, "y": args.y, "size": args.size},
        "quality_filter": {"qv_threshold": args.qv_threshold, "min_transcripts": args.min_transcripts},
        "retained_cells": int(len(boxes)), "retained_transcripts": int(len(rna[rna.cell_id.isin(boxes.cell_id)])),
        "rna_feature_genes": len(rna_feature_genes), "label_genes": len(label_genes),
        "rna_feature_dim": int(rna_feature.shape[1]), "molecular_label_dim": int(molecular.shape[1]),
        "clusters": int(args.clusters), "visual_only": visual_result, "visual_plus_rna": fused_result,
        "delta_macro_f1": float(delta_f1), "delta_balanced_accuracy": float(delta_balanced),
        "acceptance": "delta macro-F1 and balanced accuracy >= 0.02", "passed_preliminary_gate": bool(passed),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# COAD Xenium RNA feature pilot\n\n"
        f"Preliminary gate: **{'passed' if passed else 'not passed'}**. This compares spatial-block validation of handcrafted H&E/geometry features with the same features plus an independent RNA feature view. It does not measure final FewClick segmentation quality.\n\n"
        f"- retained cells: {len(boxes)}\n- visual macro-F1: {visual_result['macro_f1_mean']:.4f}\n- fused macro-F1: {fused_result['macro_f1_mean']:.4f}\n- delta macro-F1: {delta_f1:.4f}\n- visual balanced accuracy: {visual_result['balanced_accuracy_mean']:.4f}\n- fused balanced accuracy: {fused_result['balanced_accuracy_mean']:.4f}\n- delta balanced accuracy: {delta_balanced:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "output": str(output), **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
