#!/usr/bin/env python3
"""Empirically calibrate Phikon-v2 H&E context size when WSI mpp is unknown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SCRIPT_DIR), str(SRC_ROOT)]
from phikon_v2 import PhikonV2Encoder  # noqa: E402
from run_phikon_backbone_probe import annotated_cells  # noqa: E402
from run_xenium_rna_pilot import spatial_cv  # noqa: E402
from run_xenium_xcell_baseline import read_polygons  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xcell-artifact", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806"))
    p.add_argument("--annotations", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/annotations.csv"))
    p.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    p.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crop-sizes", type=int, nargs="+", default=[64, 96, 128, 192, 256, 384])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shuffle-seed", type=int, default=20260907)
    return p.parse_args()


def pad_crop(array: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    half = size // 2; h, w = array.shape[:2]
    value = array[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
    padding = ((max(0, half-cy), max(0, cy+half-h)), (max(0, half-cx), max(0, cx+half-w)))
    return np.pad(value, padding + (() if array.ndim == 2 else ((0, 0),)), mode="edge")


def crops_for_size(image: np.ndarray, cells: pd.DataFrame, polygons: dict[str, np.ndarray], size: int) -> list[Image.Image]:
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    ids = {cell_id: index + 1 for index, cell_id in enumerate(cells.cell_id)}
    for cell_id, label in ids.items(): cv2.fillPoly(mask, [polygons[cell_id]], label)
    sigma = max(2.0, size * 12.0 / 128.0)
    crops = []
    for row in cells.itertuples(index=False):
        cy, cx, label = int(round(row.center_y - 2000)), int(round(row.center_x - 1000)), ids[row.cell_id]
        local_image, local_mask = pad_crop(image, cy, cx, size), pad_crop(mask, cy, cx, size) == label
        blur = ndimage.gaussian_filter(local_mask.astype(np.float32), sigma=sigma, mode="nearest")
        attended = local_image.astype(np.float32) / 255.0 * (0.35 + 0.65 * blur / (blur.max() + 1e-6))[..., None]
        crops.append(Image.fromarray((np.clip(attended, 0, 1) * 255).astype(np.uint8), mode="RGB"))
    return crops


def main() -> None:
    a = parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    if any(size < 32 or size % 2 for size in a.crop_sizes): raise ValueError("crop sizes must be even and >=32")
    a.output_dir.mkdir(parents=True)
    device = torch.device(a.device)
    summary = pd.read_csv(a.xcell_artifact / "cell_summary.csv", dtype={"cell_id": str})
    annotations = pd.read_csv(a.annotations, dtype={"cell_id": str})
    cells = annotated_cells(summary, annotations, 30)
    vips = pyvips.Image.new_from_file(str(a.data_root / "HE.tif"), access="sequential").crop(1000, 2000, 1500, 1500)
    image = np.ndarray(buffer=vips.write_to_memory(), dtype=np.uint8, shape=(1500, 1500, vips.bands))[:, :, :3]
    polygons = read_polygons(a.data_root / "segmentation_mask/cell_boundaries.csv", set(cells.cell_id), 1000, 2000)
    model = PhikonV2Encoder(a.model_dir, device=device, dtype=torch.float16 if device.type == "cuda" else None); model.freeze_backbone()
    labels, x, y = cells.annotation.to_numpy(), cells.center_x.to_numpy(), cells.center_y.to_numpy()
    results = []
    for size in a.crop_sizes:
        features = []
        crops = crops_for_size(image, cells, polygons, size)
        for start in range(0, len(crops), a.batch_size): features.append(model.encode_pil(crops[start:start+a.batch_size], device).float().cpu())
        feature = torch.cat(features).numpy()
        paired = spatial_cv(feature, labels, x, y, 1000, 2000, 1500)
        shuffled = spatial_cv(feature[np.random.default_rng(a.shuffle_seed).permutation(len(feature))], labels, x, y, 1000, 2000, 1500)
        np.save(a.output_dir / f"phikon_1024_crop{size}.npy", feature)
        results.append({"crop_px": size, "feature_shape": list(feature.shape), "paired": paired, "shuffled": shuffled, "paired_minus_shuffled_macro_f1": paired["macro_f1_mean"]-shuffled["macro_f1_mean"], "paired_minus_shuffled_balanced_accuracy": paired["balanced_accuracy_mean"]-shuffled["balanced_accuracy_mean"]})
    metrics = {"purpose": "empirical H&E context-scale calibration; not a physical mpp claim", "cells": len(cells), "classes": sorted(set(labels)), "crop_sizes_px": a.crop_sizes, "results": results, "selection_rule": "choose only a crop with paired-minus-shuffled positive and reproducible; then verify across sections"}
    cells.to_csv(a.output_dir / "cells.csv", index=False)
    (a.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event":"complete", "output":str(a.output_dir), **metrics}, ensure_ascii=False))


if __name__ == "__main__": main()
