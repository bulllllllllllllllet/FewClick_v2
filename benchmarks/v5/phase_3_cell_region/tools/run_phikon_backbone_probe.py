#!/usr/bin/env python3
"""Compare frozen CTransPath and Phikon-v2 on matched annotated H&E cell crops.

This is the A/B/C backbone gate from the v5 plan: A=CTransPath, B=Phikon-v2,
C=Phikon-v2 feature rows shuffled across cells.  It is a bounded COAD section
implementation gate, not a cross-cohort final evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "benchmarks/v5/phase_3_cell_region/src"
for root in (SCRIPT_DIR, SRC_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
from phikon_v2 import PhikonV2Encoder  # noqa: E402
from run_xenium_rna_pilot import spatial_cv  # noqa: E402
from run_xenium_xcell_baseline import (  # noqa: E402
    crop_mask, crop_region, load_ctranspath, read_polygons, soft_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcell-artifact", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806"))
    parser.add_argument("--annotations", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/annotations.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    parser.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    parser.add_argument("--ctranspath-checkpoint", type=Path, default=Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/backbone_probe"))
    parser.add_argument("--timestamp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=20260907)
    parser.add_argument("--min-class-total", type=int, default=30)
    return parser.parse_args()


def annotated_cells(summary: pd.DataFrame, annotations: pd.DataFrame, minimum: int) -> pd.DataFrame:
    cells = summary.merge(annotations, on="cell_id", how="left", validate="one_to_one")
    eligible = cells.annotation.value_counts()
    eligible = eligible[eligible >= minimum].index
    cells = cells[cells.annotation.isin(eligible)].copy()
    block_counts = pd.crosstab(cells.annotation, cells.region).reindex(columns=range(4), fill_value=0)
    good = block_counts.index[(block_counts > 0).all(axis=1)]
    cells = cells[cells.annotation.isin(good)].copy().reset_index(drop=True)
    if len(good) < 2 or len(cells) < 80:
        raise RuntimeError(f"insufficient spatially distributed annotation: classes={list(good)}, cells={len(cells)}")
    return cells


def build_attended_crops(image: np.ndarray, cells: pd.DataFrame, polygons: dict[str, np.ndarray], x0: int, y0: int) -> list[Image.Image]:
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    local_ids = {cell_id: index + 1 for index, cell_id in enumerate(cells.cell_id)}
    for cell_id, label in local_ids.items():
        cv2.fillPoly(mask, [polygons[cell_id]], label)
    crops: list[Image.Image] = []
    for row in cells.itertuples(index=False):
        label = local_ids[row.cell_id]
        # `image` and `mask` are local to the selected 1500x1500 WSI window;
        # polygon vertices were already shifted in `read_polygons`.
        cy, cx = int(round(row.center_y - y0)), int(round(row.center_x - x0))
        attended = soft_attention(crop_region(image, cy, cx), crop_mask(mask, cy, cx) == label)
        crops.append(Image.fromarray((np.clip(attended, 0, 1) * 255).astype(np.uint8), mode="RGB"))
    return crops


@torch.inference_mode()
def encode_ctranspath(model: torch.nn.Module, crops: list[Image.Image], device: torch.device, batch_size: int) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    values = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack([preprocess(item) for item in crops[start:start + batch_size]]).to(device)
        values.append(model(batch).float().cpu())
    result = torch.cat(values).numpy()
    if result.shape != (len(crops), 768) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid CTransPath features: {result.shape}")
    return result


def encode_phikon(model: PhikonV2Encoder, crops: list[Image.Image], device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(crops), batch_size):
        values.append(model.encode_pil(crops[start:start + batch_size], device).float().cpu())
    result = torch.cat(values).numpy()
    if result.shape != (len(crops), 1024) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid Phikon-v2 features: {result.shape}")
    return result


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"phikon_backbone_probe_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    summary = pd.read_csv(args.xcell_artifact / "cell_summary.csv", dtype={"cell_id": str})
    annotations = pd.read_csv(args.annotations, dtype={"cell_id": str})
    cells = annotated_cells(summary, annotations, args.min_class_total)
    image_vips = pyvips.Image.new_from_file(str(args.data_root / "HE.tif"), access="sequential").crop(1000, 2000, 1500, 1500)
    image = np.ndarray(buffer=image_vips.write_to_memory(), dtype=np.uint8, shape=(1500, 1500, image_vips.bands))[:, :, :3]
    polygons = read_polygons(args.data_root / "segmentation_mask/cell_boundaries.csv", set(cells.cell_id), 1000, 2000)
    crops = build_attended_crops(image, cells, polygons, 1000, 2000)
    ctranspath = load_ctranspath(args.ctranspath_checkpoint, device)
    phikon = PhikonV2Encoder(args.model_dir, device=device, dtype=torch.float16 if device.type == "cuda" else None)
    phikon.freeze_backbone()
    ctranspath_feature = encode_ctranspath(ctranspath, crops, device, args.batch_size)
    phikon_feature = encode_phikon(phikon, crops, device, args.batch_size)
    permutation = np.random.default_rng(args.shuffle_seed).permutation(len(cells))
    labels = cells.annotation.to_numpy()
    x, y = cells.center_x.to_numpy(), cells.center_y.to_numpy()
    ctranspath_result = spatial_cv(ctranspath_feature, labels, x, y, 1000, 2000, 1500)
    phikon_result = spatial_cv(phikon_feature, labels, x, y, 1000, 2000, 1500)
    shuffled_result = spatial_cv(phikon_feature[permutation], labels, x, y, 1000, 2000, 1500)
    cells.to_csv(output / "cells.csv", index=False)
    np.save(output / "ctranspath_768.npy", ctranspath_feature)
    np.save(output / "phikon_v2_1024.npy", phikon_feature)
    metrics = {
        "purpose": "frozen A/B/C backbone gate on one COAD section; not cross-cohort or FewClick segmentation evaluation",
        "retained_cells": int(len(cells)), "classes": sorted(cells.annotation.unique().tolist()),
        "class_by_spatial_block": pd.crosstab(cells.annotation, cells.region).to_dict(),
        "crop_contract": "same 128x128 soft-attended H&E crop -> 224x224 backbone preprocessing", "device": str(device),
        "ctranspath_frozen": ctranspath_result, "phikon_v2_frozen": phikon_result,
        "phikon_v2_h_and_e_shuffled": shuffled_result, "shuffle_seed": args.shuffle_seed,
        "phikon_minus_ctranspath_macro_f1": phikon_result["macro_f1_mean"] - ctranspath_result["macro_f1_mean"],
        "phikon_minus_shuffled_macro_f1": phikon_result["macro_f1_mean"] - shuffled_result["macro_f1_mean"],
        "acceptance": "implementation gate only: feature shapes finite; shuffled H&E must be lower than paired Phikon-v2",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
