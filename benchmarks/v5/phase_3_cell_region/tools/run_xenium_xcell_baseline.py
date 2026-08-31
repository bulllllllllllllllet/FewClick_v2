#!/usr/bin/env python3
"""Evaluate the frozen v4 XCellFormer visual encoder on the COAD RNA pilot."""
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
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_xenium_rna_pilot import (  # noqa: E402
    load_transcripts, normalized_counts, spatial_cv, split_gene_views, svd_features, visual_features,
)
from XCellFormer import XCellFormer  # noqa: E402
from module.TransPath.ctran import ConvStem, ctranspath  # noqa: E402

CROP_SIZE, HALF_CROP, GAUSS_SIGMA, BG_WEIGHT, FG_WEIGHT = 128, 64, 12.0, 0.35, 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_rna_feature_pilot_0831_023421/cell_summary.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot"))
    parser.add_argument("--timestamp")
    parser.add_argument("--x", type=int, default=1000)
    parser.add_argument("--y", type=int, default=2000)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--max-cells-per-region", type=int, default=255)
    parser.add_argument("--cell-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rna-shuffle-seed", type=int, default=20260831)
    parser.add_argument("--ctranspath-checkpoint", type=Path, default=Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"))
    parser.add_argument("--xcell-checkpoint", type=Path, default=Path("/home/zhaoyh/FewClick/module/checkpoint/he_model_best.pth"))
    return parser.parse_args()


def crop_region(image: np.ndarray, cy: int, cx: int) -> np.ndarray:
    h, w = image.shape[:2]
    region = image[max(0, cy - HALF_CROP):min(h, cy + HALF_CROP), max(0, cx - HALF_CROP):min(w, cx + HALF_CROP)]
    return np.pad(region, ((max(0, HALF_CROP - cy), max(0, cy + HALF_CROP - h)), (max(0, HALF_CROP - cx), max(0, cx + HALF_CROP - w)), (0, 0)), mode="edge")


def crop_mask(mask: np.ndarray, cy: int, cx: int) -> np.ndarray:
    h, w = mask.shape
    region = mask[max(0, cy - HALF_CROP):min(h, cy + HALF_CROP), max(0, cx - HALF_CROP):min(w, cx + HALF_CROP)]
    return np.pad(region, ((max(0, HALF_CROP - cy), max(0, cy + HALF_CROP - h)), (max(0, HALF_CROP - cx), max(0, cx + HALF_CROP - w))), mode="edge")


def soft_attention(image: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if not selected.any():
        return image.astype(np.float32) / 255.0
    blur = ndimage.gaussian_filter(selected.astype(np.float32), sigma=GAUSS_SIGMA, mode="nearest")
    return image.astype(np.float32) / 255.0 * (BG_WEIGHT + FG_WEIGHT * blur / (blur.max() + 1e-6))[..., None]


def load_ctranspath(path: Path, device: torch.device) -> torch.nn.Module:
    model = ctranspath()
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    adapted = {}
    for key, value in state.items():
        if "layers.0.downsample" in key:
            key = key.replace("layers.0.downsample", "layers.1.downsample")
        elif "layers.1.downsample" in key:
            key = key.replace("layers.1.downsample", "layers.2.downsample")
        elif "layers.2.downsample" in key:
            key = key.replace("layers.2.downsample", "layers.3.downsample")
        adapted[key] = value
    model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=model.embed_dim)
    model.load_state_dict(adapted, strict=False)

    class GlobalPoolHead(torch.nn.Module):
        def __init__(self): super().__init__(); self.pool = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            if x.dim() == 4: x = self.pool(x.permute(0, 3, 1, 2))
            return torch.flatten(x, 1)
    model.head = GlobalPoolHead()
    return model.to(device).eval()


def load_xcell(path: Path, device: torch.device) -> XCellFormer:
    model = XCellFormer(768, 512, 8, 4, output_dim=64, max_cells=255, use_large_vit=False, device=str(device))
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    return model.to(device).eval()


def read_polygons(path: Path, ids: set[str], x0: int, y0: int) -> dict[str, np.ndarray]:
    chunks = []
    for chunk in pd.read_csv(path, usecols=["cell_id", "vertex_x", "vertex_y"], chunksize=500_000):
        chunk["cell_id"] = chunk.cell_id.astype(str)
        subset = chunk[chunk.cell_id.isin(ids)]
        if not subset.empty: chunks.append(subset)
    values = pd.concat(chunks, ignore_index=True)
    result = {}
    for cell_id, group in values.groupby("cell_id", sort=False):
        result[cell_id] = np.round(group[["vertex_x", "vertex_y"]].to_numpy() - np.array([x0, y0])).astype(np.int32)
    missing = ids - set(result)
    if missing: raise RuntimeError(f"missing boundary polygons for {len(missing)} selected cells")
    return result


def select_regions(cells: pd.DataFrame, x0: int, y0: int, size: int, maximum: int) -> pd.DataFrame:
    cells = cells.copy()
    cells["region"] = (cells.center_x >= x0 + size / 2).astype(int) + 2 * (cells.center_y >= y0 + size / 2).astype(int)
    selected = []
    for region in range(4):
        value = cells[cells.region == region].sort_values(["center_y", "center_x", "cell_id"]).head(maximum)
        if len(value) < 20: raise RuntimeError(f"region {region} has only {len(value)} cells")
        selected.append(value)
    return pd.concat(selected, ignore_index=True)


def extract_xcell(image: np.ndarray, cells: pd.DataFrame, polygons: dict[str, np.ndarray], x0: int, y0: int, ctp: torch.nn.Module, xcell: XCellFormer, device: torch.device, batch_size: int, maximum: int) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    local_ids = {cell_id: index + 1 for index, cell_id in enumerate(cells.cell_id)}
    for cell_id, label in local_ids.items(): cv2.fillPoly(mask, [polygons[cell_id]], label)
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), interpolation=Image.BILINEAR), transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    output = np.zeros((len(cells), 64), dtype=np.float32)
    for region in range(4):
        region_cells = cells[cells.region == region]
        labels = np.array([local_ids[cell_id] for cell_id in region_cells.cell_id], dtype=np.int32)
        centers = ndimage.center_of_mass(np.ones_like(mask), labels=mask, index=labels)
        ordered = sorted(zip(labels, centers), key=lambda pair: (pair[1][0], pair[1][1]))
        tensors = []
        for label, (cy, cx) in ordered:
            attended = soft_attention(crop_region(image, int(cy), int(cx)), crop_mask(mask, int(cy), int(cx)) == label)
            tensors.append(preprocess(Image.fromarray((np.clip(attended, 0, 1) * 255).astype(np.uint8))))
        raw = []
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size): raw.append(ctp(torch.stack(tensors[start:start + batch_size]).to(device)).float().cpu())
            values = torch.cat(raw).numpy()
            padded = np.zeros((1, maximum, 768), dtype=np.float32); valid = np.zeros((1, maximum), dtype=np.float32)
            padded[0, :len(values)] = values; valid[0, :len(values)] = 1
            _, reg, _, _ = xcell(None, torch.from_numpy(padded).to(device), torch.from_numpy(valid).to(device))
        ordered_labels = [label for label, _ in ordered]
        for index, label in enumerate(ordered_labels): output[label - 1] = reg[0, index].float().cpu().numpy()
        print(json.dumps({"event": "region_complete", "region": region, "cells": len(values)}), flush=True)
    if not np.isfinite(output).all(): raise RuntimeError("non-finite XCell features")
    return output


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"xenium_xcell_baseline_{stamp}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    source = pd.read_csv(args.pilot_summary); source.cell_id = source.cell_id.astype(str)
    cells = select_regions(source, args.x, args.y, args.size, args.max_cells_per_region)
    transcript_path = args.data_root / "transcripts/transcripts.parquet"
    rna = load_transcripts(transcript_path, set(source.cell_id), args.x, args.y, args.size, 20.0)
    count = normalized_counts(rna, source.cell_id.tolist(), 30)
    if not count.index.equals(pd.Index(source.cell_id)): raise RuntimeError("pilot RNA cell ordering changed")
    feature_genes, _ = split_gene_views(count.columns.tolist()); rna_all = svd_features(count[feature_genes].to_numpy())
    source_index = pd.Index(source.cell_id)
    rna_positions = source_index.get_indexer(cells.cell_id)
    if (rna_positions < 0).any(): raise RuntimeError("selected cell absent from pilot RNA order")
    rna_feature = rna_all[rna_positions]
    image_vips = pyvips.Image.new_from_file(str(args.data_root / "HE.tif"), access="sequential").crop(args.x, args.y, args.size, args.size)
    image = np.ndarray(buffer=image_vips.write_to_memory(), dtype=np.uint8, shape=(args.size, args.size, image_vips.bands))[:, :, :3]
    polygons = read_polygons(args.data_root / "segmentation_mask/cell_boundaries.csv", set(cells.cell_id), args.x, args.y)
    ctp, xcell = load_ctranspath(args.ctranspath_checkpoint, device), load_xcell(args.xcell_checkpoint, device)
    xcell_feature = extract_xcell(image, cells, polygons, args.x, args.y, ctp, xcell, device, args.cell_batch_size, args.max_cells_per_region)
    handcrafted, _ = visual_features(args.data_root / "HE.tif", cells, args.x, args.y, args.size)
    labels = cells.molecular_cluster.to_numpy()
    positional = (cells.center_x.to_numpy(), cells.center_y.to_numpy())
    base = spatial_cv(handcrafted, labels, *positional, args.x, args.y, args.size)
    xcell_result = spatial_cv(xcell_feature, labels, *positional, args.x, args.y, args.size)
    fused = spatial_cv(np.concatenate((xcell_feature, rna_feature), axis=1), labels, *positional, args.x, args.y, args.size)
    shuffled = spatial_cv(np.concatenate((xcell_feature, rna_feature[np.random.default_rng(args.rna_shuffle_seed).permutation(len(rna_feature))]), axis=1), labels, *positional, args.x, args.y, args.size)
    cells.to_csv(out / "cell_summary.csv", index=False)
    np.save(out / "xcell_reg64.npy", xcell_feature)
    metrics = {"purpose": "frozen v4 XCellFormer reg-64 visual baseline; not final FewClick segmentation", "pilot_summary": str(args.pilot_summary), "retained_cells": len(cells), "regions": cells.region.value_counts().sort_index().to_dict(), "feature": "soft-attention H&E -> CTransPath-768 -> XCellFormer reg-64", "handcrafted_visual": base, "xcell_visual": xcell_result, "xcell_plus_rna": fused, "xcell_plus_shuffled_rna": shuffled, "rna_gain_macro_f1": fused["macro_f1_mean"] - xcell_result["macro_f1_mean"], "rna_gain_balanced_accuracy": fused["balanced_accuracy_mean"] - xcell_result["balanced_accuracy_mean"], "real_minus_shuffled_macro_f1": fused["macro_f1_mean"] - shuffled["macro_f1_mean"], "real_minus_shuffled_balanced_accuracy": fused["balanced_accuracy_mean"] - shuffled["balanced_accuracy_mean"], "acceptance": "all reg-64 finite; real RNA exceeds shuffled RNA by >= 0.10 on both metrics"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "README.md").write_text("# XCellFormer visual baseline\\n\\nThis is a frozen-v4 encoder compatibility pilot, not a FewClick segmentation score. See `metrics.json`.\\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
