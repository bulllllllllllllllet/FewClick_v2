#!/usr/bin/env python3
"""Audit a reproducible CosMx global-pixel to H&E-pixel coordinate transform."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyvips
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/cosmx_coordinate_audit"))
    parser.add_argument("--timestamp")
    parser.add_argument("--thumbnail-width", type=int, default=512)
    return parser.parse_args()


def global_bounds(path: Path) -> tuple[float, float, float, float]:
    xmin = ymin = float("inf"); xmax = ymax = float("-inf")
    for chunk in pd.read_csv(path, usecols=["x_global_px", "y_global_px"], chunksize=1_000_000):
        xmin, xmax = min(xmin, float(chunk.x_global_px.min())), max(xmax, float(chunk.x_global_px.max()))
        ymin, ymax = min(ymin, float(chunk.y_global_px.min())), max(ymax, float(chunk.y_global_px.max()))
    return xmin, ymin, xmax, ymax


def thumbnail(image: pyvips.Image, width: int) -> np.ndarray:
    value = image.resize(width / image.width)
    array = np.ndarray(buffer=value.write_to_memory(), dtype=np.uint8, shape=(value.height, value.width, value.bands))
    return array


def masks_and_overlay(he: pyvips.Image, dapi: pyvips.Image, width: int, output: Path, cohort: str) -> tuple[float, float]:
    he_image = thumbnail(he, width)[:, :, :3]
    dapi_image = thumbnail(dapi, width)[:, :, 0]
    he_gray = cv2.cvtColor(he_image, cv2.COLOR_RGB2GRAY)
    _, he_mask = cv2.threshold(he_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, dapi_mask = cv2.threshold(dapi_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    he_mask, dapi_mask = he_mask > 0, dapi_mask > 0
    if he_mask.shape != dapi_mask.shape:
        raise RuntimeError(f"thumbnail shape mismatch: {he_mask.shape} vs {dapi_mask.shape}")
    intersection = int((he_mask & dapi_mask).sum()); union = int((he_mask | dapi_mask).sum())
    dice = 2 * intersection / max(1, int(he_mask.sum() + dapi_mask.sum()))
    iou = intersection / max(1, union)
    overlay = he_image.copy()
    overlay[he_mask] = (0.65 * overlay[he_mask] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    overlay[dapi_mask] = (0.65 * overlay[dapi_mask] + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
    Image.fromarray(he_image).save(output / f"{cohort}_he_thumbnail.png")
    Image.fromarray(dapi_image).save(output / f"{cohort}_dapi_thumbnail.png")
    Image.fromarray(overlay).save(output / f"{cohort}_tissue_mask_overlay.png")
    return float(dice), float(iou)


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"cosmx_he_transform_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cells = pd.read_csv(args.manifest_dir / "sampled_cells.csv", dtype={"cell_id": str})
    rows = []
    for cohort, part in cells[cells.platform.eq("CosMx")].groupby("cohort_id", sort=True):
        root = Path(part.he_path.iloc[0]).parent
        he = pyvips.Image.new_from_file(str(root / "HE.tif"), access="sequential")
        dapi = pyvips.Image.new_from_file(str(root / "DAPI.tif"), access="sequential")
        xmin, ymin, xmax, ymax = global_bounds(root / "segmentation_mask/polygons.csv")
        global_width, global_height = xmax - xmin, ymax - ymin
        scale_x, scale_y = dapi.width / he.width, dapi.height / he.height
        span_error_x = abs(global_width - dapi.width) / dapi.width
        span_error_y = abs(global_height - dapi.height) / dapi.height
        converted = part.copy()
        converted["he_x"] = (converted.x - xmin) / scale_x
        converted["he_y"] = (converted.y - ymin) / scale_y
        inside = (converted.he_x >= 0) & (converted.he_x < he.width) & (converted.he_y >= 0) & (converted.he_y < he.height)
        converted.to_csv(output / f"{cohort}_converted_sampled_cells.csv", index=False)
        dice, iou = masks_and_overlay(he, dapi, args.thumbnail_width, output, cohort)
        rows.append({"cohort_id": cohort, "he_size": [he.width, he.height], "dapi_size": [dapi.width, dapi.height], "global_bounds": [xmin, ymin, xmax, ymax], "global_span": [global_width, global_height], "scale_dapi_per_he": [scale_x, scale_y], "transform": {"x_he": "(x_global - xmin) / scale_x", "y_he": "(y_global - ymin) / scale_y"}, "span_relative_error": [span_error_x, span_error_y], "sampled_cells": len(converted), "mapped_inside_he": int(inside.sum()), "tissue_mask_dice": dice, "tissue_mask_iou": iou, "contract_pass": bool(span_error_x <= .01 and span_error_y <= .01 and inside.all())})
    report = {"purpose": "audit inferred global-pixel to H&E-pixel transform; tissue mask overlap is auxiliary, not registration proof", "manifest_dir": str(args.manifest_dir), "cohorts": rows, "all_contract_pass": all(row["contract_pass"] for row in rows)}
    (output / "transform_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "all_contract_pass": report["all_contract_pass"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
