#!/usr/bin/env python3
"""Validate manifest coordinates with real dual-view H&E Phikon-v2 extraction.

Only manifest rows explicitly marked ``ready_for_he_crop_smoke`` are processed.
This is a coordinate/data-loader gate, not a cell-classification result.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "benchmarks/v5/phase_3_cell_region/src"
sys.path.insert(0, str(SRC_ROOT))
from phikon_v2 import PhikonV2Encoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke"))
    parser.add_argument("--timestamp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--views", type=int, nargs="+", default=[256, 384])
    return parser.parse_args()


def crop_at(image: pyvips.Image, x: float, y: float, side: int) -> tuple[Image.Image, bool]:
    """Read a full crop, shifting it only when a valid center is close to an edge."""
    left = int(round(x - side / 2))
    top = int(round(y - side / 2))
    adjusted_left = min(max(left, 0), image.width - side)
    adjusted_top = min(max(top, 0), image.height - side)
    shifted = adjusted_left != left or adjusted_top != top
    piece = image.crop(adjusted_left, adjusted_top, side, side)
    array = np.ndarray(buffer=piece.write_to_memory(), dtype=np.uint8, shape=(side, side, piece.bands))[:, :, :3]
    return Image.fromarray(array, mode="RGB"), shifted


def main() -> None:
    args = parse_args()
    if not args.manifest_dir.is_dir():
        raise FileNotFoundError(args.manifest_dir)
    if any(side <= 0 for side in args.views) or args.batch_size <= 0:
        raise ValueError("views and batch size must be positive")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"multisection_cell_smoke_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    rows = pd.read_csv(args.manifest_dir / "sampled_cells.csv")
    eligible_mask = rows.coordinate_status.astype(str).str.startswith("ready_for_he_crop_smoke")
    eligible = rows[eligible_mask].copy().reset_index(drop=True)
    excluded = rows[~eligible_mask].copy()
    if eligible.empty:
        raise RuntimeError("manifest has no H&E crop-smoke eligible cells")
    encoder = PhikonV2Encoder(args.model_dir, device=device, dtype=torch.float16 if device.type == "cuda" else None)
    encoder.freeze_backbone()
    section_metrics: list[dict] = []
    feature_views: dict[int, list[np.ndarray]] = defaultdict(list)
    feature_ids: list[str] = []
    for cohort, part in eligible.groupby("cohort_id", sort=True):
        image = pyvips.Image.new_from_file(str(part.he_path.iloc[0]), access="random")
        centers_inside = ((part.x >= 0) & (part.x < image.width) & (part.y >= 0) & (part.y < image.height))
        if not bool(centers_inside.all()):
            raise RuntimeError(f"{cohort}: {(~centers_inside).sum()} sampled centers outside H&E raster")
        if min(args.views) > min(image.width, image.height):
            raise RuntimeError(f"{cohort}: H&E too small for requested crop")
        crops = {side: [] for side in args.views}
        shifts = {side: 0 for side in args.views}
        for row in part.itertuples(index=False):
            for side in args.views:
                crop, shifted = crop_at(image, row.x, row.y, side)
                crops[side].append(crop)
                shifts[side] += int(shifted)
        section_features = {side: [] for side in args.views}
        with torch.inference_mode():
            for side in args.views:
                for start in range(0, len(part), args.batch_size):
                    values = encoder.encode_pil(crops[side][start:start + args.batch_size], device).float().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise RuntimeError(f"{cohort}: non-finite Phikon values")
                    section_features[side].append(values)
                feature_views[side].append(np.concatenate(section_features[side], axis=0))
        feature_ids.extend(part.cell_id.astype(str).tolist())
        section_metrics.append({"cohort_id": cohort, "he_path": str(part.he_path.iloc[0]), "image_width": image.width, "image_height": image.height, "sampled_cells": len(part), "center_inside_count": int(centers_inside.sum()), "crop_edge_shift_count": {str(side): shifts[side] for side in args.views}, "feature_shape_by_view": {str(side): [len(part), 1024] for side in args.views}, "finite": True})
    payload = {"cell_id": np.asarray(feature_ids)}
    for side in args.views:
        payload[f"phikon_cls_{side}px"] = np.concatenate(feature_views[side], axis=0)
    np.savez_compressed(output / "phikon_dualview_features.npz", **payload)
    eligible.to_csv(output / "smoked_cells.csv", index=False)
    excluded.to_csv(output / "excluded_coordinate_transform_cells.csv", index=False)
    report = {"purpose": "real H&E dual-view crop and Phikon-v2 data-loader coordinate smoke; no RNA model input and no training metric", "manifest_dir": str(args.manifest_dir), "model_dir": str(args.model_dir), "device": str(device), "views_px": args.views, "eligible_cells": len(eligible), "excluded_requires_coordinate_transform": len(excluded), "sections": section_metrics, "all_features_finite": True}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "eligible_cells": len(eligible), "sections": len(section_metrics), "all_features_finite": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
