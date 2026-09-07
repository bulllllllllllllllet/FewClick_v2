#!/usr/bin/env python3
"""Build cohort-sharded, H&E-only dual-view Phikon features from a manifest."""
from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache"))
    parser.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    parser.add_argument("--timestamp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--views", type=int, nargs="+", default=[256, 384])
    parser.add_argument("--max-cells-per-cohort", type=int, help="bounded real-loader smoke only")
    return parser.parse_args()


def crop_at(image: pyvips.Image, x: float, y: float, side: int) -> Image.Image:
    left = min(max(int(round(x - side / 2)), 0), image.width - side)
    top = min(max(int(round(y - side / 2)), 0), image.height - side)
    piece = image.crop(left, top, side, side)
    array = np.ndarray(buffer=piece.write_to_memory(), dtype=np.uint8, shape=(side, side, piece.bands))[:, :, :3]
    return Image.fromarray(array, mode="RGB")


def write_index(output: Path, index: dict) -> None:
    (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or any(side <= 0 for side in args.views):
        raise ValueError("batch size and views must be positive")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"multisection_cell_feature_cache_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    cells = pd.read_csv(args.manifest_dir / "sampled_cells.csv", dtype={"cell_id": str})
    eligible = cells[cells.coordinate_status.astype(str).str.startswith("ready_for_he_crop_smoke")].copy()
    excluded = cells.loc[~cells.index.isin(eligible.index)]
    if len(eligible) != len(cells):
        raise RuntimeError(f"manifest still has {len(excluded)} non-eligible cells")
    output.mkdir(parents=True)
    index = {"purpose": "cohort-sharded dual-view H&E-only Phikon feature cache; RNA is not model input", "manifest_dir": str(args.manifest_dir), "model_dir": str(args.model_dir), "device": str(device), "views_px": args.views, "batch_size": args.batch_size, "max_cells_per_cohort": args.max_cells_per_cohort, "cohorts": {}}
    write_index(output, index)
    encoder = PhikonV2Encoder(args.model_dir, device=device, dtype=torch.float16 if device.type == "cuda" else None)
    encoder.freeze_backbone()
    for cohort, group in eligible.groupby("cohort_id", sort=True):
        part = group.copy()
        if args.max_cells_per_cohort is not None:
            part = part.iloc[:args.max_cells_per_cohort].copy()
        shard = output / cohort
        shard.mkdir()
        image = pyvips.Image.new_from_file(str(part.he_path.iloc[0]), access="random")
        inside = (part.x >= 0) & (part.x < image.width) & (part.y >= 0) & (part.y < image.height)
        if not bool(inside.all()):
            raise RuntimeError(f"{cohort}: sampled center outside H&E raster")
        values_by_view: dict[int, list[np.ndarray]] = {side: [] for side in args.views}
        with torch.inference_mode():
            for side in args.views:
                for start in range(0, len(part), args.batch_size):
                    batch = part.iloc[start:start + args.batch_size]
                    crops = [crop_at(image, row.x, row.y, side) for row in batch.itertuples(index=False)]
                    values = encoder.encode_pil(crops, device).float().cpu().numpy()
                    if values.shape != (len(batch), 1024) or not np.isfinite(values).all():
                        raise RuntimeError(f"{cohort}: invalid {side}px feature batch {values.shape}")
                    values_by_view[side].append(values)
        payload = {"cell_id": part.cell_id.to_numpy(dtype=str)}
        for side in args.views:
            payload[f"phikon_cls_{side}px"] = np.concatenate(values_by_view[side], axis=0)
        np.savez_compressed(shard / "phikon_dualview_features.npz", **payload)
        part.to_csv(shard / "cells.csv", index=False)
        completion = {"cohort_id": cohort, "cells": len(part), "he_path": str(part.he_path.iloc[0]), "image_size": [image.width, image.height], "views_px": args.views, "feature_shapes": {str(side): list(payload[f"phikon_cls_{side}px"].shape) for side in args.views}, "finite": True, "status": "complete"}
        (shard / "COMPLETED.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        index["cohorts"][cohort] = completion
        write_index(output, index)
        print(json.dumps({"event": "cohort_complete", "output": str(output), **completion}, ensure_ascii=False), flush=True)
    index["all_complete"] = True
    write_index(output, index)
    print(json.dumps({"event": "complete", "output": str(output), "cohorts": len(index["cohorts"]), "cells": sum(row["cells"] for row in index["cohorts"].values())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
