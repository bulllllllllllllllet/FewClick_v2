#!/usr/bin/env python3
"""Real GDPH H&E/GT/XCell forward-backward smoke, not a CRC accuracy experiment."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyvips  # Load image libraries before torch, as required by the existing loader.
import openslide
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v5.phase_3_cell_region.src.semantic_cell_region import SemanticCellToRegion
from benchmarks.v5.phase_3_cell_region.src.composition import physical_region_area
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import distribution_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp must be MMDD_HHMMSS")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use the FewClick artifact root")
    out = args.output_root / f"real_smoke_{args.timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260911)
    root = Path("/nfs-medical3/zyh/v4")
    p2cfg = load_config(root / "phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json")
    dataset = CellRegionDataset(root / "phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet",
                               root / "phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet",
                               root / "phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet",
                               p2cfg, "train")
    chosen = None
    for idx in range(min(32, len(dataset))):
        meta, count = dataset._feature(dataset.route[idx], dataset.region.rows[idx]["patch_id"])
        if 0 < count <= 255 and len(meta) == count:
            chosen = idx
            break
    if chosen is None:
        raise RuntimeError("no complete <=255-cell patch among first 32; no subsampling fallback")
    row = dataset.region.rows[chosen]
    batch = collate_cell_region([dataset[chosen]], 255)
    # Remove existing nuclear-label information even on this diagnostic path.
    batch["cells"][..., 3:10] = 0
    batch["cells"][..., 3] = 1
    print(json.dumps({"event": "loaded_real_patch", "patch_id": row["patch_id"], "cells": count}), flush=True)
    phase2 = DeepRegionEncoder(p2cfg, len(p2cfg["data"]["class_map"]), 64, 256)
    phase2.load_state_dict(torch.load(root / "phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth", map_location="cpu", weights_only=False)["model"])
    phase2.eval().requires_grad_(False)
    with torch.no_grad():
        region = phase2(batch["image"], return_full_assignment=False, return_tokens=True)
    assignment = region["assignment_low"]
    with openslide.OpenSlide(row["wsi_path"]) as slide:
        mpp_x, mpp_y = float(slide.properties["openslide.mpp-x"]), float(slide.properties["openslide.mpp-y"])
    grid_area = mpp_x * mpp_y * row["width_level0"] * row["height_level0"] / (assignment.shape[-1] * assignment.shape[-2])
    area = physical_region_area(assignment, grid_area)
    cells = batch["cells"]
    mass = F.grid_sample(assignment, (cells[..., :2] * 2 - 1).unsqueeze(1), align_corners=False, padding_mode="border").squeeze(2)
    inputs = {"region_tokens": region["region_tokens"], "cells": cells, "cell_valid": batch["cell_valid"],
              "assignment_at_cells": mass, "region_area_um2": area}
    model = SemanticCellToRegion(3, 12)
    baseline_path = root / "phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth"
    base_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    model.baseline.load_state_dict(base_payload["cell"], strict=True)
    model.tissue_head.load_state_dict(base_payload["head"], strict=True)
    old = CellToRegionAttention(256, 74)
    old.load_state_dict(base_payload["cell"], strict=True)
    with torch.no_grad():
        expected = old(inputs["region_tokens"], mass, cells, batch["cell_valid"])["fused_tokens"]
        initial_error = float((model(**inputs)["fused_tokens"] - expected).abs().max())
    if initial_error != 0:
        raise RuntimeError("initialization does not preserve baseline")
    mask = batch["mask"]
    full_assignment = F.interpolate(assignment, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    valid_pixels = mask != 255
    labels = F.one_hot(mask.clamp(0, 11), 12).float() * valid_pixels[..., None]
    votes = torch.einsum("bkhw,bhwc->bkc", full_assignment, labels)
    target = votes / votes.sum(-1, keepdim=True).clamp_min(1e-8)
    target_valid = votes.sum(-1) > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        result = model(**inputs)
        loss, supervised = distribution_loss(result["tissue_logits"], target, target_valid)
        if not supervised or not torch.isfinite(loss):
            raise RuntimeError("invalid real tissue loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        result = model(**inputs)
    torch.save({"schema_version": model.schema_version, "model": model.state_dict(), "diagnostic_only": True}, out / "checkpoint.pth")
    reload = SemanticCellToRegion(3, 12)
    reload.load_state_dict(torch.load(out / "checkpoint.pth", weights_only=True)["model"], strict=True)
    reload.eval()
    with torch.no_grad():
        reload_error = float((reload(**inputs)["fused_tokens"] - result["fused_tokens"]).abs().max())
    if reload_error != 0 or float(result["residual"].abs().sum()) == 0:
        raise RuntimeError("reload mismatch or residual did not train")
    torch.save({"patch_id": row["patch_id"], "region_tokens": result["fused_tokens"],
                "assignment_low": assignment, "diagnostic_only": True}, out / "region_cache.pth")
    report = {"status": "complete", "patch_id": row["patch_id"], "cells": count,
              "input_source": "GDPH real H&E/GT and cached XCell; not SRRSH CRC",
              "identity": "random frozen 3-class head for shape smoke only; no identity training/accuracy claim",
              "nuclei_class_input": "unknown; removed original labels",
              "mpp_xy": [mpp_x, mpp_y], "region_area_sum_um2": float(area.sum()),
              "initial_max_error": initial_error, "reload_max_error": reload_error,
              "real_tissue_loss_3steps": losses, "output_shape": list(result["fused_tokens"].shape),
              "phase4_6_evaluated": False, "final_segmentation_evaluated": False}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
