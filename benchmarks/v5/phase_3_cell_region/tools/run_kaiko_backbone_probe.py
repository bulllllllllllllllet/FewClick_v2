#!/usr/bin/env python3
"""Compare frozen CTransPath, Phikon-v2, and public Kaiko on matched H&E crops."""
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
from safetensors.torch import load_file
from torchvision import transforms

HERE = Path(__file__).resolve().parent
PHIKON_OVERLAY = Path("/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2")
KAIKO_OVERLAY = Path("/nfs-medical3/zyh/fewclick_v2/vendor/kaiko_timm_0907_194700")
# This must precede imports from historical CTransPath helpers: they import timm.
for path in (KAIKO_OVERLAY, PHIKON_OVERLAY, HERE, Path(__file__).resolve().parents[4], Path(__file__).resolve().parents[4] / "benchmarks/v5/phase_3_cell_region/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from phikon_v2 import PhikonV2Encoder  # noqa: E402
from run_phikon_backbone_probe import annotated_cells, build_attended_crops, encode_phikon  # noqa: E402
from run_xenium_rna_pilot import spatial_cv  # noqa: E402
from run_xenium_xcell_baseline import read_polygons  # noqa: E402

ARCHITECTURE = "vit_large_patch14_reg4_dinov2"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xcell-artifact", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806"))
    p.add_argument("--annotations", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/annotations.csv"))
    p.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    p.add_argument("--phikon-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    p.add_argument("--kaiko-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/kaiko_vit_large_patch14_reg4/kaiko_0907_194700"))
    p.add_argument("--timm-overlay", type=Path, default=KAIKO_OVERLAY)
    p.add_argument("--ctranspath-checkpoint", type=Path, default=Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"))
    p.add_argument("--ctranspath-probe", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/backbone_probe/phikon_backbone_probe_0907_024957"))
    p.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/backbone_probe"))
    p.add_argument("--timestamp", default=None); p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--shuffle-seed", type=int, default=20260907)
    return p.parse_args()


def load_kaiko(overlay: Path, directory: Path, device: torch.device):
    sys.path.insert(0, str(overlay)); import timm
    if ARCHITECTURE not in timm.list_models("*vit_large_patch14*"):
        raise RuntimeError(f"timm {timm.__version__} lacks {ARCHITECTURE}")
    model = timm.create_model(ARCHITECTURE, pretrained=False, img_size=518, num_classes=0, global_pool="token")
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device="cpu"), strict=True)
    return model.to(device).eval(), timm.__version__


@torch.inference_mode()
def encode_kaiko(model, crops, device, batch_size):
    prep = transforms.Compose([transforms.ToTensor(), transforms.Resize((518, 518)), transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    rows = []
    for start in range(0, len(crops), batch_size):
        rows.append(model(torch.stack([prep(x) for x in crops[start:start + batch_size]]).to(device)).float().cpu())
    result = torch.cat(rows).numpy()
    if result.shape != (len(crops), 1024) or not np.isfinite(result).all(): raise RuntimeError(f"invalid Kaiko features: {result.shape}")
    return result


def main() -> None:
    a = args(); stamp = a.timestamp or datetime.now().strftime("%m%d_%H%M%S"); output = a.output_root / f"kaiko_backbone_probe_{stamp}"
    if output.exists(): raise FileExistsError(output)
    if not PHIKON_OVERLAY.is_dir(): raise FileNotFoundError(f"missing required Phikon v5 dependency overlay: {PHIKON_OVERLAY}")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    summary = pd.read_csv(a.xcell_artifact / "cell_summary.csv", dtype={"cell_id": str}); ann = pd.read_csv(a.annotations, dtype={"cell_id": str})
    cells = annotated_cells(summary, ann, 30)
    vips = pyvips.Image.new_from_file(str(a.data_root / "HE.tif"), access="sequential").crop(1000, 2000, 1500, 1500)
    image = np.ndarray(buffer=vips.write_to_memory(), dtype=np.uint8, shape=(1500, 1500, vips.bands))[:, :, :3]
    polygons = read_polygons(a.data_root / "segmentation_mask/cell_boundaries.csv", set(cells.cell_id), 1000, 2000)
    crops = build_attended_crops(image, cells, polygons, 1000, 2000)
    prior_cells = pd.read_csv(a.ctranspath_probe / "cells.csv", dtype={"cell_id": str})
    if not np.array_equal(prior_cells.cell_id.to_numpy(dtype=str), cells.cell_id.to_numpy(dtype=str)):
        raise RuntimeError("historical CTransPath feature cell IDs do not match this Kaiko probe")
    ctrans = np.load(a.ctranspath_probe / "ctranspath_768.npy")
    if ctrans.shape != (len(cells), 768) or not np.isfinite(ctrans).all():
        raise RuntimeError(f"invalid historical CTransPath features: {ctrans.shape}")
    phikon_model = PhikonV2Encoder(a.phikon_dir, device=device, dtype=torch.float16); phikon_model.freeze_backbone()
    phikon = encode_phikon(phikon_model, crops, device, a.batch_size)
    del phikon_model; torch.cuda.empty_cache()
    kaiko_model, timm_version = load_kaiko(a.timm_overlay, a.kaiko_dir, device); kaiko = encode_kaiko(kaiko_model, crops, device, a.batch_size)
    labels, x, y = cells.annotation.to_numpy(), cells.center_x.to_numpy(), cells.center_y.to_numpy(); perm = np.random.default_rng(a.shuffle_seed).permutation(len(cells))
    result = {"ctranspath_frozen": spatial_cv(ctrans, labels, x, y, 1000, 2000, 1500), "phikon_v2_frozen": spatial_cv(phikon, labels, x, y, 1000, 2000, 1500), "kaiko_frozen": spatial_cv(kaiko, labels, x, y, 1000, 2000, 1500), "kaiko_h_and_e_shuffled": spatial_cv(kaiko[perm], labels, x, y, 1000, 2000, 1500)}
    output.mkdir(parents=True); cells.to_csv(output / "cells.csv", index=False); np.save(output / "kaiko_1024.npy", kaiko)
    report = {"purpose": "same-crop frozen CTransPath/Phikon/Kaiko gate; not FewClick segmentation", "retained_cells": len(cells), "classes": sorted(set(labels)), "crop_contract": "same 128px soft-attention RGB crop; per-backbone resize/normalization", "ctranspath_feature_source": str(a.ctranspath_probe), "kaiko_input_px": 518, "timm_version": timm_version, "results": result, "kaiko_minus_ctranspath_macro_f1": result["kaiko_frozen"]["macro_f1_mean"] - result["ctranspath_frozen"]["macro_f1_mean"], "kaiko_minus_shuffle_macro_f1": result["kaiko_frozen"]["macro_f1_mean"] - result["kaiko_h_and_e_shuffled"]["macro_f1_mean"]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps({"event": "complete", "output": str(output), **report}, ensure_ascii=False))


if __name__ == "__main__": main()
