#!/usr/bin/env python3
"""Run a real frozen-Phase-2 CRC panel-2 region-evidence forward/backward smoke."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.region_footprint_contract import patch_pixel_grid_indices, project_assignment_to_grid, project_observed_targets
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss


COUNT_COLUMNS = ("panel2_cd4_t_count", "panel2_cd8_t_count", "panel2_treg_count", "panel2_b_cell_count", "panel2_cd163_macrophage_count")


def tensor_image(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.
    value = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return (value - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]


def targets_for_patch(rows):
    out = {}
    for _, row in rows.iterrows():
        key = (int(row.grid_x), int(row.grid_y))
        value = (row[list(COUNT_COLUMNS)].to_numpy(np.float32), float(row.panel2_observed_cells), float(row.panel2_coverage))
        if key in out and not np.array_equal(out[key][0], value[0]):
            raise ValueError(f"conflicting panel-2 target for grid {key}")
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--alignment-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0921_170938"))
    parser.add_argument("--phase2-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--min-token-grid-fraction", type=float, default=.80)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp must be MMDD_HHMMSS")
    out = args.output_root / f"crc_panel2_region_evidence_smoke_{args.timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    rows = pd.read_json(args.alignment_dir / "alignment_candidates.jsonl", lines=True)
    required = {"case_id", "patch_name", "he_path", "grid_x", "grid_y", "panel2_observed_cells", "panel2_coverage", *COUNT_COLUMNS}
    # The current alignment artifact intentionally does not duplicate source image paths.
    if "he_path" not in rows:
        cache = json.loads(Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600/metadata.json").read_text())
        paths = {(x["case_id"], x["patch_name"]): x["he_path"] for x in cache["records"]}
        rows["he_path"] = [paths[(x.case_id, x.patch_name)] for _, x in rows.iterrows()]
    if not required.issubset(rows.columns):
        raise ValueError(f"alignment rows missing {sorted(required-set(rows.columns))}")
    config = load_config(args.phase2_root / "config_snapshot.json")
    device = torch.device(args.device)
    phase2 = DeepRegionEncoder(config, len(config["data"]["class_map"]), 64, 256).to(device)
    phase2.load_state_dict(torch.load(args.phase2_root / "checkpoint_epoch_028.pth", map_location=device, weights_only=False)["model"])
    phase2.eval().requires_grad_(False)
    head = ObservedRegionEvidenceHeads().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    matrix_cache = {}
    examples, losses = [], []
    for case in sorted(rows.case_id.unique()):
        patch_name = rows.loc[rows.case_id == case, "patch_name"].sort_values().iloc[0]
        selected = rows[(rows.case_id == case) & (rows.patch_name == patch_name)]
        origin = tuple(map(float, re.fullmatch(r"x(\d+)_y(\d+)", patch_name).groups()))
        if case not in matrix_cache:
            manifest = Path("/nfs-medical6/data/SRRSH/crc/panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations") / case / "coordinate_transform_manifest.json"
            matrix_cache[case] = json.loads(manifest.read_text())["panel2_mif_dapi_to_panel1_he_level0_composed"]
        image = tensor_image(selected.he_path.iloc[0]).to(device)
        with torch.no_grad():
            value = phase2(image, return_full_assignment=False, return_tokens=True)
        assignment = value["assignment_low"][0]
        pixel_grid = patch_pixel_grid_indices(assignment.shape[-2], assignment.shape[-1], origin, matrix_cache[case], .2457, 1024., source_pixel_stride=image.shape[-1] / assignment.shape[-1])
        projection = project_assignment_to_grid(assignment, pixel_grid)
        target = project_observed_targets(projection, targets_for_patch(selected), min_token_grid_fraction=args.min_token_grid_fraction)
        logits = head(value["region_tokens"])["immune_observed_logits"][0]
        loss, supervised = observed_composition_loss(logits, target["immune_counts"], target["immune_observed_cells"], target["immune_coverage"])
        if supervised < 1 or not torch.isfinite(loss):
            raise RuntimeError(f"no valid panel-2 region target for {case}/{patch_name}")
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        examples.append({"case_id": case, "patch_name": patch_name, "assignment_shape": list(assignment.shape), "physical_grids": projection["grid_xy"].tolist(), "supervised_tokens": int(supervised), "max_token_grid_fraction": float(projection["token_grid_fraction"].max()), "coordinate_proxy": "registered H&E patch uses panel-2 DAPI patch origin; local registration residual not elevated to an exact calibration"})
        losses.append(float(loss.detach()))
    report = {"status": "complete", "scope": "real CRC H&E -> frozen Phase-2 token -> panel-2 observed composition head smoke; not a training or segmentation result", "examples": examples, "losses": losses, "min_token_grid_fraction": args.min_token_grid_fraction, "candidate_only": True}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    torch.save({"head": head.state_dict(), "diagnostic_only": True}, out / "diagnostic_head.pth")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
