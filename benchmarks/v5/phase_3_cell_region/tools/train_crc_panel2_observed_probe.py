#!/usr/bin/env python3
"""Small, spatially isolated H&E-to-panel-2 observed-composition probe."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v5.phase_3_cell_region.src.observed_region_evidence import ObservedRegionEvidenceHeads
from benchmarks.v5.phase_3_cell_region.src.region_footprint_contract import patch_pixel_grid_indices, project_assignment_to_grid, project_observed_targets
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import observed_composition_loss


def tensor_image(path):
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.
    tensor = torch.from_numpy(value)
    return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def balanced_limit(rows, split, per_case):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row["case_id"]].append(row)
    selected = []
    for case in sorted(grouped):
        choices = sorted(grouped[case], key=lambda x: x["sample_id"])
        if len(choices) < per_case:
            raise ValueError(f"{split}/{case} has only {len(choices)} samples, needs {per_case}")
        selected.extend(choices[:per_case])
    return selected


def load_cache_paths():
    source = Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0916_105600/metadata.json")
    records = json.loads(source.read_text())["records"]
    return {(row["case_id"], row["patch_name"]): row["he_path"] for row in records}


def cache_split(rows, phase2, device, matrices):
    source_paths = load_cache_paths(); result = []
    for row in rows:
        case, patch = row["case_id"], row["sample_id"].split(":", 1)[1]
        match = re.fullmatch(r"x(\d+)_y(\d+)", patch)
        origin = tuple(map(float, match.groups()))
        image = tensor_image(source_paths[(case, patch)]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = phase2(image, return_full_assignment=False, return_tokens=True)
        assignment = output["assignment_low"][0]
        pixel_grid = patch_pixel_grid_indices(assignment.shape[-2], assignment.shape[-1], origin, matrices[case], .2457, 1024., source_pixel_stride=image.shape[-1] / assignment.shape[-1])
        projection = project_assignment_to_grid(assignment, pixel_grid)
        targets = {(int(x["grid_x"]), int(x["grid_y"])): (x["counts"], x["observed_cells"], x["coverage"]) for x in row["panel2_targets"]}
        target = project_observed_targets(projection, targets, min_token_grid_fraction=.8)
        if not target["immune_target_valid"].any():
            raise RuntimeError(f"no supervised token for {row['sample_id']}")
        result.append({"tokens": output["region_tokens"][0].detach().cpu(), **{key: value.detach().cpu() for key, value in target.items()}, "case_id": case, "sample_id": row["sample_id"]})
    return result


def evaluate(head, items, device):
    head.eval(); total, weight, count = 0., 0., 0
    with torch.no_grad():
        for item in items:
            logits = head(item["tokens"].unsqueeze(0).to(device))["immune_observed_logits"][0]
            loss, valid = observed_composition_loss(logits, item["immune_counts"].to(device), item["immune_observed_cells"].to(device), item["immune_coverage"].to(device))
            if valid:
                total += float(loss) * valid; weight += valid; count += valid
    return total / max(weight, 1), int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_spatial_manifest_0921_173523/manifest.json"))
    parser.add_argument("--phase2-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--device", default="cuda:1"); parser.add_argument("--train-per-case", type=int, default=32); parser.add_argument("--val-per-case", type=int, default=10); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or min(args.train_per_case, args.val_per_case, args.epochs) < 1:
        raise ValueError("timestamp and positive budgets required")
    torch.manual_seed(args.seed); device = torch.device(args.device); out = args.output_root / f"crc_panel2_observed_probe_{args.timestamp}"; out.mkdir(parents=True, exist_ok=False); spec = json.loads(args.manifest.read_text()); train_rows, val_rows = balanced_limit(spec["samples"], "train", args.train_per_case), balanced_limit(spec["samples"], "val", args.val_per_case)
    config = load_config(args.phase2_root / "config_snapshot.json"); phase2 = DeepRegionEncoder(config, len(config["data"]["class_map"]), 64, 256).to(device)
    phase2.load_state_dict(torch.load(args.phase2_root / "checkpoint_epoch_028.pth", map_location=device, weights_only=False)["model"]); phase2.eval().requires_grad_(False)
    common = Path("/nfs-medical6/data/SRRSH/crc/panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations")
    matrices = {case: json.loads((common / case / "coordinate_transform_manifest.json").read_text())["panel2_mif_dapi_to_panel1_he_level0_composed"] for case in {row["case_id"] for row in train_rows + val_rows}}
    train, val = cache_split(train_rows, phase2, device, matrices), cache_split(val_rows, phase2, device, matrices)
    total_counts = sum((item["immune_counts"][item["immune_target_valid"]].sum(0) for item in train), torch.zeros(5)); mean_prob = total_counts / total_counts.sum()
    baseline = ObservedRegionEvidenceHeads().to(device)
    with torch.no_grad(): baseline.immune_head.weight.zero_(); baseline.immune_head.bias.copy_(mean_prob.clamp_min(1e-8).log().to(device))
    baseline_val_loss, valid_val = evaluate(baseline, val, device)
    head = ObservedRegionEvidenceHeads().to(device); optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4); best, history = float("inf"), []
    for epoch in range(args.epochs):
        head.train(); losses = []
        order = torch.randperm(len(train)).tolist()
        for index in order:
            item = train[index]; optimizer.zero_grad(); logits = head(item["tokens"].unsqueeze(0).to(device))["immune_observed_logits"][0]
            loss, valid = observed_composition_loss(logits, item["immune_counts"].to(device), item["immune_observed_cells"].to(device), item["immune_coverage"].to(device))
            if not valid: raise RuntimeError("cached train target unexpectedly invalid")
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach()))
        val_loss, valid = evaluate(head, val, device); history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_valid_tokens": valid})
        if val_loss < best: best = val_loss; torch.save({"head": head.state_dict(), "epoch": epoch + 1, "manifest": str(args.manifest), "diagnostic_only": True}, args.output_root / f"crc_panel2_observed_probe_{args.timestamp}" / "best_head.pth")
    report = {"status": "complete", "scope": "frozen Phase-2 H&E token panel-2 observed-composition probe; not segmentation evaluation", "candidate_only": True, "train_samples": len(train), "val_samples": len(val), "train_cases": sorted({x['case_id'] for x in train}), "val_cases": sorted({x['case_id'] for x in val}), "baseline_train_only_mean_val_loss": baseline_val_loss, "best_val_loss": best, "improves_train_only_mean_baseline": bool(best < baseline_val_loss), "history": history}
    (out / "metrics.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
