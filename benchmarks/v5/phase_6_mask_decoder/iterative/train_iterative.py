"""Exploratory spatial-split training of the v5 iterative decoder.

The frozen v4 initialization has historical exposure to some new spatial
validation patches. Metrics from this script are developmental, not a sealed
generalization estimate.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import JointPixelEpisodeDataset, collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import JointPromptMaskModel, load_joint_components
from .episodes import append_click, correction_loss, select_correction_click
from .iterative_decoder import IterativePromptDecoder, click_values


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-data", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def load_baseline(metadata: dict, checkpoint: Path, device: torch.device) -> JointPromptMaskModel:
    cfg = metadata["config"]
    paths = metadata["inputs"]
    p2cfg = load_config(Path(paths["phase2_config"]))
    p5cfg = load_config(Path(paths["phase5_config"]))
    phase2, cell, prompt, _ = load_joint_components(
        p2cfg, paths["phase2_checkpoint"], paths["cell_checkpoint"],
        paths["phase5_checkpoint"], p5cfg, int(cfg["data"]["cell_feature_dim"]),
    )
    mc = cfg["model"]
    model = JointPromptMaskModel(
        phase2, cell, prompt, region_dim=int(mc["region_dim"]), graph_heads=int(mc["graph_heads"]),
        graph_layers=int(mc["graph_layers"]), graph_neighbours=int(mc["graph_neighbours"]),
        graph_dropout=float(mc["graph_dropout"]), residual_limit=float(mc["residual_limit"]),
        train_phase2_embedding=False, train_phase2_assignment=False, train_cell=False,
        train_prompt=False, train_decoder=False, train_backbone_layer4=False,
        cross_scale_variant=str(mc.get("cross_scale_variant", "fine_only")),
        cross_scale_blocks=int(mc.get("cross_scale_blocks", 1)),
        cross_scale_top_k=int(mc.get("cross_scale_top_k", 4)),
        use_cell_branch=bool(mc.get("use_cell_branch", True)),
        regionization_variant=str(mc.get("regionization_variant", "learned")),
        prompt_set_variant=str(mc.get("prompt_set_variant", "set_encoder")),
        num_classes=len(p2cfg["data"]["class_map"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model


def make_loader(args: argparse.Namespace, metadata: dict, split: str) -> DataLoader:
    cfg = metadata["config"]
    path = args.spatial_data
    episodes = PromptEpisodeDataset(
        path / "cache_index.parquet", path / "labels_index.parquet",
        path / "patches_index.parquet", split, seed=args.seed,
        size_probabilities=cfg["data"]["size_probabilities"],
        target_class_ids=tuple(cfg["data"]["class_ids"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
        centroid_knn=int(cfg["data"]["centroid_knn"]),
        eligibility_index=path / "eligibility_index.parquet",
    )
    p2cfg = load_config(Path(metadata["inputs"]["phase2_config"]))
    dataset = JointPixelEpisodeDataset(
        episodes, path / "patches_index.parquet", args.routing / f"cell_routing_{split}.parquet",
        p2cfg["data"]["class_map"], int(cfg["data"]["ignore_index"]),
        int(cfg["data"]["max_cells_per_patch"]),
        include_multiscale=str(cfg["model"].get("cross_scale_variant", "fine_only")) != "fine_only",
    )
    candidates = np.arange(len(dataset))
    historical_source = getattr(args, "historical_source", None)
    if historical_source is not None:
        manifest = pd.read_parquet(path / "spatial_manifest.parquet")
        allowed = set(manifest.loc[(manifest.split == split) &
                                   (manifest.historical_split == historical_source), "patch_id"].astype(str))
        candidates = np.asarray([index for index, patch_id in enumerate(episodes.patch_ids)
                                 if str(patch_id) in allowed], dtype=np.int64)
    count = min(len(candidates), args.train_samples if split == "train" else args.val_samples)
    if count < 1:
        raise ValueError(f"no {split} episodes")
    indices = np.random.default_rng(args.seed + (0 if split == "train" else 1)).choice(candidates, count, replace=False)
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=1, num_workers=0,
                      collate_fn=functools.partial(collate_joint_pixel_episodes,
                                                   max_cells=int(cfg["data"]["max_cells_per_patch"])))


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def initial_history(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = batch["positive_xy"][:, batch["positive_mask"][0]]
    negative = batch["negative_xy"][:, batch["negative_mask"][0]]
    xy = torch.cat((positive, negative), 1)
    signs = torch.cat((torch.ones_like(positive[..., 0], dtype=torch.long),
                       torch.zeros_like(negative[..., 0], dtype=torch.long)), 1)
    valid = torch.ones_like(signs, dtype=torch.bool)
    return xy, signs, valid


def run_item(decoder: IterativePromptDecoder, baseline: JointPromptMaskModel,
             batch: dict, rounds: int, optimizer: torch.optim.Optimizer | None,
             ignore_index: int) -> list[dict]:
    with torch.no_grad():
        base = baseline(batch)
    assignment = base["assignment"].detach()
    region = base["online_contextual_tokens"].detach()
    xy, signs, valid_prompt = initial_history(batch)
    prior_logits = base["logits"].detach()
    prior_probability = base["pixel_probability"].detach()
    truth = batch["pixel_gt"] == batch["target_class"][:, None, None]
    valid_pixels = batch["pixel_gt"] != ignore_index
    records: list[dict] = []
    losses = []
    for round_index in range(rounds):
        preferred = round_index % 2
        try:
            new_xy, new_sign = select_correction_click(prior_probability, truth, valid_pixels, preferred)
        except ValueError:
            try:
                new_xy, new_sign = select_correction_click(prior_probability, truth, valid_pixels, 1 - preferred)
            except ValueError:
                break
        packed = append_click(xy, signs, valid_prompt, new_xy, new_sign, region, assignment)
        output = decoder(
            region_tokens=region, region_xy=base["online_region_xy"].detach(),
            region_area=base["online_region_area"].detach(), active=batch["fine_active"],
            previous_logits=prior_logits, assignment=assignment, **packed,
        )
        loss, terms = correction_loss(output, prior_probability, truth, valid_pixels,
                                      packed["prompt_xy"], packed["prompt_sign"],
                                      packed["prompt_valid"], packed["new_index"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite correction loss")
        if not output["click_conflict"].item():
            losses.append(loss)
        click_after = click_values(output["pixel_probability"], new_xy[:, None])[0, 0]
        click_before = click_values(prior_probability, new_xy[:, None])[0, 0]
        change = output["pixel_probability"].detach() - prior_probability
        correction_area = (truth & (prior_probability < 0.5) if new_sign.item() == 1
                           else ~truth & (prior_probability >= 0.5)) & valid_pixels
        protected_area = (~truth & (prior_probability < 0.5) if new_sign.item() == 1
                          else truth & (prior_probability >= 0.5)) & valid_pixels
        remote_change = float(change[correction_area].mean()) if correction_area.any() else None
        protected_change = float(change[protected_area].mean()) if protected_area.any() else None
        region_change = output["region_pixel_probability"].detach() - prior_probability
        protected_region_change = float(region_change[protected_area].mean()) if protected_area.any() else None
        protected_anchor_change = (
            float((output["pixel_probability"].detach() - output["region_pixel_probability"].detach())[protected_area].mean())
            if protected_area.any() else None
        )
        weights = assignment.float() / assignment.float().sum(1, keepdim=True).clamp_min(1e-8)
        evidence_pixel = torch.einsum("bkhw,bk->bhw", weights, output["signed_evidence"].detach())
        error_evidence = float(evidence_pixel[correction_area].mean()) if correction_area.any() else None
        protected_evidence = float(evidence_pixel[protected_area].mean()) if protected_area.any() else None
        def dice(probability: torch.Tensor) -> float:
            predicted = (probability >= 0.5) & valid_pixels
            target = truth & valid_pixels
            return float((2 * (predicted & target).sum() + 1) /
                         (predicted.sum() + target.sum() + 1))
        records.append({"round": round_index + 1, "sign": int(new_sign.item()),
                        "loss": float(loss.detach()), "click_before": float(click_before),
                        "click_after": float(click_after.detach()),
                        "conflict": bool(output["click_conflict"].item()),
                        "region_conflict": bool(output["region_conflict"].item()),
                        "remote_change": remote_change,
                        "protected_change": protected_change,
                        "protected_region_change": protected_region_change,
                        "protected_anchor_change": protected_anchor_change,
                        "error_evidence": error_evidence,
                        "protected_evidence": protected_evidence,
                        "dice_before": dice(prior_probability),
                        "dice_after": dice(output["pixel_probability"].detach()),
                        "terms": {key: float(value) for key, value in terms.items()}})
        if output["click_conflict"].item():
            break
        xy, signs, valid_prompt = packed["prompt_xy"], packed["prompt_sign"], packed["prompt_valid"]
        prior_logits = output["logits"].detach()
        prior_probability = output["pixel_probability"].detach()
    if optimizer is not None and losses:
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()
    return records


def main() -> None:
    args = args_parser()
    if args.train_samples < 1 or args.val_samples < 1 or args.epochs < 1 or args.rounds < 1:
        raise ValueError("sample, epoch and round counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    metadata = json.loads(args.baseline_metadata.read_text())
    device = torch.device(args.device)
    baseline = load_baseline(metadata, args.baseline_checkpoint, device)
    decoder = IterativePromptDecoder(dim=int(metadata["config"]["model"]["region_dim"])).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loaders = {split: make_loader(args, metadata, split) for split in ("train", "val")}
    run = {"created": datetime.now().isoformat(), "baseline_metadata": str(args.baseline_metadata),
           "baseline_checkpoint": str(args.baseline_checkpoint), "spatial_data": str(args.spatial_data),
           "routing": str(args.routing), "seed": args.seed, "rounds": args.rounds,
           "validation_status": "exploratory: frozen v4 initialization may have seen spatial validation patches"}
    (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    best = float("inf")
    for epoch in range(args.epochs):
        summary = {"epoch": epoch}
        for split in ("train", "val"):
            decoder.train(split == "train")
            rows = []
            for index, batch in enumerate(loaders[split]):
                batch = move_batch(batch, device)
                # Validation still needs gradients for the differentiable projection,
                # but it never updates parameters.
                rows.extend(run_item(decoder, baseline, batch, args.rounds,
                                     optimizer if split == "train" else None,
                                     int(metadata["config"]["data"]["ignore_index"])))
                if (index + 1) % 10 == 0:
                    print(f"{split} epoch={epoch} items={index+1} steps={len(rows)}", flush=True)
            summary[split] = {"items": len(loaders[split]), "steps": len(rows),
                              "mean_loss": float(np.mean([row["loss"] for row in rows])) if rows else None,
                              "click_conflicts": sum(row["conflict"] for row in rows),
                              "region_conflicts": sum(row["region_conflict"] for row in rows),
                              "positive_steps": sum(row["sign"] == 1 for row in rows),
                              "negative_steps": sum(row["sign"] == 0 for row in rows),
                              "by_sign": {name: {
                                  "steps": len(part),
                                  "dice_gain": float(np.mean([row["dice_after"] - row["dice_before"] for row in part])) if part else None,
                                  "remote_change": float(np.mean([row["remote_change"] for row in part if row["remote_change"] is not None])) if any(row["remote_change"] is not None for row in part) else None,
                                  "protected_change": float(np.mean([row["protected_change"] for row in part if row["protected_change"] is not None])) if any(row["protected_change"] is not None for row in part) else None,
                              } for name, part in (("positive", [row for row in rows if row["sign"] == 1]),
                                                   ("negative", [row for row in rows if row["sign"] == 0]))}}
            with (args.output / f"{split}_epoch_{epoch:03d}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        (args.output / f"summary_epoch_{epoch:03d}.json").write_text(json.dumps(summary, indent=2))
        torch.save({"model": decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "summary": summary, "run": run},
                   args.output / f"checkpoint_epoch_{epoch:03d}.pth")
        val_loss = summary["val"]["mean_loss"]
        positive = summary["val"]["by_sign"]["positive"]
        negative = summary["val"]["by_sign"]["negative"]
        quality_checks = {
            "both_signs_present": positive["steps"] > 0 and negative["steps"] > 0,
            "no_click_conflicts": summary["val"]["click_conflicts"] == 0,
            "positive_dice_noninferior": positive["dice_gain"] is not None and positive["dice_gain"] >= -0.001,
            "negative_dice_noninferior": negative["dice_gain"] is not None and negative["dice_gain"] >= -0.001,
            "positive_remote_correct": positive["remote_change"] is not None and positive["remote_change"] > 0,
            "negative_remote_correct": negative["remote_change"] is not None and negative["remote_change"] < 0,
            "old_negative_preserved": positive["protected_change"] is not None and positive["protected_change"] <= 0.01,
            "old_positive_preserved": negative["protected_change"] is not None and negative["protected_change"] >= -0.01,
        }
        (args.output / f"quality_epoch_{epoch:03d}.json").write_text(json.dumps(quality_checks, indent=2))
        if val_loss is not None and all(quality_checks.values()) and val_loss < best:
            best = val_loss
            (args.output / "best_checkpoint.json").write_text(json.dumps(
                {"checkpoint": str(args.output / f"checkpoint_epoch_{epoch:03d}.pth"),
                 "validation_mean_loss": best, "epoch": epoch,
                 "quality_checks": quality_checks}, indent=2))
        print(json.dumps(summary), flush=True)
    (args.output / "status.json").write_text(json.dumps(
        {"status": "complete" if best < float("inf") else "no_eligible_checkpoint",
         "best_val_loss": best if best < float("inf") else None}))


if __name__ == "__main__":
    main()
