"""Frozen J5 loading and spatial episodes for boundary experiments.

Utilities copied from the existing uncommitted iterative experiment; no v4
model source is modified. Baseline checkpoints load strictly.
"""
from __future__ import annotations
import argparse
import functools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import JointPixelEpisodeDataset, collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import JointPromptMaskModel, load_joint_components

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
