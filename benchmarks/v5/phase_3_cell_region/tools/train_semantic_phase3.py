#!/usr/bin/env python3
"""Train the first-stage identity or region-fusion adapter on verified contracts."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_3_cell_region.src.semantic_cell_region import SemanticCellToRegion
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import SemanticDataset, sha256_file
from benchmarks.v5.phase_3_cell_region.src.semantic_training import evaluate, training_loss


def effective_number_weights(counts, beta, cap):
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or (values <= 0).any() or not 0 <= beta < 1 or cap < 1:
        raise ValueError("positive train-only class counts, 0<=beta<1, and cap>=1 are required")
    weights = (1 - beta) / (1 - torch.pow(torch.tensor(beta, dtype=torch.float64), values))
    weights = weights / weights.mean()
    return weights.clamp(max=cap).float()


def train_identity_counts(dataset):
    counts = torch.zeros(len(dataset.spec["identity_classes"]), dtype=torch.long)
    for item in dataset:
        allowed, valid = item["targets"]["identity_allowed"][0], item["targets"]["identity_valid"][0] & item["inputs"]["cell_valid"][0]
        exact = valid & (allowed.sum(-1) == 1)
        counts += allowed[exact].long().sum(0)
    if (counts == 0).any():
        raise ValueError("cannot balance a class absent from the train split")
    return counts


def identity_patch_sampling_weights(dataset, class_weights):
    """Train-only patch probabilities; each present rare class increases a patch's chance once."""
    values = []
    for item in dataset:
        allowed, valid = item["targets"]["identity_allowed"][0], item["targets"]["identity_valid"][0] & item["inputs"]["cell_valid"][0]
        present = allowed[valid & (allowed.sum(-1) == 1)].any(0)
        values.append(float(class_weights[present].sum()))
    weights = torch.tensor(values, dtype=torch.float64)
    if len(weights) != len(dataset) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("identity patch sampling requires supervised labels in every train patch")
    return (weights / weights.mean()).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--identity-checkpoint", type=Path)
    parser.add_argument("--stage", choices=("identity", "fusion"), required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--identity-balance", choices=("none", "effective_num"), default="none")
    parser.add_argument("--effective-num-beta", type=float, default=.999)
    parser.add_argument("--max-class-weight", type=float, default=5.)
    parser.add_argument("--identity-patch-sampling", choices=("uniform", "effective_num"), default="uniform")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or args.epochs < 1 or args.patience < 1:
        raise ValueError("timestamp MMDD_HHMMSS and positive budgets required")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("generated outputs must stay in FewClick artifact root")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, val = SemanticDataset(args.manifest, "train"), SemanticDataset(args.manifest, "val")
    # Preflight every record, including label shapes/provenance, before creating outputs.
    for dataset in (train, val):
        for _ in dataset:
            pass
    device = torch.device(args.device)
    identity_weights = None
    identity_counts = None
    patch_weights = None
    if args.stage == "identity" and args.identity_balance == "effective_num":
        identity_counts = train_identity_counts(train)
        identity_weights = effective_number_weights(identity_counts, args.effective_num_beta, args.max_class_weight).to(device)
        if args.identity_patch_sampling == "effective_num":
            patch_weights = identity_patch_sampling_weights(train, identity_weights.cpu())
    elif args.identity_patch_sampling != "uniform":
        raise ValueError("effective_num patch sampling requires effective_num identity balance")
    model = SemanticCellToRegion(len(train.spec["identity_classes"]), len(train.spec["tissue_classes"])).to(device)
    baseline = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    model.baseline.load_state_dict(baseline["cell"], strict=True)
    if args.stage == "fusion":
        if args.identity_checkpoint is None:
            raise ValueError("fusion requires a trained identity checkpoint, not random identities")
        checkpoint = torch.load(args.identity_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["stage"] != "identity" or checkpoint["manifest_sha256"] != train.manifest_hash:
            raise ValueError("identity checkpoint must use this verified split manifest")
        model.identity.load_state_dict({key.removeprefix("identity."): value for key, value in checkpoint["model"].items() if key.startswith("identity.")}, strict=True)
    model.set_stage(args.stage)
    lr = 1e-3 if args.stage == "identity" else 3e-4
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    output = args.output_root / f"{args.stage}_{args.timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": model.schema_version, "manifest_sha256": train.manifest_hash,
                "baseline_sha256": sha256_file(args.baseline_checkpoint), "stage": args.stage,
                "identity_classes": train.spec["identity_classes"], "tissue_classes": train.spec["tissue_classes"],
                "identity_balance": args.identity_balance,
                "identity_train_class_counts": identity_counts.tolist() if identity_counts is not None else None,
                "identity_class_weights": identity_weights.cpu().tolist() if identity_weights is not None else None,
                "identity_patch_sampling": args.identity_patch_sampling,
                "identity_patch_sampling_weights": patch_weights,
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "calibration": "temperature=1; fitting calibration remains a separate development-set step"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    groups = {}
    for idx, row in enumerate(train.rows):
        groups.setdefault(row["case_id"], []).append(idx)
    best, stale = -1., 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, steps, supervised = 0., 0, 0
        # One patch per optimizer step; sample cases uniformly, then patches.
        for _ in range(len(train)):
            case = rng.choice(sorted(groups))
            indices = groups[case]
            if patch_weights is None:
                selected = rng.choice(indices)
            else:
                selected = rng.choices(indices, weights=[patch_weights[index] for index in indices], k=1)[0]
            item = train[selected]
            inputs = {key: value.to(device) for key, value in item["inputs"].items()}
            targets = {key: value.to(device) for key, value in item["targets"].items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**inputs)
            loss, count = training_loss(model, result, targets, inputs["cell_valid"], identity_weights)
            if not count:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
            supervised += count
            steps += 1
        if not steps:
            raise RuntimeError("no valid supervision; do not train on invented labels")
        metrics = evaluate(model, val, device)
        score = metrics["patient_macro_f1"]
        if score is None:
            raise RuntimeError("validation has no supported classification labels")
        record = {"epoch": epoch, "loss": loss_sum / steps, "steps": steps, "supervised": supervised, **metrics}
        history.append(record)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if score > best:
            best, stale = score, 0
            # Keep every best candidate in its own immutable file.
            torch.save({**metadata, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, output / f"best_epoch_{epoch:03d}.pth")
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "loss": record["loss"], "patient_macro_f1": score}), flush=True)
        if stale >= args.patience:
            break
    (output / "complete.json").write_text(json.dumps({"status": "complete", "best_patient_macro_f1": best, "epochs": len(history), "final_segmentation_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
