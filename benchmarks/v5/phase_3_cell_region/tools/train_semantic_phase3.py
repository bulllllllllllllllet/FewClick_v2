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
            item = train[rng.choice(groups[rng.choice(sorted(groups))])]
            inputs = {key: value.to(device) for key, value in item["inputs"].items()}
            targets = {key: value.to(device) for key, value in item["targets"].items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**inputs)
            loss, count = training_loss(model, result, targets, inputs["cell_valid"])
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
