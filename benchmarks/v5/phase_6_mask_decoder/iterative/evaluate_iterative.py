"""Reload a v5 iterative checkpoint and run paired spatial validation steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .iterative_decoder import IterativePromptDecoder
from .train_iterative import load_baseline, make_loader, move_batch, run_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--iterative-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-data", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--historical-source", choices=("train", "val"))
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1 or args.rounds < 1:
        raise ValueError("output must be new and sample/round counts positive")
    args.output.mkdir(parents=True)
    args.val_samples = args.samples
    metadata = json.loads(args.baseline_metadata.read_text())
    device = torch.device(args.device)
    baseline = load_baseline(metadata, args.baseline_checkpoint, device)
    decoder = IterativePromptDecoder(dim=int(metadata["config"]["model"]["region_dim"])).to(device)
    payload = torch.load(args.iterative_checkpoint, map_location=device, weights_only=False)
    decoder.load_state_dict(payload["model"], strict=True)
    decoder.eval()
    rows = []
    with torch.no_grad():
        for batch in make_loader(args, metadata, "val"):
            rows.extend(run_item(decoder, baseline, move_batch(batch, device),
                                 args.rounds, None, int(metadata["config"]["data"]["ignore_index"])))
    with (args.output / "steps.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {"checkpoint": str(args.iterative_checkpoint), "split": "val",
               "historical_source": args.historical_source,
               "status": "exploratory_initialization_exposure", "steps": len(rows),
               "pixel_conflicts": sum(row["conflict"] for row in rows),
               "sign": {name: {
                   "count": len(part),
                   "dice_gain": float(np.mean([row["dice_after"] - row["dice_before"] for row in part])) if part else None,
                   "remote_change": float(np.mean([row["remote_change"] for row in part if row["remote_change"] is not None])) if any(row["remote_change"] is not None for row in part) else None,
                   "protected_change": float(np.mean([row["protected_change"] for row in part if row["protected_change"] is not None])) if any(row["protected_change"] is not None for row in part) else None,
               } for name, part in (("positive", [row for row in rows if row["sign"] == 1]),
                                    ("negative", [row for row in rows if row["sign"] == 0]))}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
